// SPDX-License-Identifier: GPL-2.0-only
/*
 * zero_copy_dma.c  –  Zero-Copy Scatter-Gather DMA Character Driver
 *
 * PURPOSE
 * ───────
 * This module provides a production-quality, zero-copy DMA pipeline for
 * high-throughput camera / sensor workloads.  User-space processes receive
 * live frame data directly from DMA-mapped kernel pages — never copied
 * through a kernel buffer — achieving the minimum possible CPU overhead.
 *
 * ENVIRONMENT
 * ───────────
 * Designed for Linux v6.1 LTS and v6.6+ LTS kernels running inside a
 * QEMU/KVM virtual machine where no physical PCIe or AXI DMA controller
 * is present.  A "Software Hardware-Mocking Engine" (a kthread) simulates
 * the periodic DMA-complete interrupt that real camera silicon would fire,
 * writing a deterministic test pattern directly into the pre-allocated
 * pages so that user-space can verify zero-copy integrity.
 *
 * ARCHITECTURE OVERVIEW
 * ─────────────────────
 *
 *  ┌──────────────────────────── kernel module ──────────────────────────────┐
 *  │                                                                          │
 *  │   kthread (mock DMA HW)             character device /dev/zero_copy_dma │
 *  │   ─────────────────────             ─────────────────────────────────── │
 *  │   every 33 ms:                      open()   – enforce single consumer  │
 *  │     dma_sync_sg_for_cpu()           release()– stop stream on close     │
 *  │     zcd_fill_pattern()  ──writes──► mmap()   – remap_pfn_range()        │
 *  │     dma_sync_sg_for_device()        ioctl()  – START/STOP/GET/RELEASE   │
 *  │     smp_store_release(write_idx)              QUERY                     │
 *  │     wake_up_interruptible()                                              │
 *  │                                ▲                                        │
 *  │                                │  wait_event_interruptible(frame_wq)    │
 *  └────────────────────────────────┼────────────────────────────────────────┘
 *                                   │
 *  ┌──────────── user-space ────────┼────────────────────────────────────────┐
 *  │   mmap_ptr = mmap(ZCD_TOTAL_MAP_SIZE)                                   │
 *  │   ioctl(ZCD_IOCTL_GET_FRAME, &fi)  ──blocks until frame ready──────────►│
 *  │   pixel = mmap_ptr[fi.index * ZCD_FRAME_SIZE + offset]   (zero copy)   │
 *  │   ioctl(ZCD_IOCTL_RELEASE_FRAME, fi.index)                              │
 *  └────────────────────────────────────────────────────────────────────────┘
 *
 * RING BUFFER LAYOUT (4 × 1 MiB)
 * ───────────────────────────────
 *   slot[0] [1 MiB]  order-8 compound page, physically contiguous
 *   slot[1] [1 MiB]  …
 *   slot[2] [1 MiB]
 *   slot[3] [1 MiB]
 *
 *   User mmap VA window: ZCD_TOTAL_MAP_SIZE = 4 MiB (one contiguous VA range)
 *   Frame i starts at:  mmap_base + i * ZCD_FRAME_SIZE
 *
 * LOCK-FREE SPSC RING
 * ───────────────────
 *   write_idx  – owned by producer (kthread); read by consumer with
 *                smp_load_acquire() to observe up-to-date data.
 *   read_idx   – owned by consumer (ioctl handler); read by producer with
 *                READ_ONCE() to detect ring-full.
 *   Both updated with smp_store_release() to ensure data writes are
 *   visible before the index update crosses to the other CPU.
 *
 * Copyright (C) 2024 Zero-Copy DMA Driver Authors
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/atomic.h>
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/platform_device.h>
#include <linux/spinlock.h>
#include <linux/bug.h>
#include <linux/highmem.h>
#include <linux/compiler.h>

#include "zero_copy_dma.h"

/* -----------------------------------------------------------------------
 * Module metadata
 * ----------------------------------------------------------------------- */

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Zero-Copy DMA Driver Authors");
MODULE_DESCRIPTION("Zero-Copy Scatter-Gather DMA Character Driver (Software Mock)");
MODULE_VERSION("1.0");

/* -----------------------------------------------------------------------
 * Per-slot descriptor
 *
 * One instance per ring buffer position.  Keeps together every kernel
 * resource that belongs to that 1-MiB DMA frame.
 * ----------------------------------------------------------------------- */

/**
 * struct zcd_frame_slot - Kernel-side bookkeeping for one DMA ring slot.
 *
 * @pages:      Pointer to the compound page returned by alloc_pages().
 *              Covers ZCD_PAGES_PER_FRAME physically contiguous 4-KiB pages.
 * @vaddr:      Kernel virtual address of the first byte in this slot
 *              (page_address(pages)).
 * @sgt:        Scatter-gather table built from the individual sub-pages.
 *              Used for dma_map_sg() / dma_unmap_sg() bookkeeping.
 * @orig_nents: Number of entries passed to dma_map_sg() (= ZCD_PAGES_PER_FRAME).
 *              The DMA API requires the *original* count for dma_unmap_sg().
 * @mapped_nents: Number of DMA segments returned by dma_map_sg() (may be
 *              ≤ orig_nents after the IOMMU merges adjacent pages).
 * @dma_mapped: True once dma_map_sg() has succeeded for this slot.
 * @sequence:   Frame sequence number stamped by the mock producer thread.
 */
struct zcd_frame_slot {
	struct page    *pages;
	void           *vaddr;
	struct sg_table sgt;
	int             orig_nents;
	int             mapped_nents;
	bool            dma_mapped;
	u64             sequence;
};

/* -----------------------------------------------------------------------
 * Driver-wide state container
 *
 * One global instance.  Concurrency strategy:
 *
 *   stream_lock  (spinlock_irqsave)
 *       Protects: streaming flag, state field.
 *       Held only for brief flag reads/writes — never across sleeps.
 *
 *   open_count  (atomic_t)
 *       Enforces single-consumer constraint.
 *
 *   write_idx / read_idx  (unsigned int, lock-free SPSC)
 *       write_idx owned by producer kthread.
 *       read_idx  owned by consumer ioctl handler.
 *       Updated with smp_store_release(); loaded with smp_load_acquire()
 *       or READ_ONCE() on the non-owning side.
 *
 *   frame_wq  (wait_queue_head_t)
 *       Producer wakes consumers via wake_up_interruptible() after
 *       advancing write_idx.
 * ----------------------------------------------------------------------- */

/**
 * struct zcd_device - Complete driver state.
 */
struct zcd_device {
	/* ── Character device plumbing ── */
	dev_t              devnum;
	struct cdev        cdev;
	struct class      *cls;
	struct device     *dev;

	/* ── Virtual DMA master device (used only for DMA API calls) ── */
	struct platform_device *pdev;

	/* ── DMA ring buffer ── */
	struct zcd_frame_slot slots[ZCD_RING_DEPTH];

	/* ── Lock-free SPSC ring indices ── */
	unsigned int write_idx;   /* next slot the producer will write */
	unsigned int read_idx;    /* next slot the consumer will read  */

	/* ── Frame-ready wait queue ── */
	wait_queue_head_t frame_wq;

	/* ── Stream control ── */
	spinlock_t  stream_lock;
	bool        streaming;
	int         state;        /* ZCD_STATE_* */

	/* ── Mock hardware thread ── */
	struct task_struct *mock_thread;
	u64                 frame_sequence;

	/* ── Open / close reference count ── */
	atomic_t open_count;
};

/* Single global instance (statically allocated, zeroed at link time). */
static struct zcd_device zcd_dev;

/* -----------------------------------------------------------------------
 * Forward declarations of file-operation handlers
 * ----------------------------------------------------------------------- */

static int  zcd_open(struct inode *inode, struct file *filp);
static int  zcd_release(struct inode *inode, struct file *filp);
static int  zcd_mmap(struct file *filp, struct vm_area_struct *vma);
static long zcd_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);

/* -----------------------------------------------------------------------
 * Test pattern generation
 *
 * Each byte at offset i in slot[s] for frame sequence n is:
 *
 *     byte[i] = ( n ^ s ^ i ) & 0xFF
 *
 * Because n, s, and i are all available to user-space (from the
 * zcd_frame_info struct and the known slot layout), the application can
 * verify every byte without any shared state — proving zero-copy integrity.
 * ----------------------------------------------------------------------- */

/**
 * zcd_fill_pattern() - Write a mock "pixel" pattern into one DMA slot.
 * @slot:     Slot to fill.
 * @slot_idx: Index of @slot within the ring (0 … ZCD_RING_DEPTH-1).
 * @seq:      Frame sequence number.
 *
 * Runs in kthread context (producer side).  The caller must call
 * dma_sync_sg_for_cpu() before this function and dma_sync_sg_for_device()
 * after it so that cache coherency is maintained on non-coherent platforms.
 */
static void zcd_fill_pattern(struct zcd_frame_slot *slot,
			      unsigned int slot_idx, u64 seq)
{
	u8  *buf = (u8 *)slot->vaddr;
	u32  i;

	for (i = 0; i < ZCD_FRAME_SIZE; i++)
		buf[i] = (u8)((seq ^ (u64)slot_idx ^ (u64)i) & 0xFF);
}

/* -----------------------------------------------------------------------
 * SPSC ring buffer helper
 *
 * Returns true when at least one frame is available for the consumer, or
 * when streaming has been stopped (so blocked callers can wake and return
 * an appropriate error).
 * ----------------------------------------------------------------------- */

/**
 * zcd_frame_available() - SPSC availability predicate used in wait_event.
 * @d: Device state.
 *
 * Uses smp_load_acquire() on write_idx to obtain an acquire barrier: any
 * data written by the producer before advancing write_idx is guaranteed to
 * be visible to this CPU after the acquire load.
 *
 * Return: true if a frame is ready or streaming has stopped.
 */
static inline bool zcd_frame_available(struct zcd_device *d)
{
	unsigned int w = smp_load_acquire(&d->write_idx);
	unsigned int r = READ_ONCE(d->read_idx);

	return (w != r) || !READ_ONCE(d->streaming);
}

/* -----------------------------------------------------------------------
 * Mock hardware kthread  (Software Hardware-Mocking Engine)
 *
 * Simulates the interrupt-driven bottom-half of a DMA controller.
 * Every ZCD_MOCK_INTERVAL_MS milliseconds the thread:
 *
 *   1. Verifies streaming is active.
 *   2. Checks ring capacity (overflow protection — drop frame if full).
 *   3. Calls dma_sync_sg_for_cpu() — would flush device cache on real HW.
 *   4. Writes the deterministic test pattern (simulates a DMA burst).
 *   5. Calls dma_sync_sg_for_device() — would flush CPU cache on real HW.
 *   6. Stamps the frame sequence number into the slot descriptor.
 *   7. Advances write_idx with smp_store_release() (release barrier
 *      ensures the data write is visible before the index update).
 *   8. Wakes any user-space thread blocked in ZCD_IOCTL_GET_FRAME.
 * ----------------------------------------------------------------------- */

/**
 * zcd_mock_thread() - kthread simulating a DMA-interrupt source.
 * @data: Pointer to &struct zcd_device (cast from void *).
 *
 * Return: 0 when asked to stop via kthread_stop().
 */
static int zcd_mock_thread(void *data)
{
	struct zcd_device *d = (struct zcd_device *)data;

	pr_info("zcd: mock hardware thread started (pid %d)\n",
		current->pid);

	while (!kthread_should_stop()) {
		unsigned int widx, next_widx, ridx;

		msleep_interruptible(ZCD_MOCK_INTERVAL_MS);

		if (kthread_should_stop())
			break;

		/* Check streaming flag; skip if not active. */
		if (!READ_ONCE(d->streaming))
			continue;

		widx      = READ_ONCE(d->write_idx);
		ridx      = READ_ONCE(d->read_idx);
		next_widx = (widx + 1U) % ZCD_RING_DEPTH;

		/*
		 * Overflow protection.
		 *
		 * The ring is full when advancing write_idx would make it
		 * equal to read_idx.  Drop the frame rather than overwriting
		 * a slot the consumer has not yet read.
		 */
		if (next_widx == ridx) {
			pr_debug_ratelimited(
				"zcd: ring full — dropping frame %llu\n",
				d->frame_sequence);
			d->frame_sequence++;
			continue;
		}

		/*
		 * Simulate DMA-complete cache-coherency maintenance.
		 *
		 * On cache-coherent architectures (x86, QEMU) these are
		 * no-ops.  On ARM/MIPS without hardware cache coherency they
		 * flush the CPU-side caches so the CPU can read DMA data.
		 */
		dma_sync_sg_for_cpu(&d->pdev->dev,
				    d->slots[widx].sgt.sgl,
				    d->slots[widx].orig_nents,
				    DMA_FROM_DEVICE);

		/* Write the test pattern (simulates the DMA burst). */
		zcd_fill_pattern(&d->slots[widx], widx, d->frame_sequence);

		/* Hand the buffer back to the "device" view. */
		dma_sync_sg_for_device(&d->pdev->dev,
				       d->slots[widx].sgt.sgl,
				       d->slots[widx].orig_nents,
				       DMA_FROM_DEVICE);

		/* Stamp the sequence number into the slot descriptor. */
		d->slots[widx].sequence = d->frame_sequence;
		d->frame_sequence++;

		/*
		 * Release barrier: the pattern write and sequence stamp above
		 * must be visible to other CPUs *before* write_idx advances.
		 * smp_store_release() provides this ordering guarantee.
		 */
		smp_store_release(&d->write_idx, next_widx);

		/* Wake any process blocked in ZCD_IOCTL_GET_FRAME. */
		wake_up_interruptible(&d->frame_wq);
	}

	pr_info("zcd: mock hardware thread stopped\n");
	return 0;
}

/* -----------------------------------------------------------------------
 * Scatter-gather table management
 *
 * Each ring slot owns one SG table.  The table is constructed from the
 * individual 4-KiB sub-pages of the compound allocation and then mapped
 * into the DMA address space of our virtual platform device.  This
 * mirrors exactly what a real driver does before programming a hardware
 * DMA descriptor chain.
 * ----------------------------------------------------------------------- */

/**
 * zcd_build_sg_table() - Allocate, populate, and DMA-map the SG table for
 *                         one ring slot.
 * @d:    Device state.
 * @slot: Frame slot to initialise.
 *
 * Return: 0 on success, negative errno on failure.
 */
static int zcd_build_sg_table(struct zcd_device *d, struct zcd_frame_slot *slot)
{
	struct scatterlist *sg;
	int i, ret;

	BUILD_BUG_ON(ZCD_PAGES_PER_FRAME == 0);

	/* Allocate ZCD_PAGES_PER_FRAME SG entries. */
	ret = sg_alloc_table(&slot->sgt, ZCD_PAGES_PER_FRAME, GFP_KERNEL);
	if (ret) {
		pr_err("zcd: sg_alloc_table failed (%d)\n", ret);
		return ret;
	}

	/*
	 * Populate: one SG entry per 4-KiB sub-page of the compound
	 * allocation.  nth_page() returns the i-th struct page within the
	 * compound page's physically contiguous extent.
	 */
	i  = 0;
	for_each_sg(slot->sgt.sgl, sg, ZCD_PAGES_PER_FRAME, i)
		sg_set_page(sg, nth_page(slot->pages, i), PAGE_SIZE, 0);

	slot->orig_nents = ZCD_PAGES_PER_FRAME;

	/*
	 * Map the SG list for bidirectional access.
	 *
	 * On a real system this programs the IOMMU to allow the DMA
	 * controller to read/write these pages via bus addresses.  On
	 * QEMU with an identity-mapped platform device it returns the
	 * physical addresses unchanged.
	 *
	 * We pass DMA_BIDIRECTIONAL because:
	 *   - The mock producer (CPU kthread) writes into the pages.
	 *   - User-space reads the pages through the mmap window.
	 */
	ret = dma_map_sg(&d->pdev->dev, slot->sgt.sgl,
			 slot->orig_nents, DMA_BIDIRECTIONAL);
	if (ret == 0) {
		pr_err("zcd: dma_map_sg returned 0 — mapping failed\n");
		sg_free_table(&slot->sgt);
		return -EIO;
	}
	slot->mapped_nents = ret;
	slot->dma_mapped   = true;

	pr_debug("zcd: SG table built: %d input pages → %d DMA segments\n",
		 slot->orig_nents, slot->mapped_nents);
	return 0;
}

/**
 * zcd_destroy_sg_table() - DMA-unmap and free the SG table for one slot.
 * @d:    Device state.
 * @slot: Frame slot to clean up.
 *
 * Safe to call even if zcd_build_sg_table() was never called (idempotent
 * due to the dma_mapped guard).
 */
static void zcd_destroy_sg_table(struct zcd_device *d,
				  struct zcd_frame_slot *slot)
{
	if (slot->dma_mapped) {
		/*
		 * dma_unmap_sg() requires the *original* entry count, not
		 * the post-merge mapped_nents returned by dma_map_sg().
		 */
		dma_unmap_sg(&d->pdev->dev, slot->sgt.sgl,
			     slot->orig_nents, DMA_BIDIRECTIONAL);
		slot->dma_mapped = false;
	}

	if (slot->sgt.sgl)
		sg_free_table(&slot->sgt);
}

/* -----------------------------------------------------------------------
 * Ring buffer memory allocation / deallocation
 * ----------------------------------------------------------------------- */

/**
 * zcd_alloc_buffers() - Allocate page-aligned DMA ring buffers.
 * @d: Device state.
 *
 * For each of the ZCD_RING_DEPTH ring slots:
 *   1. Calls alloc_pages(GFP_KERNEL | __GFP_ZERO, ZCD_PAGES_ORDER) to
 *      obtain a physically contiguous, zero-initialised 1-MiB compound
 *      page.
 *   2. Records the kernel virtual address via page_address().
 *   3. Builds and DMA-maps the scatter-gather table.
 *
 * On any failure the already-allocated slots are freed before returning.
 *
 * Return: 0 on success, negative errno on failure.
 */
static int zcd_alloc_buffers(struct zcd_device *d)
{
	int i, ret;

	for (i = 0; i < ZCD_RING_DEPTH; i++) {
		struct zcd_frame_slot *slot = &d->slots[i];

		/*
		 * Allocate 2^ZCD_PAGES_ORDER = 256 physically contiguous
		 * 4-KiB pages (= 1 MiB).  __GFP_ZERO ensures the buffer
		 * contains no stale kernel data before first use.
		 */
		slot->pages = alloc_pages(GFP_KERNEL | __GFP_ZERO,
					  ZCD_PAGES_ORDER);
		if (!slot->pages) {
			pr_err("zcd: alloc_pages(order=%u) failed for slot %d\n",
			       ZCD_PAGES_ORDER, i);
			ret = -ENOMEM;
			goto err_free_partial;
		}

		slot->vaddr      = page_address(slot->pages);
		slot->dma_mapped = false;
		slot->sequence   = 0;

		ret = zcd_build_sg_table(d, slot);
		if (ret)
			goto err_free_partial;

		pr_info("zcd: slot[%d] phys=0x%llx kva=%p size=%u KiB\n",
			i,
			(unsigned long long)page_to_phys(slot->pages),
			slot->vaddr,
			ZCD_FRAME_SIZE / 1024U);
	}

	return 0;

err_free_partial:
	/*
	 * Unwind: release every slot that was successfully set up.
	 * If slot i failed, slots 0 … i-1 need cleanup; slot i may have
	 * had its pages allocated but SG table not yet built, so the guard
	 * inside zcd_destroy_sg_table() handles that transparently.
	 */
	while (--i >= 0) {
		zcd_destroy_sg_table(d, &d->slots[i]);
		__free_pages(d->slots[i].pages, ZCD_PAGES_ORDER);
		d->slots[i].pages = NULL;
		d->slots[i].vaddr = NULL;
	}
	return ret;
}

/**
 * zcd_free_buffers() - Release all ring buffer resources.
 * @d: Device state.
 *
 * Called from zcd_exit() (or from zcd_init() error paths).  Safe to call
 * with partially-initialised state.
 */
static void zcd_free_buffers(struct zcd_device *d)
{
	int i;

	for (i = ZCD_RING_DEPTH - 1; i >= 0; i--) {
		if (!d->slots[i].pages)
			continue;

		zcd_destroy_sg_table(d, &d->slots[i]);
		__free_pages(d->slots[i].pages, ZCD_PAGES_ORDER);
		d->slots[i].pages = NULL;
		d->slots[i].vaddr = NULL;
	}
}

/* -----------------------------------------------------------------------
 * File operations
 * ----------------------------------------------------------------------- */

/**
 * zcd_open() - Handle open(2) on /dev/zero_copy_dma.
 *
 * Enforces a single-consumer constraint: if a second process tries to open
 * the device while one is already using it, -EBUSY is returned.
 */
static int zcd_open(struct inode *inode, struct file *filp)
{
	if (atomic_inc_return(&zcd_dev.open_count) > 1) {
		atomic_dec(&zcd_dev.open_count);
		pr_warn("zcd: device busy — only one consumer allowed\n");
		return -EBUSY;
	}

	filp->private_data = &zcd_dev;
	pr_info("zcd: opened by pid %d\n", current->pid);
	return 0;
}

/**
 * zcd_release() - Handle close(2) on /dev/zero_copy_dma.
 *
 * Guarantees the streaming flag is cleared so the kthread stops producing
 * new frames, then decrements the open counter.
 */
static int zcd_release(struct inode *inode, struct file *filp)
{
	struct zcd_device *d = filp->private_data;
	unsigned long flags;

	spin_lock_irqsave(&d->stream_lock, flags);
	d->streaming = false;
	d->state     = ZCD_STATE_IDLE;
	spin_unlock_irqrestore(&d->stream_lock, flags);

	/* Wake any blocked ZCD_IOCTL_GET_FRAME so it can return -ECANCELED. */
	wake_up_interruptible(&d->frame_wq);

	atomic_dec(&d->open_count);
	pr_info("zcd: released by pid %d\n", current->pid);
	return 0;
}

/**
 * zcd_mmap() - Map all DMA ring slots into the calling process's VA space.
 *
 * The user process must request exactly ZCD_TOTAL_MAP_SIZE bytes with
 * a zero page offset.  The mapping covers all ZCD_RING_DEPTH slots laid
 * out contiguously:
 *
 *   [mmap_base + 0 * ZCD_FRAME_SIZE]  →  slot[0]
 *   [mmap_base + 1 * ZCD_FRAME_SIZE]  →  slot[1]
 *   …
 *
 * Each slot's pages are remapped individually with remap_pfn_range() so
 * that the physical pages are never copied — this is the zero-copy path.
 *
 * VM flags applied:
 *   VM_DONTEXPAND – prevents mremap(2) from growing the mapping (the SG
 *                   table is sized at mmap time and cannot be extended).
 *   VM_DONTDUMP   – excludes the region from core dumps (sensor data may
 *                   be proprietary or large enough to make dumps useless).
 *
 * Return: 0 on success, negative errno on failure.
 */
static int zcd_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct zcd_device *d = filp->private_data;
	unsigned long vma_size = vma->vm_end - vma->vm_start;
	unsigned long uaddr = vma->vm_start;
	int i, j, ret;

	if (vma_size != (unsigned long)ZCD_TOTAL_MAP_SIZE) {
		pr_err("zcd: mmap size mismatch: got %lu, expected %lu\n",
		       vma_size, (unsigned long)ZCD_TOTAL_MAP_SIZE);
		return -EINVAL;
	}

	/* Disallow offset-based partial mappings for simplicity. */
	if (vma->vm_pgoff != 0)
		return -EINVAL;

	/*
	 * VM_DONTEXPAND: prevents enlarging via mremap(2).
	 * VM_DONTDUMP  : excludes from core dumps.
	 *
	 * Kernel 6.3 made vm_flags immutable through direct assignment and
	 * introduced vm_flags_set().  Use the appropriate API.
	 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 3, 0)
	vm_flags_set(vma, VM_DONTEXPAND | VM_DONTDUMP);
#else
	vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
#endif

	/*
	 * For real hardware with a non-cache-coherent DMA controller,
	 * mark pages as write-combining to avoid stale CPU cache entries:
	 *
	 *   vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	 *
	 * On QEMU/x86 the platform is cache-coherent so we use the default
	 * page protection to avoid the WC performance penalty in emulation.
	 */

	/* Remap each ring slot page-by-page into the VMA. */
	for (i = 0; i < ZCD_RING_DEPTH; i++) {
		struct zcd_frame_slot *slot = &d->slots[i];

		for (j = 0; j < ZCD_PAGES_PER_FRAME; j++) {
			unsigned long pfn = page_to_pfn(nth_page(slot->pages, j));

			ret = remap_pfn_range(vma, uaddr, pfn, PAGE_SIZE,
					      vma->vm_page_prot);
			if (ret) {
				pr_err("zcd: remap_pfn_range failed "
				       "(slot=%d page=%d err=%d)\n",
				       i, j, ret);
				return ret;
			}
			uaddr += PAGE_SIZE;
		}
	}

	pr_info("zcd: mmap: %lu KiB mapped at user VA 0x%lx\n",
		vma_size / 1024UL, vma->vm_start);
	return 0;
}

/**
 * zcd_ioctl() - Handle all ioctl(2) commands.
 *
 * ZCD_IOCTL_START_STREAM
 *   Atomically set streaming = true and reset ring indices.  Returns
 *   -EALREADY if the stream is already running.
 *
 * ZCD_IOCTL_STOP_STREAM
 *   Set streaming = false; wake any blocked GET_FRAME callers.
 *
 * ZCD_IOCTL_GET_FRAME
 *   Block (interruptibly) until write_idx != read_idx, then copy a
 *   struct zcd_frame_info to user-space.  Returns -EINTR on signal,
 *   -ECANCELED when streaming was stopped with no frame available.
 *
 * ZCD_IOCTL_RELEASE_FRAME
 *   Advance read_idx so the producer can reuse the slot.  The caller
 *   must pass back the exact index received from GET_FRAME.
 *
 * ZCD_IOCTL_QUERY_STATUS
 *   Copy the current ZCD_STATE_* value to user-space.  Non-blocking.
 *
 * Return: 0 on success, negative errno on failure.
 */
static long zcd_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct zcd_device *d = filp->private_data;
	unsigned long flags;
	int ret = 0;

	switch (cmd) {

	/* ── START_STREAM ─────────────────────────────────────────── */
	case ZCD_IOCTL_START_STREAM: {
		spin_lock_irqsave(&d->stream_lock, flags);
		if (d->streaming) {
			spin_unlock_irqrestore(&d->stream_lock, flags);
			return -EALREADY;
		}
		/* Reset ring and sequence counter before enabling. */
		WRITE_ONCE(d->write_idx, 0U);
		WRITE_ONCE(d->read_idx, 0U);
		d->frame_sequence = 0ULL;
		d->streaming      = true;
		d->state          = ZCD_STATE_STREAMING;
		spin_unlock_irqrestore(&d->stream_lock, flags);
		pr_info("zcd: stream started\n");
		break;
	}

	/* ── STOP_STREAM ──────────────────────────────────────────── */
	case ZCD_IOCTL_STOP_STREAM: {
		spin_lock_irqsave(&d->stream_lock, flags);
		d->streaming = false;
		d->state     = ZCD_STATE_IDLE;
		spin_unlock_irqrestore(&d->stream_lock, flags);
		wake_up_interruptible(&d->frame_wq);
		pr_info("zcd: stream stopped\n");
		break;
	}

	/* ── GET_FRAME ────────────────────────────────────────────── */
	case ZCD_IOCTL_GET_FRAME: {
		struct zcd_frame_info fi;
		unsigned int ridx, widx;

		/*
		 * Sleep until the producer has advanced write_idx past
		 * read_idx (a new frame is ready) or streaming stops.
		 * wait_event_interruptible() re-evaluates the condition
		 * after each wake_up call and handles signal delivery.
		 */
		ret = wait_event_interruptible(d->frame_wq,
					       zcd_frame_available(d));
		if (ret == -ERESTARTSYS)
			return -EINTR;

		ridx = READ_ONCE(d->read_idx);
		widx = smp_load_acquire(&d->write_idx);

		if (widx == ridx) {
			/* Stream was stopped; no frame is available. */
			return -ECANCELED;
		}

		fi.sequence   = d->slots[ridx].sequence;
		fi.index      = (u32)ridx;
		fi.size_bytes = (u32)ZCD_FRAME_SIZE;

		if (copy_to_user((void __user *)arg, &fi, sizeof(fi)))
			return -EFAULT;
		break;
	}

	/* ── RELEASE_FRAME ────────────────────────────────────────── */
	case ZCD_IOCTL_RELEASE_FRAME: {
		int frame_idx;
		unsigned int ridx;

		if (copy_from_user(&frame_idx, (const void __user *)arg,
				   sizeof(frame_idx)))
			return -EFAULT;

		if (frame_idx < 0 || (unsigned int)frame_idx >= ZCD_RING_DEPTH)
			return -EINVAL;

		ridx = READ_ONCE(d->read_idx);
		if ((unsigned int)frame_idx != ridx) {
			pr_warn("zcd: RELEASE_FRAME: got index %d, "
				"expected %u\n", frame_idx, ridx);
			return -EINVAL;
		}

		/*
		 * Release barrier: ensures the consumer's read of the frame
		 * data is complete before the producer sees the updated
		 * read_idx and potentially overwrites the slot.
		 */
		smp_store_release(&d->read_idx,
				  (ridx + 1U) % ZCD_RING_DEPTH);
		break;
	}

	/* ── QUERY_STATUS ─────────────────────────────────────────── */
	case ZCD_IOCTL_QUERY_STATUS: {
		int state_copy;

		spin_lock_irqsave(&d->stream_lock, flags);
		state_copy = d->state;
		spin_unlock_irqrestore(&d->stream_lock, flags);

		if (copy_to_user((void __user *)arg, &state_copy,
				 sizeof(state_copy)))
			return -EFAULT;
		break;
	}

	default:
		return -ENOTTY;
	}

	return ret;
}

/* -----------------------------------------------------------------------
 * File operations table
 * ----------------------------------------------------------------------- */

static const struct file_operations zcd_fops = {
	.owner          = THIS_MODULE,
	.open           = zcd_open,
	.release        = zcd_release,
	.mmap           = zcd_mmap,
	.unlocked_ioctl = zcd_ioctl,
	.llseek         = noop_llseek,
};

/* -----------------------------------------------------------------------
 * Virtual platform device (DMA master)
 *
 * We register a minimal platform_device to obtain a struct device * that
 * satisfies the DMA mapping API (dma_map_sg / dma_unmap_sg).  Without a
 * real hardware node in the device tree, this is the standard approach for
 * out-of-tree drivers targeting virtual environments.
 * ----------------------------------------------------------------------- */

/**
 * zcd_pdev_alloc() - Allocate and register the virtual DMA platform device.
 *
 * Return: pointer to the new platform_device on success, ERR_PTR() on error.
 */
static struct platform_device *zcd_pdev_alloc(void)
{
	struct platform_device *pdev;
	int ret;

	pdev = platform_device_alloc("zero_copy_dma_hw", PLATFORM_DEVID_NONE);
	if (!pdev)
		return ERR_PTR(-ENOMEM);

	ret = platform_device_add(pdev);
	if (ret) {
		pr_err("zcd: platform_device_add failed (%d)\n", ret);
		platform_device_put(pdev);
		return ERR_PTR(ret);
	}

	/*
	 * Set the DMA address mask.  Try 64-bit first (modern systems);
	 * fall back to 32-bit for older or restricted platforms.
	 */
	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	if (ret) {
		pr_warn("zcd: 64-bit DMA mask failed, trying 32-bit\n");
		ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
		if (ret) {
			pr_err("zcd: cannot configure DMA mask (%d)\n", ret);
			platform_device_del(pdev);
			platform_device_put(pdev);
			return ERR_PTR(ret);
		}
	}

	return pdev;
}

/* -----------------------------------------------------------------------
 * Module init / exit
 * ----------------------------------------------------------------------- */

/**
 * zcd_init() - Module entry point.
 *
 * Initialisation sequence (reverse order is the tear-down path):
 *
 *   Step 1: alloc_chrdev_region()    — dynamic major/minor numbers
 *   Step 2: class_create()           — /sys/class/zero_copy_dma
 *   Step 3: zcd_pdev_alloc()         — virtual DMA device
 *   Step 4: zcd_alloc_buffers()      — 4×1 MiB pages + SG tables
 *   Step 5: kthread_run()            — mock hardware thread
 *   Step 6: cdev_init() + cdev_add() — character device registration
 *   Step 7: device_create()          — /dev/zero_copy_dma node
 *
 * Every step is followed by a labelled error path that undoes all prior
 * steps.  No resource is allocated without a corresponding free.
 *
 * Return: 0 on success, negative errno on failure.
 */
static int __init zcd_init(void)
{
	int ret;

	pr_info("zcd: loading zero-copy DMA driver\n");

	/* Ensure global state is clean (it is zero-initialised at link time,
	 * but be explicit in case of future multi-instance support). */
	spin_lock_init(&zcd_dev.stream_lock);
	init_waitqueue_head(&zcd_dev.frame_wq);
	atomic_set(&zcd_dev.open_count, 0);
	zcd_dev.state = ZCD_STATE_IDLE;

	/* ── Step 1: Dynamic major/minor allocation ── */
	ret = alloc_chrdev_region(&zcd_dev.devnum, 0, 1, "zero_copy_dma");
	if (ret) {
		pr_err("zcd: alloc_chrdev_region failed (%d)\n", ret);
		return ret;
	}

	/* ── Step 2: Device class ── */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
	zcd_dev.cls = class_create("zero_copy_dma");
#else
	zcd_dev.cls = class_create(THIS_MODULE, "zero_copy_dma");
#endif
	if (IS_ERR(zcd_dev.cls)) {
		ret = PTR_ERR(zcd_dev.cls);
		pr_err("zcd: class_create failed (%d)\n", ret);
		goto err_unregister_chrdev;
	}

	/* ── Step 3: Virtual platform device for DMA API ── */
	zcd_dev.pdev = zcd_pdev_alloc();
	if (IS_ERR(zcd_dev.pdev)) {
		ret = PTR_ERR(zcd_dev.pdev);
		goto err_class_destroy;
	}

	/* ── Step 4: DMA ring buffers (4 × 1 MiB) + SG tables ── */
	ret = zcd_alloc_buffers(&zcd_dev);
	if (ret)
		goto err_pdev_del;

	/* ── Step 5: Mock hardware kthread ── */
	zcd_dev.mock_thread = kthread_run(zcd_mock_thread, &zcd_dev,
					  "zcd/mock_hw");
	if (IS_ERR(zcd_dev.mock_thread)) {
		ret = PTR_ERR(zcd_dev.mock_thread);
		pr_err("zcd: kthread_run failed (%d)\n", ret);
		goto err_free_bufs;
	}

	/* ── Step 6: Character device registration ── */
	cdev_init(&zcd_dev.cdev, &zcd_fops);
	zcd_dev.cdev.owner = THIS_MODULE;

	ret = cdev_add(&zcd_dev.cdev, zcd_dev.devnum, 1);
	if (ret) {
		pr_err("zcd: cdev_add failed (%d)\n", ret);
		goto err_stop_thread;
	}

	/* ── Step 7: /dev/zero_copy_dma node ── */
	zcd_dev.dev = device_create(zcd_dev.cls, NULL, zcd_dev.devnum,
				    NULL, "zero_copy_dma");
	if (IS_ERR(zcd_dev.dev)) {
		ret = PTR_ERR(zcd_dev.dev);
		pr_err("zcd: device_create failed (%d)\n", ret);
		goto err_cdev_del;
	}

	pr_info("zcd: loaded  major=%d minor=%d  ring=%u×%u KiB\n",
		MAJOR(zcd_dev.devnum), MINOR(zcd_dev.devnum),
		ZCD_RING_DEPTH, ZCD_FRAME_SIZE / 1024U);
	return 0;

/* ── Error unwind (strict reverse order) ── */
err_cdev_del:
	cdev_del(&zcd_dev.cdev);
err_stop_thread:
	WRITE_ONCE(zcd_dev.streaming, false);
	kthread_stop(zcd_dev.mock_thread);
	zcd_dev.mock_thread = NULL;
err_free_bufs:
	zcd_free_buffers(&zcd_dev);
err_pdev_del:
	platform_device_del(zcd_dev.pdev);
	platform_device_put(zcd_dev.pdev);
	zcd_dev.pdev = NULL;
err_class_destroy:
	class_destroy(zcd_dev.cls);
	zcd_dev.cls = NULL;
err_unregister_chrdev:
	unregister_chrdev_region(zcd_dev.devnum, 1);
	return ret;
}

/**
 * zcd_exit() - Module teardown; releases all resources in reverse init order.
 *
 * Tear-down sequence:
 *   1. Remove /dev/zero_copy_dma device node.
 *   2. Delete character device.
 *   3. Stop mock hardware kthread (waits for it to finish).
 *   4. Free DMA ring buffers + SG tables.
 *   5. Unregister virtual platform device.
 *   6. Destroy device class.
 *   7. Release character device numbers.
 *
 * After this function returns the driver has no references to any kernel
 * object; no dangling pointers remain.
 */
static void __exit zcd_exit(void)
{
	pr_info("zcd: unloading\n");

	/* 1. /dev node */
	device_destroy(zcd_dev.cls, zcd_dev.devnum);
	zcd_dev.dev = NULL;

	/* 2. cdev */
	cdev_del(&zcd_dev.cdev);

	/* 3. Mock hardware kthread */
	if (zcd_dev.mock_thread) {
		/* Clear streaming so the thread does not block in msleep. */
		WRITE_ONCE(zcd_dev.streaming, false);
		wake_up_interruptible(&zcd_dev.frame_wq);
		kthread_stop(zcd_dev.mock_thread);
		zcd_dev.mock_thread = NULL;
	}

	/* 4. DMA ring buffers */
	zcd_free_buffers(&zcd_dev);

	/* 5. Virtual platform device */
	if (zcd_dev.pdev) {
		platform_device_del(zcd_dev.pdev);
		platform_device_put(zcd_dev.pdev);
		zcd_dev.pdev = NULL;
	}

	/* 6. Device class */
	if (zcd_dev.cls) {
		class_destroy(zcd_dev.cls);
		zcd_dev.cls = NULL;
	}

	/* 7. Char device numbers */
	unregister_chrdev_region(zcd_dev.devnum, 1);

	pr_info("zcd: unloaded cleanly\n");
}

module_init(zcd_init);
module_exit(zcd_exit);
