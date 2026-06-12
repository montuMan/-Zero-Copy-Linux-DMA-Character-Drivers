/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * zero_copy_dma.h - Public interface for the Zero-Copy Scatter-Gather DMA
 *                   Character Driver.
 *
 * This header is shared between the kernel module and the user-space test
 * harness.  It defines:
 *
 *   - Buffer geometry constants (ring depth, frame size, page order).
 *   - Mock hardware timing parameters.
 *   - Device state codes.
 *   - IOCTL command numbers (built with the _IO/_IOR/_IOW macros so the
 *     kernel can validate direction and size automatically).
 *   - The struct zcd_frame_info structure that is copied to/from user-space
 *     on every ZCD_IOCTL_GET_FRAME call.
 *
 * Copyright (C) 2024 Zero-Copy DMA Driver Authors
 */

#ifndef _ZERO_COPY_DMA_H
#define _ZERO_COPY_DMA_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* -----------------------------------------------------------------------
 * Buffer geometry
 * ----------------------------------------------------------------------- */

/** Number of frame slots in the DMA ring buffer. */
#define ZCD_RING_DEPTH          4U

/**
 * Size of each frame buffer in bytes (1 MiB).
 * Must be an exact power-of-two multiple of PAGE_SIZE (4 KiB).
 */
#define ZCD_FRAME_SIZE          (1U * 1024U * 1024U)

/**
 * Total size of the contiguous virtual-address window exposed to user-space
 * via mmap().  The user process maps this entire region once; individual
 * frames are accessed at offset (frame_index * ZCD_FRAME_SIZE).
 */
#define ZCD_TOTAL_MAP_SIZE      (ZCD_RING_DEPTH * ZCD_FRAME_SIZE)

/**
 * alloc_pages() order for one frame slot.
 * order 8  =>  2^8 = 256 contiguous 4-KiB pages = 1 MiB.
 */
#define ZCD_PAGES_ORDER         8U

/** Number of 4-KiB pages per frame slot (must equal 2^ZCD_PAGES_ORDER). */
#define ZCD_PAGES_PER_FRAME     (ZCD_FRAME_SIZE / 4096U)

/* -----------------------------------------------------------------------
 * Mock hardware timing
 * ----------------------------------------------------------------------- */

/**
 * Interval between simulated "frame-ready" interrupts, in milliseconds.
 * 33 ms ≈ 30 fps — a typical machine-vision sensor cadence.
 */
#define ZCD_MOCK_INTERVAL_MS    33U

/* -----------------------------------------------------------------------
 * Device state codes (returned via ZCD_IOCTL_QUERY_STATUS)
 * ----------------------------------------------------------------------- */

#define ZCD_STATE_IDLE          0   /* Stream not running                  */
#define ZCD_STATE_STREAMING     1   /* Mock DMA engine delivering frames   */
#define ZCD_STATE_ERROR         2   /* Unrecoverable error; re-open needed */

/* -----------------------------------------------------------------------
 * IOCTL magic number and command definitions
 *
 * Magic byte: 0xDC  (mnemonic: "DMA Character")
 * Sequence  : 0 … 4
 * ----------------------------------------------------------------------- */

#define ZCD_IOC_MAGIC           0xDC

/**
 * ZCD_IOCTL_START_STREAM
 *   Arm the software-mock DMA engine.  Frames begin arriving at
 *   ZCD_MOCK_INTERVAL_MS cadence.  Returns -EALREADY if already streaming.
 *   Argument: none.
 */
#define ZCD_IOCTL_START_STREAM  _IO(ZCD_IOC_MAGIC, 0)

/**
 * ZCD_IOCTL_STOP_STREAM
 *   Halt the mock DMA engine.  Any process blocked in ZCD_IOCTL_GET_FRAME
 *   is woken and receives -ECANCELED.
 *   Argument: none.
 */
#define ZCD_IOCTL_STOP_STREAM   _IO(ZCD_IOC_MAGIC, 1)

/**
 * ZCD_IOCTL_GET_FRAME
 *   Block until the next frame is available (or the stream is stopped /
 *   a signal is delivered), then write a struct zcd_frame_info describing
 *   the ready slot into the supplied user-space pointer.
 *
 *   The caller must subsequently call ZCD_IOCTL_RELEASE_FRAME with the
 *   same index to return the slot to the producer.
 *
 *   Argument: __user struct zcd_frame_info * (output).
 */
#define ZCD_IOCTL_GET_FRAME     _IOR(ZCD_IOC_MAGIC, 2, struct zcd_frame_info)

/**
 * ZCD_IOCTL_RELEASE_FRAME
 *   Return a consumed frame slot back to the producer ring.  Must be
 *   called with the index reported by the preceding ZCD_IOCTL_GET_FRAME.
 *   Returns -EINVAL for an out-of-order or out-of-range index.
 *
 *   Argument: __user int * (input — frame slot index).
 */
#define ZCD_IOCTL_RELEASE_FRAME _IOW(ZCD_IOC_MAGIC, 3, int)

/**
 * ZCD_IOCTL_QUERY_STATUS
 *   Return the current device state (one of the ZCD_STATE_* values).
 *   Non-blocking.
 *
 *   Argument: __user int * (output).
 */
#define ZCD_IOCTL_QUERY_STATUS  _IOR(ZCD_IOC_MAGIC, 4, int)

/* -----------------------------------------------------------------------
 * Shared data structure
 * ----------------------------------------------------------------------- */

/**
 * struct zcd_frame_info - Metadata for one completed DMA frame.
 *
 * Filled by the kernel on ZCD_IOCTL_GET_FRAME and copied to user-space.
 * The frame payload begins at:
 *
 *     mmap_base + (index * ZCD_FRAME_SIZE)
 *
 * @sequence:   Monotonically increasing frame counter (starts at 0 after
 *              each ZCD_IOCTL_START_STREAM).  Use this to detect dropped
 *              frames.
 * @index:      Ring-buffer slot index in [0, ZCD_RING_DEPTH).  Pass this
 *              back verbatim to ZCD_IOCTL_RELEASE_FRAME.
 * @size_bytes: Number of valid payload bytes written by the mock DMA engine
 *              (always ZCD_FRAME_SIZE in the current implementation).
 *
 * Layout note: @sequence is first so the __u64 member is naturally aligned
 * without padding, keeping the struct size a clean 16 bytes on both 32-bit
 * and 64-bit platforms.
 */
struct zcd_frame_info {
	__u64 sequence;
	__u32 index;
	__u32 size_bytes;
};

#endif /* _ZERO_COPY_DMA_H */
