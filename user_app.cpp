/*
 * user_app.cpp  –  Zero-Copy DMA Driver  ·  User-Space Test Harness
 *
 * PURPOSE
 * ───────
 * This program exercises the zero_copy_dma kernel module end-to-end:
 *
 *   1. Opens /dev/zero_copy_dma (fails if module is not loaded).
 *   2. mmap()s the entire DMA ring (ZCD_TOTAL_MAP_SIZE = 4 MiB) — after
 *      this call the process can read frame data directly from DMA pages
 *      without any copy_to_user() or read(2) syscall.
 *   3. Queries device status and verifies the initial state is IDLE.
 *   4. Issues ZCD_IOCTL_START_STREAM to arm the mock DMA engine.
 *   5. Loops NUM_FRAMES times:
 *        a. ZCD_IOCTL_GET_FRAME  — blocks until the kernel writes a new
 *                                   frame and wakes this thread.
 *        b. Verifies a sample of bytes using the known test pattern
 *           byte[i] = (seq ^ slot ^ i) & 0xFF
 *           Any mismatch is reported as a data-integrity failure.
 *        c. ZCD_IOCTL_RELEASE_FRAME — returns the slot to the producer.
 *   6. Issues ZCD_IOCTL_STOP_STREAM.
 *   7. Unmaps the buffer and closes the device.
 *
 * BUILD
 * ─────
 *   g++ -O2 -Wall -Wextra -o user_app user_app.cpp
 *
 * RUN
 * ───
 *   sudo ./user_app           # /dev/zero_copy_dma requires root (or udev rule)
 *
 * Copyright (C) 2024 Zero-Copy DMA Driver Authors
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

/*
 * Include the shared header.  On the build host the kernel headers must be
 * installed (package linux-headers-$(uname -r) on Debian/Ubuntu).
 * The header uses only <linux/ioctl.h> and <linux/types.h> which are always
 * present in user-space /usr/include/linux/.
 */
#include "zero_copy_dma.h"

/* -----------------------------------------------------------------------
 * Configuration
 * ----------------------------------------------------------------------- */

/** Device node created by the module. */
static constexpr const char *ZCD_DEV = "/dev/zero_copy_dma";

/** Number of frames to capture and verify in a single test run. */
static constexpr int NUM_FRAMES = 30;

/**
 * Number of bytes checked per frame (checking every byte of 1 MiB takes
 * ~1 ms; a 4096-byte sample is instant and sufficient for demonstration).
 */
static constexpr uint32_t CHECK_BYTES = 4096U;

/* -----------------------------------------------------------------------
 * Helpers
 * ----------------------------------------------------------------------- */

/**
 * check_pattern() - Verify that @len bytes at @buf match the expected
 *                   zero-copy test pattern.
 *
 * The kernel fills each byte as:  buf[i] = (seq ^ slot ^ i) & 0xFF
 *
 * @buf:      Pointer into the mmap'd window for this frame.
 * @len:      Number of bytes to verify.
 * @slot:     Ring slot index (reported in zcd_frame_info::index).
 * @seq:      Frame sequence number (zcd_frame_info::sequence).
 *
 * @return    true if all bytes are correct, false otherwise.
 */
static bool check_pattern(const uint8_t *buf, uint32_t len,
                           uint32_t slot, uint64_t seq)
{
    if (len > ZCD_FRAME_SIZE) {
        std::fprintf(stderr,
            "  [FAIL] check_pattern: len %u exceeds frame size %u\n",
            len, ZCD_FRAME_SIZE);
        return false;
    }

    for (uint32_t i = 0; i < len; ++i) {
        uint8_t expected = static_cast<uint8_t>((seq ^ slot ^ i) & 0xFF);
        if (buf[i] != expected) {
            std::fprintf(stderr,
                "  [FAIL] byte[%u]: got 0x%02x, expected 0x%02x "
                "(slot=%u seq=%llu)\n",
                i, buf[i], expected,
                slot, static_cast<unsigned long long>(seq));
            return false;
        }
    }
    return true;
}

/* -----------------------------------------------------------------------
 * main()
 * ----------------------------------------------------------------------- */

int main(void)
{
    int      fd   = -1;
    void    *mmap_base = MAP_FAILED;
    int      ret  = EXIT_FAILURE;

    /* ── 1. Open the device ── */
    fd = open(ZCD_DEV, O_RDWR);
    if (fd < 0) {
        std::perror("open");
        std::fprintf(stderr,
            "Hint: is the zero_copy_dma kernel module loaded?\n"
            "      sudo insmod zero_copy_dma.ko\n");
        return EXIT_FAILURE;
    }
    std::printf("[OK] Opened %s  (fd=%d)\n", ZCD_DEV, fd);

    /* ── 2. mmap the entire DMA ring ── */
    mmap_base = mmap(nullptr,
                     ZCD_TOTAL_MAP_SIZE,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED,
                     fd, 0);
    if (mmap_base == MAP_FAILED) {
        std::perror("mmap");
        goto cleanup;
    }
    std::printf("[OK] mmap: %u KiB at %p  (zero-copy window ready)\n",
                ZCD_TOTAL_MAP_SIZE / 1024U, mmap_base);

    /* ── 3. Query initial device status ── */
    {
        int state = -1;
        if (ioctl(fd, ZCD_IOCTL_QUERY_STATUS, &state) < 0) {
            std::perror("ioctl QUERY_STATUS");
            goto cleanup;
        }
        std::printf("[OK] Initial state: %d (expected %d = IDLE)\n",
                    state, ZCD_STATE_IDLE);
        if (state != ZCD_STATE_IDLE) {
            std::fprintf(stderr, "[FAIL] Device not in IDLE state\n");
            goto cleanup;
        }
    }

    /* ── 4. Start stream ── */
    if (ioctl(fd, ZCD_IOCTL_START_STREAM) < 0) {
        std::perror("ioctl START_STREAM");
        goto cleanup;
    }
    std::printf("[OK] Stream started (mock DMA ~30 fps)\n\n");

    /* ── 5. Capture and verify NUM_FRAMES frames ── */
    {
        int  pass = 0, fail = 0;

        for (int frame_num = 0; frame_num < NUM_FRAMES; ++frame_num) {
            struct zcd_frame_info fi;

            /* Block until the kernel signals a new frame is ready. */
            if (ioctl(fd, ZCD_IOCTL_GET_FRAME, &fi) < 0) {
                if (errno == EINTR || errno == ECANCELED) {
                    std::fprintf(stderr, "Stream interrupted\n");
                    break;
                }
                std::perror("ioctl GET_FRAME");
                goto stop;
            }

            /*
             * Zero-copy access: read frame data directly from the
             * kernel DMA page mapped into our address space.
             * No copy_to_user(), no read(2) call — just a pointer dereference.
             */
            const uint8_t *frame_ptr =
                static_cast<const uint8_t *>(mmap_base)
                + fi.index * ZCD_FRAME_SIZE;

            bool ok = check_pattern(frame_ptr, CHECK_BYTES,
                                    fi.index, fi.sequence);

            std::printf("  frame #%3d | slot=%u seq=%-5llu size=%-7u | %s\n",
                        frame_num,
                        fi.index,
                        static_cast<unsigned long long>(fi.sequence),
                        fi.size_bytes,
                        ok ? "PASS" : "FAIL");

            if (ok)
                ++pass;
            else
                ++fail;

            /* Return the slot to the producer. */
            int idx = static_cast<int>(fi.index);
            if (ioctl(fd, ZCD_IOCTL_RELEASE_FRAME, &idx) < 0) {
                std::perror("ioctl RELEASE_FRAME");
                goto stop;
            }
        }

        std::printf("\n[SUMMARY] %d/%d frames passed pattern check\n",
                    pass, NUM_FRAMES);
        if (fail == 0 && pass == NUM_FRAMES)
            ret = EXIT_SUCCESS;
    }

stop:
    /* ── 6. Stop stream ── */
    if (ioctl(fd, ZCD_IOCTL_STOP_STREAM) < 0)
        std::perror("ioctl STOP_STREAM");
    else
        std::printf("[OK] Stream stopped\n");

cleanup:
    if (mmap_base != MAP_FAILED) {
        munmap(mmap_base, ZCD_TOTAL_MAP_SIZE);
        std::printf("[OK] mmap region released\n");
    }
    if (fd >= 0) {
        close(fd);
        std::printf("[OK] Device closed\n");
    }

    return ret;
}
