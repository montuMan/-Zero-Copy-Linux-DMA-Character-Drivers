# SPDX-License-Identifier: GPL-2.0-only
#
# Makefile for the Zero-Copy Scatter-Gather DMA Character Driver
#
# Usage (out-of-tree build against the running kernel):
#
#   make              — build zero_copy_dma.ko
#   make clean        — remove build artefacts
#   make KDIR=/path   — build against a specific kernel source tree
#
# To load the module after a successful build:
#
#   sudo insmod zero_copy_dma.ko
#   ls -la /dev/zero_copy_dma      # confirm /dev node was created
#   dmesg | tail -20               # inspect kernel log
#
# To unload:
#
#   sudo rmmod zero_copy_dma
#
# Build the user-space test harness separately with:
#
#   g++ -O2 -Wall -o user_app user_app.cpp
#
# -------------------------------------------------------------------------

# Object(s) that compose the module.
obj-m := zero_copy_dma.o

# Path to the kernel build tree.  Defaults to the running kernel's headers.
KDIR  ?= /lib/modules/$(shell uname -r)/build

# Absolute path to this Makefile's directory (works with out-of-tree builds).
PWD   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# --------------------------------------------------------------------------
# Primary targets
# --------------------------------------------------------------------------

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

# Convenience target: build + load + tail dmesg
install: all
	sudo insmod $(PWD)/zero_copy_dma.ko
	dmesg | tail -30

uninstall:
	sudo rmmod zero_copy_dma || true

# --------------------------------------------------------------------------
# Phony declarations
# --------------------------------------------------------------------------

.PHONY: all clean install uninstall
