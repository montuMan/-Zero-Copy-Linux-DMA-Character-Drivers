# -Zero-Copy-Linux-DMA-Character-Drivers
Standard Linux character drivers often copy data from hardware registers to kernel space, and then use copy_to_user() to pass it to an application. This dual-copy architecture completely kills frame rates in vision pipelines.
