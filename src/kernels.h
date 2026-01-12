#pragma once

// GPU kernel
__global__ void grayscale_kernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
);

// CPU version
void grayscale_cpu(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
);