#include "kernels.h"

// GPU kernel
__global__ void grayscale_kernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;

        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        unsigned char gray = 0.299f*r + 0.587f*g + 0.114f*b;

        output[idx]     = gray;
        output[idx + 1] = gray;
        output[idx + 2] = gray;
    }
}

// CPU version
void grayscale_cpu(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;

            unsigned char r = input[idx];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];

            unsigned char gray = 0.299f*r + 0.587f*g + 0.114f*b;

            output[idx]     = gray;
            output[idx + 1] = gray;
            output[idx + 2] = gray;
        }
    }
}