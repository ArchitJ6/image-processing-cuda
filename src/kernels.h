#pragma once

// ===== GPU =====
__global__ void grayscale_kernel(unsigned char *, unsigned char *, int, int);
__global__ void sobel_shared_kernel(unsigned char *, unsigned char *, int, int, int);
__global__ void sharpen_kernel(unsigned char *, unsigned char *, int, int);
__global__ void blur_shared_kernel(unsigned char *, unsigned char *, int, int);
__global__ void sharpen_shared_kernel(unsigned char *input, unsigned char *output, int w, int h);

// ===== CPU =====
void grayscale_cpu(unsigned char *, unsigned char *, int, int);
void blur_cpu(unsigned char *, unsigned char *, int, int);
void sobel_cpu(unsigned char *, unsigned char *, int, int, int);
void sharpen_cpu(unsigned char *, unsigned char *, int, int);