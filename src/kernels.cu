#include "kernels.h"
#include <algorithm>
#include <cmath>

#define BLOCK 16

// ==========================
// 🧠 HELPER (CPU BORDER COPY)
// ==========================
void copy_borders(unsigned char *input, unsigned char *output, int w, int h)
{
    std::copy(input, input + (w * h * 3), output);
}

// ==========================
// 🔹 GPU: GRAYSCALE
// ==========================
__global__ void grayscale_kernel(
    unsigned char *input,
    unsigned char *output,
    int w,
    int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int idx = (y * w + x) * 3;

        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];

        unsigned char gray =
            0.299f * r +
            0.587f * g +
            0.114f * b;

        output[idx] = output[idx + 1] = output[idx + 2] = gray;
    }
}

// ==========================
// 🔹 CPU: GRAYSCALE
// ==========================
void grayscale_cpu(unsigned char *input, unsigned char *output, int w, int h)
{
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int idx = (y * w + x) * 3;

            unsigned char b = input[idx];
            unsigned char g = input[idx + 1];
            unsigned char r = input[idx + 2];

            unsigned char gray =
                0.299f * r +
                0.587f * g +
                0.114f * b;

            output[idx] = output[idx + 1] = output[idx + 2] = gray;
        }
    }
}

// ==========================
// 🔹 CPU: BLUR
// ==========================
void blur_cpu(unsigned char *input, unsigned char *output, int w, int h)
{
    copy_borders(input, output, w, h);

    for (int y = 1; y < h - 1; y++)
    {
        for (int x = 1; x < w - 1; x++)
        {

            int sum = 0;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    sum += input[((y + ky) * w + (x + kx)) * 3];

            int idx = (y * w + x) * 3;
            unsigned char val = sum / 9;

            output[idx] = output[idx + 1] = output[idx + 2] = val;
        }
    }
}

// ==========================
// 🔥 GPU: SOBEL (SHARED MEMORY)
// ==========================
__global__ void sobel_shared_kernel(unsigned char *input, unsigned char *output, int w, int h, int threshold)
{
    __shared__ unsigned char tile[BLOCK + 2][BLOCK + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK + tx;
    int y = blockIdx.y * BLOCK + ty;

    // initialize everything we may touch
    tile[ty + 1][tx + 1] = 0;

    if (tx == 0)
        tile[ty + 1][0] = 0;

    if (tx == BLOCK - 1)
        tile[ty + 1][BLOCK + 1] = 0;

    if (ty == 0)
        tile[0][tx + 1] = 0;

    if (ty == BLOCK - 1)
        tile[BLOCK + 1][tx + 1] = 0;

    // initialize corners
    if (tx == 0 && ty == 0)
        tile[0][0] = 0;

    if (tx == BLOCK - 1 && ty == 0)
        tile[0][BLOCK + 1] = 0;

    if (tx == 0 && ty == BLOCK - 1)
        tile[BLOCK + 1][0] = 0;

    if (tx == BLOCK - 1 && ty == BLOCK - 1)
        tile[BLOCK + 1][BLOCK + 1] = 0;
    if (x < w && y < h)
    {
        tile[ty + 1][tx + 1] = input[(y * w + x) * 3];
    }

    if (tx == 0 && x > 0)
        tile[ty + 1][0] = input[(y * w + (x - 1)) * 3];
    if (tx == BLOCK - 1 && x < w - 1)
        tile[ty + 1][BLOCK + 1] = input[(y * w + (x + 1)) * 3];

    if (ty == 0 && y > 0)
        tile[0][tx + 1] = input[((y - 1) * w + x) * 3];
    if (ty == BLOCK - 1 && y < h - 1)
        tile[BLOCK + 1][tx + 1] = input[((y + 1) * w + x) * 3];

    // Top-left corner
    if (tx == 0 && ty == 0 &&
        x > 0 && y > 0)
    {
        tile[0][0] =
            input[((y - 1) * w + (x - 1)) * 3];
    }

    // Top-right corner
    if (tx == BLOCK - 1 && ty == 0 &&
        x < w - 1 && y > 0)
    {
        tile[0][BLOCK + 1] =
            input[((y - 1) * w + (x + 1)) * 3];
    }

    // Bottom-left corner
    if (tx == 0 && ty == BLOCK - 1 &&
        x > 0 && y < h - 1)
    {
        tile[BLOCK + 1][0] =
            input[((y + 1) * w + (x - 1)) * 3];
    }

    // Bottom-right corner
    if (tx == BLOCK - 1 &&
        ty == BLOCK - 1 &&
        x < w - 1 &&
        y < h - 1)
    {
        tile[BLOCK + 1][BLOCK + 1] =
            input[((y + 1) * w + (x + 1)) * 3];
    }
    if (x < w && y < h)
    {
        int idx = (y * w + x) * 3;

        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
    }

    __syncthreads();

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1)
    {

        int gx =
            -tile[ty][tx] + tile[ty][tx + 2] - 2 * tile[ty + 1][tx] + 2 * tile[ty + 1][tx + 2] - tile[ty + 2][tx] + tile[ty + 2][tx + 2];

        int gy =
            -tile[ty][tx] - 2 * tile[ty][tx + 1] - tile[ty][tx + 2] + tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2];

        float mag_f =
            sqrtf(
                (float)gx * gx +
                (float)gy * gy);
        int mag = min(255, (int)mag_f);

        if (threshold > 0)
        {
            mag = (mag > threshold) ? 255 : 0;
        }

        int idx = (y * w + x) * 3;
        output[idx] = output[idx + 1] = output[idx + 2] = mag;
    }
}

// ==========================
// 🔹 CPU: SOBEL
// ==========================
void sobel_cpu(unsigned char *input, unsigned char *output, int w, int h, int threshold)
{
    copy_borders(input, output, w, h);

    for (int y = 1; y < h - 1; y++)
    {
        for (int x = 1; x < w - 1; x++)
        {

            int gx =
                -input[((y - 1) * w + (x - 1)) * 3] + input[((y - 1) * w + (x + 1)) * 3] - 2 * input[(y * w + (x - 1)) * 3] + 2 * input[(y * w + (x + 1)) * 3] - input[((y + 1) * w + (x - 1)) * 3] + input[((y + 1) * w + (x + 1)) * 3];

            int gy =
                -input[((y - 1) * w + (x - 1)) * 3] - 2 * input[((y - 1) * w + x) * 3] - input[((y - 1) * w + (x + 1)) * 3] + input[((y + 1) * w + (x - 1)) * 3] + 2 * input[((y + 1) * w + x) * 3] + input[((y + 1) * w + (x + 1)) * 3];

            float mag_f =
                std::sqrt(
                    (float)gx * gx +
                    (float)gy * gy);
            int mag =
                std::min(
                    255,
                    (int)mag_f);

            if (threshold > 0)
            {
                mag =
                    (mag > threshold)
                        ? 255
                        : 0;
            }

            int idx = (y * w + x) * 3;
            output[idx] = output[idx + 1] = output[idx + 2] = mag;
        }
    }
}

// ==========================
// 🔹 GPU: SHARPEN
// ==========================
__global__ void sharpen_kernel(unsigned char *input, unsigned char *output, int w, int h)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int idx = (y * w + x) * 3;

        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
    }

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1)
    {
        int idx = (y * w + x) * 3;

        int val = 5 * input[idx] - input[((y - 1) * w + x) * 3] - input[((y + 1) * w + x) * 3] - input[(y * w + (x - 1)) * 3] - input[(y * w + (x + 1)) * 3];

        val = min(255, max(0, val));

        output[idx] = output[idx + 1] = output[idx + 2] = val;
    }
}

__global__ void sharpen_shared_kernel(
    unsigned char* input,
    unsigned char* output,
    int w,
    int h)
{
    __shared__ unsigned char tile[BLOCK + 2][BLOCK + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK + tx;
    int y = blockIdx.y * BLOCK + ty;

    // ==========================
    // Initialize shared memory
    // ==========================
    tile[ty + 1][tx + 1] = 0;

    if (tx == 0)
        tile[ty + 1][0] = 0;

    if (tx == BLOCK - 1)
        tile[ty + 1][BLOCK + 1] = 0;

    if (ty == 0)
        tile[0][tx + 1] = 0;

    if (ty == BLOCK - 1)
        tile[BLOCK + 1][tx + 1] = 0;

    if (tx == 0 && ty == 0)
        tile[0][0] = 0;

    if (tx == BLOCK - 1 && ty == 0)
        tile[0][BLOCK + 1] = 0;

    if (tx == 0 && ty == BLOCK - 1)
        tile[BLOCK + 1][0] = 0;

    if (tx == BLOCK - 1 && ty == BLOCK - 1)
        tile[BLOCK + 1][BLOCK + 1] = 0;

    // ==========================
    // Load center pixel
    // ==========================
    if (x < w && y < h)
    {
        tile[ty + 1][tx + 1] =
            input[(y * w + x) * 3];
    }

    // ==========================
    // Load left/right halo
    // ==========================
    if (tx == 0 && x > 0)
    {
        tile[ty + 1][0] =
            input[(y * w + (x - 1)) * 3];
    }

    if (tx == BLOCK - 1 && x < w - 1)
    {
        tile[ty + 1][BLOCK + 1] =
            input[(y * w + (x + 1)) * 3];
    }

    // ==========================
    // Load top/bottom halo
    // ==========================
    if (ty == 0 && y > 0)
    {
        tile[0][tx + 1] =
            input[((y - 1) * w + x) * 3];
    }

    if (ty == BLOCK - 1 && y < h - 1)
    {
        tile[BLOCK + 1][tx + 1] =
            input[((y + 1) * w + x) * 3];
    }

    // ==========================
    // Load corner halos
    // ==========================
    if (tx == 0 && ty == 0 &&
        x > 0 && y > 0)
    {
        tile[0][0] =
            input[((y - 1) * w + (x - 1)) * 3];
    }

    if (tx == BLOCK - 1 && ty == 0 &&
        x < w - 1 && y > 0)
    {
        tile[0][BLOCK + 1] =
            input[((y - 1) * w + (x + 1)) * 3];
    }

    if (tx == 0 && ty == BLOCK - 1 &&
        x > 0 && y < h - 1)
    {
        tile[BLOCK + 1][0] =
            input[((y + 1) * w + (x - 1)) * 3];
    }

    if (tx == BLOCK - 1 &&
        ty == BLOCK - 1 &&
        x < w - 1 &&
        y < h - 1)
    {
        tile[BLOCK + 1][BLOCK + 1] =
            input[((y + 1) * w + (x + 1)) * 3];
    }

    // Copy borders
    if (x < w && y < h)
    {
        int idx = (y * w + x) * 3;

        output[idx]     = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
    }

    __syncthreads();

    // ==========================
    // Sharpen computation
    // ==========================
    if (x > 0 && x < w - 1 &&
        y > 0 && y < h - 1)
    {
        int val =
              5 * tile[ty + 1][tx + 1]
            - tile[ty][tx + 1]
            - tile[ty + 2][tx + 1]
            - tile[ty + 1][tx]
            - tile[ty + 1][tx + 2];

        val = min(255, max(0, val));

        int idx = (y * w + x) * 3;

        output[idx] =
        output[idx + 1] =
        output[idx + 2] =
            (unsigned char)val;
    }
}

// ==========================
// 🔹 CPU: SHARPEN
// ==========================
void sharpen_cpu(unsigned char *input, unsigned char *output, int w, int h)
{
    copy_borders(input, output, w, h);

    for (int y = 1; y < h - 1; y++)
    {
        for (int x = 1; x < w - 1; x++)
        {

            int idx = (y * w + x) * 3;

            int val =
                5 * input[idx] - input[((y - 1) * w + x) * 3] - input[((y + 1) * w + x) * 3] - input[(y * w + (x - 1)) * 3] - input[(y * w + (x + 1)) * 3];

            val = std::min(255, std::max(0, val));

            output[idx] = output[idx + 1] = output[idx + 2] = val;
        }
    }
}

__global__ void blur_shared_kernel(
    unsigned char *input,
    unsigned char *output,
    int w,
    int h)
{
    __shared__ unsigned char tile[BLOCK + 2][BLOCK + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK + tx;
    int y = blockIdx.y * BLOCK + ty;

    tile[ty + 1][tx + 1] = 0;

    if (tx == 0)
        tile[ty + 1][0] = 0;

    if (tx == BLOCK - 1)
        tile[ty + 1][BLOCK + 1] = 0;

    if (ty == 0)
        tile[0][tx + 1] = 0;

    if (ty == BLOCK - 1)
        tile[BLOCK + 1][tx + 1] = 0;

    if (tx == 0 && ty == 0)
        tile[0][0] = 0;

    if (tx == BLOCK - 1 && ty == 0)
        tile[0][BLOCK + 1] = 0;

    if (tx == 0 && ty == BLOCK - 1)
        tile[BLOCK + 1][0] = 0;

    if (tx == BLOCK - 1 && ty == BLOCK - 1)
        tile[BLOCK + 1][BLOCK + 1] = 0;

    // Load center pixel
    if (x < w && y < h)
        tile[ty + 1][tx + 1] = input[(y * w + x) * 3];

    // Load halo (left/right)
    if (tx == 0 && x > 0)
        tile[ty + 1][0] = input[(y * w + (x - 1)) * 3];
    if (tx == BLOCK - 1 && x < w - 1)
        tile[ty + 1][BLOCK + 1] = input[(y * w + (x + 1)) * 3];

    // Load halo (top/bottom)
    if (ty == 0 && y > 0)
        tile[0][tx + 1] = input[((y - 1) * w + x) * 3];
    if (ty == BLOCK - 1 && y < h - 1)
        tile[BLOCK + 1][tx + 1] = input[((y + 1) * w + x) * 3];

    if (tx == 0 && ty == 0 &&
        x > 0 && y > 0)
    {
        tile[0][0] =
            input[((y - 1) * w + (x - 1)) * 3];
    }

    if (tx == BLOCK - 1 && ty == 0 &&
        x < w - 1 && y > 0)
    {
        tile[0][BLOCK + 1] =
            input[((y - 1) * w + (x + 1)) * 3];
    }

    if (tx == 0 && ty == BLOCK - 1 &&
        x > 0 && y < h - 1)
    {
        tile[BLOCK + 1][0] =
            input[((y + 1) * w + (x - 1)) * 3];
    }

    if (tx == BLOCK - 1 &&
        ty == BLOCK - 1 &&
        x < w - 1 &&
        y < h - 1)
    {
        tile[BLOCK + 1][BLOCK + 1] =
            input[((y + 1) * w + (x + 1)) * 3];
    }
    if (x < w && y < h)
    {
        int idx = (y * w + x) * 3;

        output[idx] = input[idx];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
    }
    __syncthreads();

    if (x >= 1 && x < w - 1 && y >= 1 && y < h - 1)
    {
        int sum = 0;

        for (int ky = -1; ky <= 1; ky++)
            for (int kx = -1; kx <= 1; kx++)
                sum += tile[ty + 1 + ky][tx + 1 + kx];

        int idx = (y * w + x) * 3;
        unsigned char val = sum / 9;

        output[idx] = output[idx + 1] = output[idx + 2] = val;
    }
}