#include <iostream>
#include <chrono>
#include "kernels.h"
#include "image_io.h"

int main() {
    try{
    std::cout << "Program started\n";

    std::cout << "Loading image...\n";

    Image img = loadPPM("data/input.ppm");

    std::cout << "Loaded image: "
          << img.width << " x " << img.height << "\n";

    int size = img.width * img.height * 3;

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // CPU buffers
    std::vector<unsigned char> cpu_output(size);

    cudaMemcpy(d_input, img.data.data(), size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(
        (img.width + 15) / 16,
        (img.height + 15) / 16
    );

    // =========================
    // 🚀 GPU TIMING (CUDA EVENTS)
    // =========================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    grayscale_kernel<<<blocks, threads>>>(
        d_input, d_output,
        img.width, img.height
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(img.data.data(), d_output, size, cudaMemcpyDeviceToHost);

    savePPM("output/output_gpu.ppm", img);

    // =========================
    // 🧠 CPU TIMING
    // =========================
    auto cpu_start = std::chrono::high_resolution_clock::now();

    grayscale_cpu(
        img.data.data(),
        cpu_output.data(),
        img.width,
        img.height
    );

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Save CPU output
    Image cpu_img = img;
    cpu_img.data = cpu_output;
    savePPM("output/output_cpu.ppm", cpu_img);

    // =========================
    // 📊 RESULTS
    // =========================
    std::cout << "\n===== Performance Comparison =====\n";
    std::cout << "GPU Time: " << gpu_ms << " ms\n";
    std::cout << "CPU Time: " << cpu_ms << " ms\n";
    std::cout << "Speedup: " << cpu_ms / gpu_ms << "x\n";

    cudaFree(d_input);
    cudaFree(d_output);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}