#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>

#include "kernels.h"
#include "image_io.h"

// ==========================
// 🔧 CLI PARSER
// ==========================
struct Config
{
    std::string filter = "grayscale";
    std::string input = "data/input.ppm";
    std::string output_prefix = "output/output";
    int threshold = 0;
};

Config parse_args(int argc, char **argv)
{
    Config cfg;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc)
            cfg.filter = argv[++i];
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            cfg.input = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            cfg.output_prefix = argv[++i];
        else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc)
            cfg.threshold = atoi(argv[++i]);
    }

    return cfg;
}

// ==========================
// 🚀 MAIN
// ==========================
int main(int argc, char **argv)
{
    try
    {
        Config cfg = parse_args(argc, argv);

        std::cout << "=== CUDA Image Processing ===\n";
        std::cout << "Filter: " << cfg.filter << "\n";

        // Load image
        Image img = loadPPM(cfg.input);
        std::cout << "Loaded: " << img.width << " x " << img.height << "\n";

        int size = img.width * img.height * 3;

        unsigned char *d_input, *d_output;
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);

        cudaMemcpy(d_input, img.data.data(), size, cudaMemcpyHostToDevice);

        std::vector<unsigned char> cpu_output(size);

        dim3 threads(16, 16);
        dim3 blocks((img.width + 15) / 16, (img.height + 15) / 16);

        // =========================
        // 🚀 GPU EXECUTION
        // =========================
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        if (cfg.filter == "grayscale")
        {
            grayscale_kernel<<<blocks, threads>>>(d_input, d_output, img.width, img.height);
        }
        else if (cfg.filter == "blur")
        {
            blur_shared_kernel<<<blocks, threads>>>(d_input, d_output, img.width, img.height);
        }
        else if (cfg.filter == "sobel")
        {
            sobel_shared_kernel<<<blocks, threads>>>(d_input, d_output, img.width, img.height, cfg.threshold);
        }
        else if (cfg.filter == "sharpen")
        {
            sharpen_kernel<<<blocks, threads>>>(d_input, d_output, img.width, img.height);
        }
        else
        {
            throw std::runtime_error("Unknown filter");
        }

        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_ms = 0;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        cudaMemcpy(img.data.data(), d_output, size, cudaMemcpyDeviceToHost);

        std::string gpu_out = cfg.output_prefix + "_gpu_" + cfg.filter + ".ppm";
        savePPM(gpu_out, img);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // =========================
        // 🧠 CPU EXECUTION
        // =========================
        auto cpu_start = std::chrono::high_resolution_clock::now();

        if (cfg.filter == "grayscale")
        {
            grayscale_cpu(img.data.data(), cpu_output.data(), img.width, img.height);
        }
        else if (cfg.filter == "blur")
        {
            blur_cpu(img.data.data(), cpu_output.data(), img.width, img.height);
        }
        else if (cfg.filter == "sobel")
        {
            sobel_cpu(img.data.data(), cpu_output.data(), img.width, img.height);
        }
        else if (cfg.filter == "sharpen")
        {
            sharpen_cpu(img.data.data(), cpu_output.data(), img.width, img.height);
        }

        auto cpu_end = std::chrono::high_resolution_clock::now();

        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        Image cpu_img = img;
        cpu_img.data = cpu_output;

        std::string cpu_out = cfg.output_prefix + "_cpu_" + cfg.filter + ".ppm";
        savePPM(cpu_out, cpu_img);

        // =========================
        // 📊 RESULTS
        // =========================
        std::cout << "\n===== Benchmark =====\n";
std::cout << "---------------------------------------\n";
std::cout << "Filter      GPU(ms)   CPU(ms)   Speedup\n";
std::cout << "---------------------------------------\n";

printf("%-10s %-9.3f %-9.3f %.2fx\n",
       cfg.filter.c_str(),
       gpu_ms,
       cpu_ms,
       cpu_ms / gpu_ms);

        std::cout << "\nOutputs:\n";
        std::cout << "GPU: " << gpu_out << "\n";
        std::cout << "CPU: " << cpu_out << "\n";

        cudaFree(d_input);
        cudaFree(d_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}