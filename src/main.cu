#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include "kernels.h"
#include "image_io.h"
#include <filesystem>
#include <iomanip>

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

void checkCuda(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

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
    if (argc == 1)
    {
        std::cout
            << "Usage:\n"
            << "app.exe --filter grayscale --input image.png\n"
            << "app.exe --filter blur --input image.png\n"
            << "app.exe --filter sobel --input image.png --threshold 100\n"
            << "app.exe --filter sharpen --input image.png\n";

        return 0;
    }

    auto app_start =
        std::chrono::high_resolution_clock::now();

    std::cout << "Program started\n";
    std::cout.flush();

    unsigned char *d_input = nullptr;
    unsigned char *d_output = nullptr;

    try
    {
        Config cfg = parse_args(argc, argv);

        std::filesystem::create_directories("output");

        if (cfg.threshold > 0 && cfg.filter != "sobel")
        {
            std::cout
                << "[Warning] Threshold only applies to Sobel filter.\n";
        }

        std::cout << "=== CUDA Image Processing ===\n";
        std::cout << "OpenCV: " << CV_VERSION << "\n";

        cudaDeviceProp prop;
        size_t free_mem, total_mem;

        checkCuda(
            cudaMemGetInfo(
                &free_mem,
                &total_mem));

        std::cout
            << "Available GPU Memory: "
            << free_mem / (1024 * 1024)
            << " MB\n";
        checkCuda(cudaGetDeviceProperties(&prop, 0));

        std::cout
            << "GPU: "
            << prop.name
            << "\n";

        std::cout
            << "Compute Capability: "
            << prop.major
            << "."
            << prop.minor
            << "\n";

        std::cout
            << "Multiprocessors: "
            << prop.multiProcessorCount
            << "\n";

        std::cout
            << "Max Threads Per Block: "
            << prop.maxThreadsPerBlock
            << "\n";

        std::cout
            << "Global Memory: "
            << (prop.totalGlobalMem / (1024 * 1024))
            << " MB\n";

        std::cout << "Filter: " << cfg.filter << "\n";

        // Load image
        Image img = loadImage(cfg.input);

        if (cfg.filter == "blur" ||
            cfg.filter == "sobel" ||
            cfg.filter == "sharpen")
        {
            cv::Mat gray;

            cv::cvtColor(
                img.mat,
                gray,
                cv::COLOR_BGR2GRAY);

            cv::cvtColor(
                gray,
                img.mat,
                cv::COLOR_GRAY2BGR);
        }

        std::cout << "Loaded: " << img.width << " x " << img.height << "\n";
        std::cout
            << "Pixels: "
            << img.width * img.height
            << " ("
            << img.width << "x"
            << img.height
            << ")\n";

        size_t size = img.mat.total() * img.mat.elemSize();

        checkCuda(cudaMalloc(&d_input, size));
        checkCuda(cudaMalloc(&d_output, size));

        checkCuda(
            cudaMemcpy(
                d_input,
                img.mat.data,
                size,
                cudaMemcpyHostToDevice));

        std::vector<unsigned char> cpu_output(size);

        dim3 threads(16, 16);
        dim3 blocks((img.width + 15) / 16, (img.height + 15) / 16);

        // =========================
        // 🚀 GPU EXECUTION
        // =========================
        cudaEvent_t start, stop;
        checkCuda(cudaEventCreate(&start));
        checkCuda(cudaEventCreate(&stop));

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

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_ms = 0;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        cv::Mat gpu_result(
            img.height,
            img.width,
            CV_8UC3);

        checkCuda(
            cudaMemcpy(
                gpu_result.data,
                d_output,
                size,
                cudaMemcpyDeviceToHost));

        Image gpu_img;
        gpu_img.width = img.width;
        gpu_img.height = img.height;
        gpu_img.mat = gpu_result;

        std::string gpu_out =
            cfg.output_prefix + "_gpu_" + cfg.filter + ".png";

        saveImage(gpu_out, gpu_img);

        checkCuda(cudaEventDestroy(start));
        checkCuda(cudaEventDestroy(stop));

        // =========================
        // 🧠 CPU EXECUTION
        // =========================
        auto cpu_start = std::chrono::high_resolution_clock::now();

        if (cfg.filter == "grayscale")
        {
            grayscale_cpu(img.mat.data, cpu_output.data(), img.width, img.height);
        }
        else if (cfg.filter == "blur")
        {
            blur_cpu(img.mat.data, cpu_output.data(), img.width, img.height);
        }
        else if (cfg.filter == "sobel")
        {
            sobel_cpu(img.mat.data, cpu_output.data(), img.width, img.height);
        }
        else if (cfg.filter == "sharpen")
        {
            sharpen_cpu(img.mat.data, cpu_output.data(), img.width, img.height);
        }

        auto cpu_end = std::chrono::high_resolution_clock::now();

        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        cv::Mat cpu_result(
            img.height,
            img.width,
            CV_8UC3);

        memcpy(
            cpu_result.data,
            cpu_output.data(),
            size);

        Image cpu_img;
        cpu_img.width = img.width;
        cpu_img.height = img.height;
        cpu_img.mat = cpu_result;

        std::string cpu_out =
            cfg.output_prefix + "_cpu_" + cfg.filter + ".png";

        saveImage(cpu_out, cpu_img);

        // =========================
        // 📊 RESULTS
        // =========================
        std::cout << "\n===== Benchmark =====\n";
        std::cout << std::string(52, '-') << "\n";
        std::cout
            << std::left
            << std::setw(12) << "Filter"
            << std::setw(12) << "GPU(ms)"
            << std::setw(12) << "CPU(ms)"
            << std::setw(12) << "Speedup"
            << "\n";
        std::cout << std::string(52, '-') << "\n";

        double speedup = cpu_ms / gpu_ms;

        std::cout
            << std::left
            << std::setw(12) << cfg.filter
            << std::setw(12) << std::fixed << std::setprecision(3) << gpu_ms
            << std::setw(12) << cpu_ms
            << std::setw(12) << speedup
            << "\n";

        std::cout << "\nOutputs:\n";
        std::cout << "GPU: " << gpu_out << "\n";
        std::cout << "CPU: " << cpu_out << "\n";

        bool file_exists =
            std::filesystem::exists("output/benchmark.csv");

        std::ofstream csv(
            "output/benchmark.csv",
            std::ios::app);

        if (!file_exists)
        {
            csv << "filter,image_size,gpu_ms,cpu_ms,speedup\n";
        }

        csv
            << cfg.filter << ","
            << img.width << "x" << img.height << ","
            << gpu_ms << ","
            << cpu_ms << ","
            << speedup << "\n";

        cudaFree(d_input);
        cudaFree(d_output);
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";

        if (d_input)
            cudaFree(d_input);

        if (d_output)
            cudaFree(d_output);

        return 1;
    }
    auto app_end =
        std::chrono::high_resolution_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(
            app_end - app_start)
            .count();

    std::cout
        << "\nTotal Runtime: "
        << total_ms
        << " ms\n";
    return 0;
}