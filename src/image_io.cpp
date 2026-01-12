#include "image_io.h"
#include <fstream>
#include <iostream>

Image loadPPM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open file");
    }

    std::string format;
    file >> format;

    if (format != "P6") {
        throw std::runtime_error("Invalid PPM format (must be P6)");
    }

    // Skip comments
    char ch;
    file >> std::ws;
    while (file.peek() == '#') {
        std::string comment;
        std::getline(file, comment);
    }

    int width, height, maxval;
    file >> width >> height >> maxval;
    file.get(); // consume newline

    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height * 3);

    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());

    if (!file) {
        throw std::runtime_error("Error reading pixel data");
    }

    return img;
}

void savePPM(const std::string& filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);

    file << "P6\n" << img.width << " " << img.height << "\n255\n";
    file.write((char*)img.data.data(), img.data.size());
}