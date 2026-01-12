#pragma once
#include <vector>
#include <string>

struct Image {
    int width;
    int height;
    std::vector<unsigned char> data; // RGB
};

Image loadPPM(const std::string& filename);
void savePPM(const std::string& filename, const Image& img);