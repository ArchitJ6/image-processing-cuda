#pragma once

#include <opencv2/opencv.hpp>
#include <string>

struct Image
{
    int width;
    int height;
    cv::Mat mat;
};

Image loadImage(const std::string &filename);
void saveImage(const std::string &filename, const Image &img);