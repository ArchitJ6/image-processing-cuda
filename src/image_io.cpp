#include "image_io.h"
#include <stdexcept>

Image loadImage(const std::string &filename)
{
    cv::Mat img = cv::imread(filename);

    if (img.empty())
        throw std::runtime_error("Failed to load image: " + filename);

    Image out;
    out.width = img.cols;
    out.height = img.rows;
    out.mat = img;

    return out;
}

void saveImage(const std::string &filename, const Image &img)
{
    cv::imwrite(filename, img.mat);
}