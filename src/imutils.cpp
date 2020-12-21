#include "imutils.hpp"

#include <opencv2/imgproc.hpp>

void resize(const cv::Mat& src, cv::Mat& dst, int width, int height) noexcept {
  int w = src.cols;
  int h = src.rows;

  if (width == 0 && height == 0) {
    dst = src.clone();
    return;
  }

  cv::Size dsize{};
  if (width == 0) {
    double ratio = height / static_cast<double>(h);
    dsize.width = static_cast<int>(w * ratio);
    dsize.height = height;
  } else {
    double ratio = width / static_cast<double>(w);
    dsize.width = width;
    dsize.height = static_cast<double>(h * ratio);
  }

  cv::resize(src, dst, dsize);
}
