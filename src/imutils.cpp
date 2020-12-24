#include "imutils.hpp"

#include <opencv2/imgproc.hpp>

void Resize(const cv::Mat& src, cv::Mat& dst, int width, int height) noexcept {
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

void RotateWithoutCropping(const cv::Mat& src, cv::Mat& dst,
                           double angle) noexcept {
  int w = src.cols;
  int h = src.rows;
  int cx = w / 2;
  int cy = h / 2;

  cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), -angle, 1.0);
  double cos = cv::abs(M.at<double>(0, 0));
  double sin = cv::abs(M.at<double>(0, 1));

  // compute the new width and height of the image
  int nw = static_cast<int>((h * sin) + (w * cos));
  int nh = static_cast<int>((h * cos) + (w * sin));

  // adjust the rotation matrix to take into account translation
  M.at<double>(0, 2) += (nw / 2) - cx;
  M.at<double>(1, 2) += (nh / 2) - cy;

  cv::warpAffine(src, dst, M, cv::Size(nw, nh));
}
