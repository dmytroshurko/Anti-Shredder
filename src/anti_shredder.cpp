#include "anti_shredder.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>

std::vector<cv::RotatedRect> detect_rectangles(const cv::Mat& src) {
  cv::Mat src_gray;
  cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

  cv::GaussianBlur(src_gray, src_gray, cv::Size(3, 3), 0);

  cv::Mat canny_output;
  cv::Canny(src_gray, canny_output, 50, 150);

  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
  cv::dilate(canny_output, canny_output, element, cv::Point(-1, -1), 2);
  cv::erode(canny_output, canny_output, element, cv::Point(-1, -1), 2);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(canny_output, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::RotatedRect> rectangles(contours.size());
  for (size_t i = 0; i < contours.size(); ++i) {
    rectangles[i] = cv::minAreaRect(contours[i]);
  }

  // find the rectangle with the largest area
  auto comp = [](const auto& a, const auto& b) {
    return a.size.area() < b.size.area();
  };
  auto max_area_rect =
      *std::max_element(rectangles.begin(), rectangles.end(), comp);

  // remove rectangles whose area is less than half the area of the largest
  // rectangle
  double max_area = max_area_rect.size.area();
  auto delete_rectangle = [=](const auto& a) {
    return a.size.area() < (max_area / 2);
  };
  rectangles.erase(
      std::remove_if(rectangles.begin(), rectangles.end(), delete_rectangle),
      rectangles.end());

  return rectangles;
}
