#include "anti_shredder.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <utility>

#include "imutils.hpp"

std::vector<cv::RotatedRect> DetectRectangles(const cv::Mat& src) {
  cv::Mat src_gray;
  cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

  cv::GaussianBlur(src_gray, src_gray, cv::Size(3, 3), 0);

  cv::Mat canny_output;
  cv::Canny(src_gray, canny_output, 50, 150);

  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(12, 12));
  cv::morphologyEx(canny_output, canny_output, cv::MORPH_CLOSE, element,
                   cv::Point(-1, -1), 2);

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

std::vector<cv::Mat> CropRectangles(
    const cv::Mat& src, const std::vector<cv::RotatedRect>& rectangles,
    int approx_iterations) {
  std::vector<cv::Mat> rect_images;

  for (const auto& rectangle : rectangles) {
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);

    cv::Point2f pts[4];
    rectangle.points(pts);

    cv::Point vertices[4];
    for (int i = 0; i < 4; ++i) {
      vertices[i] = pts[i];
    }

    // create a mask for the current rectangle
    cv::fillConvexPoly(mask, vertices, 4, cv::Scalar(255, 255, 255));

    cv::Mat selected_rect;
    cv::bitwise_and(src, src, selected_rect, mask);

    cv::Mat rect_img = selected_rect(rectangle.boundingRect());

    RotateWithoutCropping(rect_img, rect_img, -rectangle.angle);

    // make sure that the rectangle is in a vertical position
    if (rect_img.rows < rect_img.cols) {
      cv::rotate(rect_img, rect_img, cv::ROTATE_90_CLOCKWISE);
    }

    int rw = rectangle.size.width;
    int rh = rectangle.size.height;
    if (rh < rw) {
      std::swap(rw, rh);
    }

    CropCenter(rect_img, rect_img, cv::Size(rw, rh));

    for (int i = 0; i < approx_iterations; ++i) {
      cv::Mat gray;
      cv::cvtColor(rect_img, gray, cv::COLOR_BGR2GRAY);

      cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);

      cv::Mat canny_output;
      cv::Canny(gray, canny_output, 100, 200);

      // calculate the up-right bounding rectangle of a non-zero pixels of
      // gray-scale image
      cv::Rect roi = cv::boundingRect(canny_output);

      rect_img = rect_img(roi);
    }

    rect_images.emplace_back(rect_img);
  }

  return rect_images;
}
