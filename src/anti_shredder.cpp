#include "anti_shredder.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

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

cv::Mat CalculateSideHist(const cv::Mat& src, int side_width, char side) {
  if (side_width <= 0 || side_width >= src.cols) {
    throw std::out_of_range{"CalculateSideHist"};
  }

  if (side_width <= 0 || (side != 'l' && side != 'r')) {
    throw std::invalid_argument{"CalculateSideHist"};
  }

  cv::Mat hsv_side;
  if (side == 'l') {
    cv::cvtColor(src(cv::Range::all(), cv::Range(0, side_width)), hsv_side,
                 cv::COLOR_BGR2HSV);
  } else {
    int w = src.cols;
    cv::cvtColor(src(cv::Range::all(), cv::Range(w - side_width, w)), hsv_side,
                 cv::COLOR_BGR2HSV);
  }

  int h_bins = 50;
  int s_bins = 60;
  int hist_size[] = {h_bins, s_bins};

  // hue varies from 0 to 179, saturation from 0 to 255
  float h_ranges[] = {0, 180};
  float s_ranges[] = {0, 256};

  const float* ranges[] = {h_ranges, s_ranges};

  // use the 0-th and 1-st channels
  int channels[] = {0, 1};

  cv::Mat hist_side;
  cv::calcHist(&hsv_side, 1, channels, cv::Mat(), hist_side, 2, hist_size,
               ranges);
  cv::normalize(hist_side, hist_side, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

  return hist_side;
}

std::pair<double, std::string> CompareHistograms(
    const std::pair<cv::Mat, cv::Mat>& a_hist,
    const std::pair<cv::Mat, cv::Mat>& b_hist) {
  double best_metric = 0.0;
  std::string best_sides{""};

  double metric = 0;
  metric = cv::compareHist(a_hist.first, b_hist.first, cv::HISTCMP_CORREL);
  if (metric > best_metric) {
    best_metric = metric;
    best_sides = "ll";
  }

  metric = cv::compareHist(a_hist.first, b_hist.second, cv::HISTCMP_CORREL);
  if (metric > best_metric) {
    best_metric = metric;
    best_sides = "lr";
  }

  metric = cv::compareHist(a_hist.second, b_hist.first, cv::HISTCMP_CORREL);
  if (metric > best_metric) {
    best_metric = metric;
    best_sides = "rl";
  }

  metric = cv::compareHist(a_hist.second, b_hist.second, cv::HISTCMP_CORREL);
  if (metric > best_metric) {
    best_metric = metric;
    best_sides = "rr";
  }

  return std::make_pair(best_metric, best_sides);
}

cv::Mat CombineImageParts(std::vector<cv::Mat> image_parts, int side_width) {
  int max_img_h = 0;
  std::vector<std::pair<cv::Mat, cv::Mat>> histograms;
  for (const auto& img : image_parts) {
    max_img_h = std::max(max_img_h, img.rows);
    cv::Mat hist_l = CalculateSideHist(img, side_width, 'l');
    cv::Mat hist_r = CalculateSideHist(img, side_width, 'r');
    histograms.emplace_back(std::make_pair(hist_l, hist_r));
  }

  // resize images
  for (auto& img : image_parts) {
    Resize(img, img, 0, max_img_h);
  }

  while (image_parts.size() > 1) {
    cv::Mat img_a = image_parts.back();
    auto a_hist = histograms.back();
    image_parts.pop_back();
    histograms.pop_back();

    size_t best_match_idx = 0;
    double best_metric = 0.0;
    std::string best_sides{""};

    for (size_t i = 0; i < histograms.size(); ++i) {
      const auto& b_hist = histograms[i];

      auto [metric, sides] = CompareHistograms(a_hist, b_hist);
      if (metric > best_metric) {
        best_match_idx = i;
        best_metric = metric;
        best_sides = sides;
      }
    }

    cv::Mat img_b = image_parts[best_match_idx];
    image_parts.erase(image_parts.begin() + best_match_idx);
    histograms.erase(histograms.begin() + best_match_idx);

    if (best_sides[0] == 'l') {
      cv::rotate(img_a, img_a, cv::ROTATE_180);
    }
    if (best_sides[1] == 'r') {
      cv::rotate(img_b, img_b, cv::ROTATE_180);
    }

    cv::Mat img_new;
    cv::hconcat(img_a, img_b, img_new);

    cv::Mat hist_l = CalculateSideHist(img_new, side_width, 'l');
    cv::Mat hist_r = CalculateSideHist(img_new, side_width, 'r');

    image_parts.emplace_back(img_new);
    histograms.emplace_back(std::make_pair(hist_l, hist_r));
  }

  return image_parts[0];
}
