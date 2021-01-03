#ifndef ANTI_SHREDDER_HPP_
#define ANTI_SHREDDER_HPP_

#include <opencv2/core.hpp>
#include <vector>

/*
 * @brief Detects rectangles around the paper strips in the image.
 *
 * @param src The input image.
 * @return A vector with detected rectangles.
 */
std::vector<cv::RotatedRect> DetectRectangles(const cv::Mat& src);

/*
 * @brief Returns image strips from the input image.
 *
 * @param src The input image.
 * @param rectangles A vector with detected rectangles.
 * @param approx_iterations Number of iterations to approximate the rectangle.
 * @return A vector with extracted image strips.
 */
std::vector<cv::Mat> CropRectangles(
    const cv::Mat& src, const std::vector<cv::RotatedRect>& rectangles,
    int approx_iterations = 1);

#endif  // ANTI_SHREDDER_HPP_
