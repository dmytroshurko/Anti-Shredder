#ifndef ANTI_SHREDDER_HPP_
#define ANTI_SHREDDER_HPP_

#include <opencv2/core.hpp>
#include <vector>

/*
 * @brief Detects rectangles around the paper strips in the image.
 *
 * @param src The input image.
 * @return The vector with detected rectangles.
 */
std::vector<cv::RotatedRect> detect_rectangles(const cv::Mat& src);

#endif  // ANTI_SHREDDER_HPP_
