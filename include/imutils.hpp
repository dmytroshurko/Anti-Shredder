#ifndef IMUTILS_HPP_
#define IMUTILS_HPP_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/*
 * @brief Resizes an image using a computed ratio between new and old width (or
 * height).
 *
 * @param src The input image.
 * @param dst The output image.
 * @param width The desired width of the output image.
 * @param height The desired height of the output image.
 */
void resize(const cv::Mat& src, cv::Mat& dst, int width = 0, int height = 0);

#endif  // IMUTILS_HPP_
