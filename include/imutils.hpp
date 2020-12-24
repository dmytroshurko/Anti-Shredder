#ifndef IMUTILS_HPP_
#define IMUTILS_HPP_

#include <opencv2/core.hpp>

/*
 * @brief Resizes the image using a computed ratio between new and old width (or
 * height).
 *
 * @param src The input image.
 * @param dst The output image.
 * @param width The desired width of the output image.
 * @param height The desired height of the output image.
 */
void Resize(const cv::Mat& src, cv::Mat& dst, int width = 0,
            int height = 0) noexcept;

/*
 * @brief Rotates the image without cropping.
 *
 * @param src The input image.
 * @param dst The output image.
 * @param angle The image rotation angle (in degrees).
 */
void RotateWithoutCropping(const cv::Mat& src, cv::Mat& dst,
                           double angle) noexcept;

#endif  // IMUTILS_HPP_
