#ifndef ANTI_SHREDDER_HPP_
#define ANTI_SHREDDER_HPP_

#include <opencv2/core.hpp>
#include <string>
#include <utility>
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

/*
 * @brief Calculates the histogram of the part (left or right side) of the
 * image.
 *
 * @param src The input image.
 * @param side_width The width of the side.
 * @param side The label of the side ('l' - left side, 'r' - right side).
 * @return A calculated histogram.
 */
cv::Mat CalculateSideHist(const cv::Mat& src, int side_width, char side);

/*
 * @brief Compares the histograms of two images.
 *
 * @param a_hist The histograms of the first image.
 * @param b_hist The histograms of the second image.
 * @return Best metric and best matching sides of the first and second images.
 */
std::pair<double, std::string> CompareHistograms(
    const std::pair<cv::Mat, cv::Mat>& a_hist,
    const std::pair<cv::Mat, cv::Mat>& b_hist);

/*
 * @brief Combines parts of an image into one image.
 *
 * @param image_parts A vector with extracted image strips.
 * @param side_width The width of the side.
 * @return A combined image.
 */
cv::Mat CombineImageParts(std::vector<cv::Mat> image_parts,
                          int side_width = 30);

#endif  // ANTI_SHREDDER_HPP_
