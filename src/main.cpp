#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "anti_shredder.hpp"
#include "imutils.hpp"

int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv,
                               "{@input | image.jpg | input image}");

  cv::Mat src = cv::imread(parser.get<cv::String>("@input"), cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cout << "Could not open or find the image!\n";
    std::cout << "Usage: " << argv[0] << " <Input image>\n";
    return -1;
  }

  cv::Mat resized;
  Resize(src, resized, 800);
  cv::imshow("Input image", resized);

  std::vector<cv::RotatedRect> rectangles = DetectRectangles(resized);
  std::vector<cv::Mat> image_parts = CropRectangles(resized, rectangles, 2);
  cv::Mat res = CombineImageParts(image_parts, 20);

  cv::imshow("Output image", res);
  cv::waitKey();

  return 0;
}
