/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file test_local.cpp
 * @brief test_local.cpp
 */

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "utility.h"
#include "local_stereo.h"
#include "test_utility.h"

int main()
{
	char* left_img_path = "/home/bobin/data/kitti/sequences/04/image_0/000000.png", *right_img_path = "/home/bobin/data/kitti/sequences/04/image_1/000000.png";
	char* output_colored_disparitymap = "example_colored_disparitymap.png";
	int levels_disparity = 200;

	unsigned char* img_left = NULL, *img_right = NULL;
	size_t width, height;
	loadImageRGB(left_img_path, width, height, img_left, 2);/*loading the gray image as RGB is fine, [gray, gray, gray]*/
	loadImageRGB(right_img_path, width, height, img_right, 2);
	int size3 = width * height * 3;
	unsigned char* img_tmp = new unsigned char[size3];
	float* guidance_left = new float[size3];
	deeprob::cv::BoxFilter(img_left, img_tmp, 1, 1, width, height, 3);/*pre-smoothing on guidance_left*/
	for (int i = 0; i < size3; ++i) guidance_left[i] = img_tmp[i];

	unsigned char* img_guidance_left_high = NULL;
	size_t width_h, height_h;
	loadImageRGB(left_img_path, width_h, height_h, img_guidance_left_high, 1);
	int size3_h = width_h * height_h * 3;
	float* guidance_left_high = new float[size3_h];
	for (int i = 0; i < size3_h; ++i) guidance_left_high[i] = img_guidance_left_high[i];

	std::cout << "Begin to calculate the disparity map..." << std::endl;
	deeprob::cv::LocalStereo matcher((int)width_h, (int)height_h, levels_disparity, 3, 3);

	matcher.setImage(img_left, img_right, 3);
	matcher.setGuidance(guidance_left, guidance_left_high, NULL, 3);
	matcher.calDisparity(); /*calculate the disparity map*/

	std::cout << "Done! Result is saved." << std::endl;
	showDataFalseColor(matcher._disparity_output, width_h, height_h, NULL, output_colored_disparitymap);/*save the disparity map in false color*/

	delete[]guidance_left_high;
	delete[]img_tmp;
	delete[]guidance_left;
	delete[]img_left;
	delete[]img_right;
	return (1);
}
