/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file test_utility.h
 * @brief test_utility.h
 */

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "utility.h"

bool loadImageRGB(const std::string& filename, size_t& width, size_t& height, unsigned char*& data, int downsample_factor = 1);

bool saveImageColor(const std::string& filename, size_t width, size_t height, unsigned char*& data);

template<typename T>
void showDataFalseColor(T*data, int width, int height, unsigned char* texture, const char* filename_save, bool used_bg_color = false)
{
	bool newTextureBuf = false;
	if (texture == NULL){
		texture = new unsigned char[width*height * 3];
		memset(texture, 0, sizeof(unsigned char)*width*height * 3);
		newTextureBuf = true;
	}
	/*visualization*/
	if (used_bg_color) deeprob::cv::displayincolor(texture, data, width, height);
	else deeprob::cv::displayincolor(texture, data, width, height, 0);
	if (filename_save != NULL) saveImageColor(filename_save, (size_t)width, (size_t)height, data);
	if (newTextureBuf == true){ delete[]texture; texture = NULL; }
}
