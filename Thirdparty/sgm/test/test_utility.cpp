/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file test_utility.cpp
 * @brief test_utility.cpp
 */

#include "test_utility.h"

bool loadImageRGB(const std::string& filename, size_t& width, size_t& height, unsigned char*& data, int downsample_factor)
{
	IplImage* img = cvLoadImage(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (!img) return false;
	if (downsample_factor > 1)
	{
		int w_src = img->width, h_src = img->height;
		IplImage *img_resized = cvCreateImage(cvSize(w_src / downsample_factor, h_src / downsample_factor), 8, 3);
		cvResize(img, img_resized);
		cvReleaseImage(&img);
		img = img_resized;
	}
	height = img->height, width = img->width;
	if (data != NULL)
	{
		delete[]data;
	}
	data = new unsigned char[width*height * 3];
	for (int y = 0; y < (int)height; ++y)for (int x = 0; x < (int)width; ++x)
	{
		int idx_pt = y*width + x;
		data[idx_pt * 3] = ((uchar*)(img->imageData + img->widthStep*y))[3 * x + 2];
		data[idx_pt * 3 + 1] = ((uchar*)(img->imageData + img->widthStep*y))[3 * x + 1];
		data[idx_pt * 3 + 2] = ((uchar*)(img->imageData + img->widthStep*y))[3 * x];
	}
	cvReleaseImage(&img);
	return true;
}

bool saveImageColor(const std::string& filename, size_t width, size_t height, unsigned char*& data)
{
	IplImage* img = cvCreateImage(cvSize(width, height), 8, 1);
	if (!img)
	{
		return false;
	}
	for (int y = 0; y < (int)height; ++y)for (int x = 0; x < (int)width; ++x)
	{
		int idx_pt = y*width + x;
 /*
		for (int cc = 0; cc < 3; ++cc)
		{
			((uchar*)(img->imageData + img->widthStep*y))[3 * x + cc] = data[idx_pt * 3 + 2 - cc];
		}*/

			((uchar*)(img->imageData + img->widthStep*y))[x] = data[idx_pt];
	}
	cvSaveImage(filename.c_str(), img);
	cvReleaseImage(&img);
	return true;
}
