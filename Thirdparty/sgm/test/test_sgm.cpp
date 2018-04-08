/*@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file test_sgm.cpp
 * @brief test_sgm.cpp
 */

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

#include "utility.h"
#include "sgm_stereo.h"
#include "test_utility.h"
#include<iomanip>

#include <vector>
#include <dirent.h>

using namespace std;

float getInterpolate(const cv::Mat mat, const float x, const float y)
{
    int ix = (int)x;
    int iy = (int)y;

    float tl = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy, ix)[0]);
    float tr = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy, ix+1)[0]);
    float bl = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy+1, ix)[0]);
    float br = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy+1, ix+1)[0]);

    float dx = x - ix;
    float dy = y - iy;

    float topInt = dx * tr + (1-dx) * tl;
    float botInt = dx * br + (1-dx) * bl;
    float leftInt = dy * bl + (1-dy) * tl;
    float rightInt = dy * br + (1-dy) * tr;

    float value = dx * rightInt + (1-dx) * leftInt;
    return value;

}

int main(int argc, char **argv)
{

    bool debug = false;

    if(debug)
    {
        std::stringstream id;
        id << std::setfill('0') << std::setw(6)<<857;
        std::string disparityPath = "/home/jiatianwu/dso/05/disparity/" + id.str() + ".png";
        std::string leftPath = "/home/jiatianwu/dso/05/image_0/" + id.str() + ".png";
        std::string rightPath = "/home/jiatianwu/dso/05/image_1/" + id.str() + ".png";

        cv::Mat disparity = cvLoadImage(disparityPath.c_str());
        printf(" %d %d \n", disparity.channels(), disparity.depth());
        IplImage* img_left = cvLoadImage(leftPath.c_str());
        IplImage* img_right = cvLoadImage(rightPath.c_str());

        IplImage* img_show = cvCreateImage( cvSize(img_left->width, img_left->height + img_right->height),IPL_DEPTH_8U,3);
        CvRect rect=cvRect(0,0,img_left->width,img_right->height);
        cvSetImageROI(img_show,rect);
        cvCopy(img_left,img_show);

        cvResetImageROI(img_show);
        rect=cvRect(0,img_left->height,img_right->width,img_right->height);
        cvSetImageROI(img_show,rect);
        cvCopy(img_right,img_show);
        cvResetImageROI(img_show);
        cv::Mat show = cv::Mat(img_show);

        cv::Point left_temp;
        cv::Point right_temp;
        int counter = 0;

        for(int i = 0; i < img_left->width; i++)
            for(int j = 0; j < img_left->height; j++)
            {
                counter++;
                left_temp.x = i; left_temp.y = j;

//                int sgmDisparityOrig = static_cast<u_char>(disparity.at<cv::Vec3b>(j, i)[0]);
//                int sgmDisparityOrig1 = static_cast<u_char>(disparity.at<cv::Vec3b>(j, i)[1]);
//                int sgmDisparityOrig2 = static_cast<u_char>(disparity.at<cv::Vec3b>(j, i)[2]);

                float sgmDisparity = getInterpolate(disparity, left_temp.x, left_temp.y);
                right_temp.x = i - sgmDisparity; right_temp.y = j + img_left->height;

                printf("%d %d %f \n", i, j, sgmDisparity);

                if(counter%4000 == 0)
                {
                    cv::circle(show, left_temp, 3, (0,255,255), 2);
                    cv::circle(show, right_temp, 3, (0,255,255), 2);
                    cv::line(show, left_temp, right_temp,(rand()%255, rand()%255, rand()%255), 2);
                }
            }

        cv::imshow("show", show);
        cvWaitKey(0);

    }
    else
    {
        for (int s = 0; s < 22; s++)
        {
            std::stringstream sequence;
            sequence << std::setfill('0') << std::setw(2) << s;

            std::string left_img_path_orig = "/home/bobin/data/kitti/sequences/" + sequence.str() + "/image_0";
            std::string right_img_path_orig = "/home/bobin/data/kitti/sequences/" + sequence.str() + "/image_1";
            std::string output_colored_disparitymap_orig = "/home/bobin/data/kitti/sequences/" + sequence.str() + "/disparity";
            system(("mkdir " + output_colored_disparitymap_orig).c_str());

            struct dirent *ptr;
            DIR *dir;
            string PATH = left_img_path_orig;
            dir=opendir(PATH.c_str());
            int counter = 0;

            while((ptr=readdir(dir))!=NULL)
            {
                if(ptr->d_name[0] == '.')
                    continue;
                counter++;
            }

            for(int number = 0; number < counter; number++)
            {
                std::stringstream ss;
                ss << std::setfill('0') << std::setw(6) << number;

                std::string left_img_path = left_img_path_orig + "/" + ss.str() + ".png";
                std::string right_img_path = right_img_path_orig + "/" + ss.str() + ".png";
                std::string output_colored_disparitymap = output_colored_disparitymap_orig + "/" + ss.str() +".png";
                int levels_disparity = 200;

                unsigned char* img_left = NULL, *img_right = NULL;
                size_t width, height;
                loadImageRGB(left_img_path.c_str(), width, height, img_left);/*loading the gray image as RGB is fine, [gray, gray, gray]*/
                loadImageRGB(right_img_path.c_str(), width, height, img_right);
                int size = width*height, size3 = width*height * 3;
                unsigned char* img_tmp = new unsigned char[size3];
                float* guidance_left = new float[size3];

                std::cout << "Begin to calculate the disparity map..." << std::endl;
                deeprob::cv::SGMStereo matcher((int)width, (int)height, levels_disparity, 3, 3);
                deeprob::cv::BoxFilter(img_left, img_tmp, 1, 1, width, height, 3);/*pre-smoothing on guidance_left*/
                for (int i = 0; i < size3; ++i) guidance_left[i] = img_tmp[i];

                matcher.setImage(img_left, img_right, 3);
                matcher.setGuidance(guidance_left, NULL, 3);
                matcher.calDisparity(); /*calculate the disparity map*/

                cv::Mat disparity(height, width, CV_8UC3);
                std::cout << "Done! Sequence "<< s << " number " << number << " is saved."<< std::endl;
                showDataFalseColor(matcher._disparity[0], width, height, disparity.data, output_colored_disparitymap.c_str(), true);/*save the disparity map in false color*/
                cv::imshow("disparity", disparity);

                cv::waitKey(10);
                delete[]img_tmp;
                delete[]guidance_left;
                delete[]img_left;
                delete[]img_right;

            }
        }
    }

    return 1;
}
