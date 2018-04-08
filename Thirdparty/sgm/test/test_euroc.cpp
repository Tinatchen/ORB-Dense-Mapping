//
// Created by bobin on 17-11-18.
//

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
#include <fstream>
#include <sstream>
#include <vector>
#include <dirent.h>

using namespace std;

float getInterpolate(const cv::Mat mat, const float x, const float y) {
    int ix = (int) x;
    int iy = (int) y;

    float tl = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy, ix)[0]);
    float tr = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy, ix + 1)[0]);
    float bl = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy + 1, ix)[0]);
    float br = (float) static_cast<u_char>(mat.at<cv::Vec3b>(iy + 1, ix + 1)[0]);

    float dx = x - ix;
    float dy = y - iy;

    float topInt = dx * tr + (1 - dx) * tl;
    float botInt = dx * br + (1 - dx) * bl;
    float leftInt = dy * bl + (1 - dy) * tl;
    float rightInt = dy * br + (1 - dy) * tr;

    float value = dx * rightInt + (1 - dx) * leftInt;
    return value;

}

void LoadImages(const string &strPathLeft, const string &strPathRight, const string &strPathDepth, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<string> &vstrImageDepth, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if (s[0] == '#') continue;
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            vstrImageDepth.push_back(strPathDepth + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}

int main(int argc, char **argv) {

    std::string data_root = "/home/bobin/data/euroc/MH_02_easy/";
    std::string left_img_path_orig = data_root + "cam0/data/";
    std::string right_img_path_orig = data_root + "cam1/data/";
    std::string time_file = "/home/bobin/Documents/code/map/sgm/stereo/test/EuRoC_TimeStamps/MH02.txt";
    std::string config_file  = "/home/bobin/Documents/code/map/sgm/stereo/test/EuRoC.yaml";
    std::string output_colored_disparitymap_orig = data_root + "/disparity";
    system(("mkdir " + output_colored_disparitymap_orig).c_str());

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<string> vstrImageDepth;
    vector<double> vTimeStamp;
    LoadImages(left_img_path_orig, right_img_path_orig, output_colored_disparitymap_orig, time_file, vstrImageLeft, vstrImageRight, vstrImageDepth, vTimeStamp);


    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
       rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);



    int counter = vstrImageLeft.size();

    for (int number = 0; number < counter; number++) {

        std::string left_img_path = vstrImageLeft[number];
        std::string right_img_path = vstrImageRight[number];
        std::stringstream ss;
        ss << std::fixed << output_colored_disparitymap_orig << "/" << vTimeStamp[number] << ".png";
        std::string output_colored_disparitymap = vstrImageDepth[number];
        int levels_disparity = 200;

        unsigned char *img_left = NULL, *img_right = NULL;
        size_t width, height;
        loadImageRGB(left_img_path.c_str(), width, height,
                     img_left);/*loading the gray image as RGB is fine, [gray, gray, gray]*/
        loadImageRGB(right_img_path.c_str(), width, height, img_right);
        int size = width * height, size3 = width * height * 3;
        unsigned char *img_tmp = new unsigned char[size3];
        float *guidance_left = new float[size3];

        cv::Mat imLeft, imRight, imLeftRect, imRightRect;
        imLeft.create(height, width, CV_8UC3);
        imRight.create(height, width, CV_8UC3);
        imLeft.data = img_left;
        imRight.data = img_right;

        cv::remap(imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
        cv::imshow("left", imLeftRect);

        std::cout << "Begin to calculate the disparity map..." << std::endl;
        deeprob::cv::SGMStereo matcher((int) width, (int) height, levels_disparity, 3, 3);
        deeprob::cv::BoxFilter(img_left, img_tmp, 1, 1, width, height, 3);/*pre-smoothing on guidance_left*/
        for (int i = 0; i < size3; ++i) guidance_left[i] = img_tmp[i];

        matcher.setImage(imLeftRect.data, imRightRect.data, 3);
        matcher.setGuidance(guidance_left, NULL, 3);
        matcher.calDisparity(); /*calculate the disparity map*/

        cv::Mat disparity(height, width, CV_8UC3);
        std::cout << "Done! Sequence " <<  " number " << number << " is saved." << std::endl;
        showDataFalseColor(matcher._disparity[0], width, height, disparity.data, output_colored_disparitymap.c_str(),
                           true);/*save the disparity map in false color*/


        cv::imshow("disparity", disparity);
        cv::waitKey(10);
        delete[]img_tmp;
        delete[]guidance_left;
        delete[]img_left;
        delete[]img_right;

    }


    return 1;
}
