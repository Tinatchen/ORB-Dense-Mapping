//
// Created by bobin on 17-12-19.
//


/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

void LoadImages(const string &strPathToSequence, const string &strPathToSeg, vector<string> &vstrImageFilenames,
                vector<string> &vstrImageSegNames);

int main(int argc, char **argv) {
    if (argc != 5) {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<string> vstrImageSegFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), string(argv[4]), vstrImageFilenames, vstrImageSegFilenames);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im, ims;
    for (int ni = 0; ni < nImages; ni++) {

        // Read image from file
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_GRAYSCALE);
        ims = cv::imread(vstrImageSegFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);

        double tframe = ni / 10.0;

        if (im.empty()) {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

//        cv::imshow("seg", ims);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe, ims);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
//        double T = 0;
//        if (ni < nImages - 1)
//            T = vTimestamps[ni + 1] - tframe;
//        else if (ni > 0)
//            T = tframe - vTimestamps[ni - 1];
//
//        if (ttrack < T)
//            usleep((T - ttrack) * 1e6);
    }

//    for (int ni = 0; ni < nImages / 4; ni++) {
//
//        // Read image from file
//        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_GRAYSCALE);
//        ims = cv::imread(vstrImageSegFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
//
//        double tframe = (ni + nImages) / 10.0;
//
//        if (im.empty()) {
//            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
//            return 1;
//        }
//
////        cv::imshow("seg", ims);
//
//#ifdef COMPILEDWITHC11
//        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
//#else
//        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
//#endif
//
//        // Pass the image to the SLAM system
//        SLAM.TrackMonocular(im, tframe, ims);
//
//#ifdef COMPILEDWITHC11
//        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//#else
//        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
//#endif
//
//        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//
//        vTimesTrack[ni] = ttrack;
//
//        // Wait to load the next frame
////        double T = 0;
////        if (ni < nImages - 1)
////            T = vTimestamps[ni + 1] - tframe;
////        else if (ni > 0)
////            T = tframe - vTimestamps[ni - 1];
////
////        if (ttrack < T)
////            usleep((T - ttrack) * 1e6);
//    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}


void LoadImages(const string &strPathToSequence, const string &strPathToSeg, vector<string> &vstrImageFilenames,
                vector<string> &vstrImageSegNames) {
    string strPrefixLeft = strPathToSequence;

    const int nTimes = 2107;
    vstrImageFilenames.resize(nTimes);
    vstrImageSegNames.resize(nTimes);

    for (int i = 0; i < nTimes; i++) {
        stringstream ss,ss1;
//        ss << setfill('0') << setw(6) << i;
        ss <<  i+4177;
        ss1 << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".jpg";
        vstrImageSegNames[i] = strPathToSeg + ss1.str() + ".png";
    }
}
