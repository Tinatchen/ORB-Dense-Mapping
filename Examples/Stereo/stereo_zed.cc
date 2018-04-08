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
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>
#include "Viewer.h"
#include<System.h>
//#include "Eigen/"
#include "Converter.h"
#include <map>
using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

void LoadTrajectory(const string &strPathToTrajectory, vector<pair<double, cv::Mat> >& trajecory);


int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    vector<pair<double, cv::Mat> > trajectory;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);
    LoadTrajectory(argv[3], trajectory);

    vector<pair<double, cv::Mat> >::iterator traj_it = trajectory.begin();
    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true);


    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // Main loop
    cv::Mat imLeft, imRight;
    for(int ni=575; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_GRAYSCALE);

        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_GRAYSCALE);
        cv::flip(imLeft,imLeft,0);
        cv::flip(imRight,imRight,0);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
        while ((*traj_it).first < tframe){
            traj_it++;

        }

        cv::imshow("test", imLeft);
        cv::waitKey(10);
        // Pass the images to the SLAM system
//        SLAM.TrackStereo(imLeft,imRight,tframe);
//        vector<double> pose_vec = *traj_it;


        SLAM.setPose((*traj_it).second);
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
//    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
//    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "left/left_time.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/left/";
    string strPrefixRight = strPathToSequence + "/right/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".jpg";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".jpg";
    }
}


void LoadTrajectory(const string &strPathToTrajectory, vector<pair<double, cv::Mat> >& trajecory){
    ifstream fTimes;
    string strPathTimeFile = strPathToTrajectory + "/tra.txt";
    fTimes.open(strPathTimeFile.c_str());

    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        vector<double> pose_vec;
        pose_vec.resize(8);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            for (int i = 0; i < 8; ++i) {
                ss >> pose_vec[i];
            }
        }
        Eigen::Quaterniond quat(pose_vec[4], pose_vec[5], pose_vec[6], pose_vec[7]);

        Eigen::Vector3d pos(pose_vec[1], pose_vec[2], pose_vec[3]);
        Eigen::Matrix4d pose;
        pose.block<3, 3>(0, 0) = quat.toRotationMatrix();
        pose.block<3, 1>(0, 3) = pos;
        std::cout << pose << std::endl;
        for (int j = 0; j < 8; ++j) {
            std::cout << pose_vec[j];
        }
        std::cout << std::endl;

        trajecory.push_back(make_pair(pose_vec[0], ORB_SLAM2::Converter::toCvMat(pose)));
    }
}