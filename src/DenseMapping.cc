/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "DenseMapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"

#include <pcl/io/pcd_io.h>
#include <boost/make_shared.hpp>

Mapping::Mapping() {
    viewerThread = make_shared<thread>(bind(&Mapping::viewer, this));
}




void Mapping::shutdown() {

    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

PointCloudMapping::PointCloudMapping(double resolution_):
    Mapping(){
    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    globalMap = boost::make_shared<PointCloud>();
//    globalOctoMap = shared_ptr<octomap::ColorOcTree>(new octomap::ColorOcTree(resolution_));
//    viewerThread = make_shared<thread>(bind(&PointCloudMapping::viewer, this));

}

void Mapping::insertKeyFrame(KeyFrame *kf, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &segment) {
//    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back(kf);
    colorImgs.push_back(color.clone());
    if (!depth.empty()) {
        depthImgs.push_back(depth);
    }
    if (!segment.empty()) {
        segmentImgs.push_back(segment.clone());
    }

    keyFrameUpdated.notify_one();
}


pcl::PointCloud<PointT>::Ptr
PointCloudMapping::generatePointCloud(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, cv::Mat plane) {
    PointCloud::Ptr tmp(new PointCloud());
    bool useSegment = false;
    if (!plane.empty()) {
        plane.assignTo(plane, CV_8U);
        useSegment = true;
    }
    float threshold = 128;


    // point cloud is null ptr
//    cv::imshow("test", depth);
    for (int m = 0; m < depth.rows; m += 3) {
        for (int n = 0; n < depth.cols; n += 3) {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 100)
                continue;
            PointT p;
            p.z = d;
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy;
            p.b = color.ptr<uchar>(m)[n * 3];
            p.g = color.ptr<uchar>(m)[n * 3 + 1];
            p.r = color.ptr<uchar>(m)[n * 3 + 2];

            if (useSegment) {
                if (plane.at<uchar>(m, n) > threshold) {
                    p.r = 255;
                }
            }

            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    octomap::Pointcloud cloud_octo;
    for (auto p : cloud->points)
        cloud_octo.push_back(p.x, p.y, p.z);

//    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}

void PointCloudMapping::viewer() {
    pcl::visualization::CloudViewer viewer("viewer");
    while (1) {
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag) {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }

        // keyframe is updated
        size_t N = 0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyframes.size();
        }

        for (size_t i = lastKeyframeSize; i < N; i++) {
            PointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i], segmentImgs[i]);
            *globalMap += *p;
        }
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud(globalMap);
        voxel.filter(*tmp);
        globalMap->swap(*tmp);
        viewer.showCloud(globalMap);
//        cout << "show global map, size=" << globalMap->points.size() << endl;
        lastKeyframeSize = N;
    }

    saveMap("map.pcd");
}


void PointCloudMapping::saveMap(const std::string &filename) {
    if (globalMap->empty())
        return;
    pcl::io::savePCDFileBinary(filename, *globalMap);

}
