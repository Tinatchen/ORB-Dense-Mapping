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

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include "string"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

using namespace ORB_SLAM2;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class Mapping {
public:
    Mapping();

    void insertKeyFrame(KeyFrame *kf, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &segment = cv::Mat());
    virtual void saveMap(const std::string &filename)=0;
    void shutdown();

    virtual void viewer()=0;
    virtual void optimize()=0;

    shared_ptr<thread> viewerThread;

    bool shutDownFlag = false;
    mutex shutDownMutex;

    condition_variable keyFrameUpdated;
    mutex keyFrameUpdateMutex;

    // data to generate point clouds
    vector<KeyFrame *> keyframes;
    vector<cv::Mat> colorImgs;
    vector<cv::Mat> depthImgs;
    vector<cv::Mat> segmentImgs;
    vector<Eigen::Vector2f> frame_scales;
    mutex keyframeMutex;
    uint16_t lastKeyframeSize = 0;
};

class PointCloudMapping : public Mapping {
public:


    PointCloudMapping(double resolution_);
    virtual void saveMap(const std::string &filename) override ;
    virtual void optimize() override {}

    virtual void viewer() override ;

protected:

    PointCloud::Ptr generatePointCloud(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, cv::Mat plane = cv::Mat());

    PointCloud::Ptr globalMap;
    double resolution = 0.04;
    pcl::VoxelGrid<PointT> voxel;
};

#endif // POINTCLOUDMAPPING_H
