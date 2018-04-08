//
// Created by bobin on 17-12-3.
//

#ifndef ORB_DENSE_SLAM2_VOXBLOXMAPPING_H
#define ORB_DENSE_SLAM2_VOXBLOXMAPPING_H

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
//#include <pcl_conversions/pcl_conversions.h>

#include <voxblox/core/esdf_map.h>
#include <voxblox/core/occupancy_map.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox/integrator/occupancy_integrator.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <chrono>
#include "KeyFrame.h"
#include "DenseMapping.h"
using namespace ORB_SLAM2;
using namespace voxblox;

enum ColorMode {
    kColor = 0,
    kHeight,
    kNormals,
    kGray,
    kLambert,
    kLambertColor
};

class VoxbloxMap : public Mapping{
public:
    VoxbloxMap();

    //virtual void insertKeyFrame(KeyFrame *kf, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &segment = cv::Mat()) override ;

    virtual void saveMap(const std::string &filename) override ;

//    void shutdown();

    virtual void viewer() override ;
    virtual void optimize() override ;

private:

    void processPointCloudAndInsert(const pcl::PointCloud<pcl::PointXYZRGBA> &pointcloud_pcl,
                                    Transformation T_G_C, const bool is_freespace_pointcloud = false);

    PointCloud::Ptr generatePointCloud(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, cv::Mat plane = cv::Mat());

    bool verbose_;

    // This is a debug option, more or less...
    bool color_ptcloud_by_weight_;

    // Which maps to generate.
    bool generate_esdf_;
    bool generate_occupancy_;

    // What output information to publish
    bool publish_tsdf_info_;
    bool publish_slices_;

    bool output_mesh_as_pointcloud_;
    bool output_mesh_as_pcl_mesh_;

    // Global/map coordinate frame. Will always look up TF transforms to this
    // frame.
    std::string world_frame_;
    // If set, overwrite sensor frame with this value. If empty, unused.
    std::string sensor_frame_;
    // Whether to use TF transform resolution (true) or fixed transforms from
    // parameters and transform topics (false).
    bool use_tf_transforms_;
    int64_t timestamp_tolerance_ns_;
    // B is the body frame of the robot, C is the camera/sensor frame creating
    // the pointclouds, and D is the 'dynamic' frame; i.e., incoming messages
    // are assumed to be T_G_D.
    Transformation T_B_C_;
    Transformation T_B_D_;

    // Pointcloud visualization settings.
    double slice_level_;

    // If the system should subscribe to a pointcloud giving points in freespace
    bool use_freespace_pointcloud_;

    // Mesh output settings. Mesh is only written to file if mesh_filename_ is
    // not empty.
    std::string mesh_filename_;
    // How to color the mesh.
    ColorMode color_mode_;

    // Keep track of these for throttling.
    double min_time_between_msgs_;

    // To be replaced (at least optionally) with odometry + static transform
    // from IMU to visual frame.

    // Maps and integrators.
    std::shared_ptr<TsdfMap> tsdf_map_;
    std::shared_ptr<TsdfIntegratorBase> tsdf_integrator_;
    // ESDF maps (optional).
    std::shared_ptr<EsdfMap> esdf_map_;
    std::shared_ptr<EsdfIntegrator> esdf_integrator_;
    // Occupancy maps (optional).
    std::shared_ptr<OccupancyMap> occupancy_map_;
    std::shared_ptr<OccupancyIntegrator> occupancy_integrator_;
    // Mesh accessories.
    std::shared_ptr<MeshLayer> mesh_layer_;
    std::shared_ptr<MeshIntegrator < TsdfVoxel>> mesh_integrator_;

    // Transform queue, used only when use_tf_transforms is false.
//        AlignedDeque<geometry_msgs::TransformStamped> transform_queue_;
};

#endif //ORB_DENSE_SLAM2_VOXBLOXMAPPING_H
