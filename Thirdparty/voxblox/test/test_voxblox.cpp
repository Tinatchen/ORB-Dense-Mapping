//
// Created by bobin on 17-12-2.
//


#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

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

namespace voxblox {
    enum ColorMode {
        kColor = 0,
        kHeight,
        kNormals,
        kGray,
        kLambert,
        kLambertColor
    };

    class VoxbloxMap {
    public:
        VoxbloxMap();

        void processPointCloudAndInsert(const pcl::PointCloud<pcl::PointXYZRGB> &pointcloud_pcl,
                                        Transformation T_G_C, const bool is_freespace_pointcloud = false);


    private:

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
        std::shared_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;

        // Transform queue, used only when use_tf_transforms is false.
//        AlignedDeque<geometry_msgs::TransformStamped> transform_queue_;
    };

    VoxbloxMap::VoxbloxMap()
            : verbose_(true),
              color_ptcloud_by_weight_(false),
              generate_esdf_(false),
              generate_occupancy_(false),
              publish_tsdf_info_(false),
              publish_slices_(false),
              output_mesh_as_pointcloud_(false),
              output_mesh_as_pcl_mesh_(false),
              world_frame_("world"),
              sensor_frame_(""),
              use_tf_transforms_(true),
            // 10 ms here:
              timestamp_tolerance_ns_(10000000),
              slice_level_(0.5),
              use_freespace_pointcloud_(false) {
        // Before subscribing, determine minimum time between messages.
        // 0 by default.
        double min_time_between_msgs_sec = 0.0;


        // Determine map parameters.
        TsdfMap::Config config;

        // Workaround for OS X on mac mini not having specializations for float
        // for some reason.
        double voxel_size = config.tsdf_voxel_size;
        int voxels_per_side = config.tsdf_voxels_per_side;

        config.tsdf_voxel_size = static_cast<FloatingPoint>(voxel_size);
        config.tsdf_voxels_per_side = voxels_per_side;
        tsdf_map_.reset(new TsdfMap(config));

        // Determine integrator parameters.
        TsdfIntegratorBase::Config integrator_config;
        integrator_config.voxel_carving_enabled = true;
        // Used to be * 4 according to Marius's experience, was changed to *2
        // This should be made bigger again if behind-surface weighting is improved.
        integrator_config.default_truncation_distance = config.tsdf_voxel_size * 2;

        double truncation_distance = integrator_config.default_truncation_distance;
        double max_weight = integrator_config.max_weight;

        integrator_config.default_truncation_distance =
                static_cast<float>(truncation_distance);
        integrator_config.max_weight = static_cast<float>(max_weight);

        std::string method("merged");
        if (method.compare("simple") == 0) {
            tsdf_integrator_.reset(new SimpleTsdfIntegrator(
                    integrator_config, tsdf_map_->getTsdfLayerPtr()));
        } else if (method.compare("merged") == 0) {
            integrator_config.enable_anti_grazing = false;
            tsdf_integrator_.reset(new MergedTsdfIntegrator(
                    integrator_config, tsdf_map_->getTsdfLayerPtr()));
        } else if (method.compare("merged_discard") == 0) {
            integrator_config.enable_anti_grazing = true;
            tsdf_integrator_.reset(new MergedTsdfIntegrator(
                    integrator_config, tsdf_map_->getTsdfLayerPtr()));
        } else if (method.compare("fast") == 0) {
            tsdf_integrator_.reset(new FastTsdfIntegrator(
                    integrator_config, tsdf_map_->getTsdfLayerPtr()));
        } else {
            tsdf_integrator_.reset(new SimpleTsdfIntegrator(
                    integrator_config, tsdf_map_->getTsdfLayerPtr()));
        }

        // ESDF settings.
        if (generate_esdf_) {
            EsdfMap::Config esdf_config;
            // TODO(helenol): add possibility for different ESDF map sizes.
            esdf_config.esdf_voxel_size = config.tsdf_voxel_size;
            esdf_config.esdf_voxels_per_side = config.tsdf_voxels_per_side;

            esdf_map_.reset(new EsdfMap(esdf_config));

            EsdfIntegrator::Config esdf_integrator_config;
            // Make sure that this is the same as the truncation distance OR SMALLER!
            esdf_integrator_config.min_distance_m =
                    integrator_config.default_truncation_distance;


            esdf_integrator_.reset(new EsdfIntegrator(esdf_integrator_config,
                                                      tsdf_map_->getTsdfLayerPtr(),
                                                      esdf_map_->getEsdfLayerPtr()));
        }

        // Occupancy settings.
        if (generate_occupancy_) {
            OccupancyMap::Config occupancy_config;
            // TODO(helenol): add possibility for different ESDF map sizes.
            occupancy_config.occupancy_voxel_size = config.tsdf_voxel_size;
            occupancy_config.occupancy_voxels_per_side = config.tsdf_voxels_per_side;
            occupancy_map_.reset(new OccupancyMap(occupancy_config));

            OccupancyIntegrator::Config occupancy_integrator_config;
            occupancy_integrator_.reset(new OccupancyIntegrator(
                    occupancy_integrator_config, occupancy_map_->getOccupancyLayerPtr()));
        }

        // Mesh settings.
        mesh_filename_ = std::string("mesh.ply");

        std::string color_mode("color");
//        nh_private_.param("color_mode", color_mode, color_mode);
        if (color_mode == "color" || color_mode == "colors") {
            color_mode_ = ColorMode::kColor;
        } else if (color_mode == "height") {
            color_mode_ = ColorMode::kHeight;
        } else if (color_mode == "normals") {
            color_mode_ = ColorMode::kNormals;
        } else if (color_mode == "lambert") {
            color_mode_ = ColorMode::kLambert;
        } else {  // Default case is gray.
            color_mode_ = ColorMode::kGray;
        }

        MeshIntegrator<TsdfVoxel>::Config mesh_config;

        mesh_layer_.reset(new MeshLayer(tsdf_map_->block_size()));

        mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
                mesh_config, tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get()));



        // If set, use a timer to progressively integrate the mesh.
        double update_mesh_every_n_sec = 0.0;


        if (update_mesh_every_n_sec > 0.0) {
//            update_mesh_timer_ =
//                    nh_private_.createTimer(ros::Duration(update_mesh_every_n_sec),
//                                            &VoxbloxMap::updateMeshEvent, this);
        }
        use_tf_transforms_ = false;
#if 0
        // If we use topic transforms, we have 2 parts: a dynamic transform from a
        // topic and a static transform from parameters.
        // Static transform should be T_G_D (where D is whatever sensor the
        // dynamic coordinate frame is in) and the static should be T_D_C (where
        // C is the sensor frame that produces the depth data). It is possible to
        // specific T_C_D and set invert_static_tranform to true.
        if (!use_tf_transforms_) {
            transform_sub_ =
                    nh_.subscribe("transform", 40, &VoxbloxMap::transformCallback, this);
            // Retrieve T_D_C from params.
            XmlRpc::XmlRpcValue T_B_D_xml;
            // TODO(helenol): split out into a function to avoid duplication.
            if (nh_private_.getParam("T_B_D", T_B_D_xml)) {
                kindr::minimal::xmlRpcToKindr(T_B_D_xml, &T_B_D_);

                // See if we need to invert it.
                bool invert_static_tranform = false;
                nh_private_.param("invert_T_B_D", invert_static_tranform,
                                  invert_static_tranform);
                if (invert_static_tranform) {
                    T_B_D_ = T_B_D_.inverse();
                }
            }
            XmlRpc::XmlRpcValue T_B_C_xml;
            if (nh_private_.getParam("T_B_C", T_B_C_xml)) {
                kindr::minimal::xmlRpcToKindr(T_B_C_xml, &T_B_C_);

                // See if we need to invert it.
                bool invert_static_tranform = false;
                nh_private_.param("invert_T_B_C", invert_static_tranform,
                                  invert_static_tranform);
                if (invert_static_tranform) {
                    T_B_C_ = T_B_C_.inverse();
                }
            }

            ROS_INFO_STREAM("Static transforms loaded from file.\nT_B_D:\n"
                                    << T_B_D_ << "\nT_B_C:" << T_B_C_);
        }
#endif
    }

    void VoxbloxMap::processPointCloudAndInsert(
            const pcl::PointCloud<pcl::PointXYZRGB> &pointcloud_pcl,
            Transformation T_G_C, const bool is_freespace_pointcloud) {

        timing::Timer ptcloud_timer("ptcloud_preprocess");
        Pointcloud points_C;
        Colors colors;
        points_C.reserve(pointcloud_pcl.size());
        colors.reserve(pointcloud_pcl.size());
        for (size_t i = 0; i < pointcloud_pcl.points.size(); ++i) {
            if (!std::isfinite(pointcloud_pcl.points[i].x) ||
                !std::isfinite(pointcloud_pcl.points[i].y) ||
                !std::isfinite(pointcloud_pcl.points[i].z)) {
                continue;
            }

            points_C.push_back(Point(pointcloud_pcl.points[i].x,
                                     pointcloud_pcl.points[i].y,
                                     pointcloud_pcl.points[i].z));
            colors.push_back(
                    Color(pointcloud_pcl.points[i].r, pointcloud_pcl.points[i].g,
                          pointcloud_pcl.points[i].b, pointcloud_pcl.points[i].a));
        }

        ptcloud_timer.Stop();

        if (verbose_) {
            std::cout << "Integrating a pointcloud with %lu points." << points_C.size();
        }

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        tsdf_integrator_->integratePointCloud(T_G_C, points_C, colors,
                                              is_freespace_pointcloud);

        if (generate_occupancy_ && !is_freespace_pointcloud) {
            occupancy_integrator_->integratePointCloud(T_G_C, points_C);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double timeCost = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
        if (verbose_) {
            std::cout << "Finished integrating in %f seconds, have %lu blocks." <<
                      timeCost << tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks();
            if (generate_occupancy_) {
                std::cout << "Occupancy: %lu blocks." <<
                          occupancy_map_->getOccupancyLayerPtr()->getNumberOfAllocatedBlocks();
            }
        }

    }


}  // namespace voxblox

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::InstallFailureSignalHandler();
    voxblox::VoxbloxMap node;
    return 0;
}
