//
// Created by bobin on 17-12-3.
//

#include "VoxbloxMapping.h"
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>
#include "Converter.h"
#include <opencv2/core/eigen.hpp>
#include <pcl/io/pcd_io.h>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

VoxbloxMap::VoxbloxMap()
        : Mapping(), verbose_(true), generate_esdf_(false), generate_occupancy_(false),
          use_freespace_pointcloud_(false) {

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
    integrator_config.max_ray_length_m = 1e9;
    double truncation_distance = integrator_config.default_truncation_distance;
    double max_weight = integrator_config.max_weight;

    integrator_config.default_truncation_distance = static_cast<float>(truncation_distance);
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
        esdf_integrator_config.min_distance_m = integrator_config.default_truncation_distance;

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


}


void VoxbloxMap::processPointCloudAndInsert(
        const pcl::PointCloud<pcl::PointXYZRGBA> &pointcloud_pcl,
        Transformation T_G_C, const bool is_freespace_pointcloud) {


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

    if (verbose_) {
        std::cout << "Integrating a pointcloud with " << points_C.size() << " points" << std::endl;
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
        std::cout << "Finished integrating in" << timeCost << " seconds, have "
                  << tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() << " blocks" << std::endl;
        if (generate_occupancy_) {
            std::cout << "Occupancy:" <<
                      occupancy_map_->getOccupancyLayerPtr()->getNumberOfAllocatedBlocks() << " blocks." << std::endl;
        }
    }

}

void VoxbloxMap::optimize() {

    if (keyframes.size() < 8)
        return;
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    int idVertex = 0;
    int idEdge = 0;
    for (int j = 0; j < keyframes.size(); ++j) {
        KeyFrame *kf = keyframes[j];
        g2o::VertexDepthScale *vertexDepthScale = new g2o::VertexDepthScale();
        vertexDepthScale->setEstimate(kf->mDepthScale.cast<double>());
        vertexDepthScale->setId(idVertex++);
        optimizer.addVertex(vertexDepthScale);
    }

    for (int k = 1; k < keyframes.size(); ++k) {
        g2o::EdgeDepthScaleScale *edgeDepthScaleScale = new g2o::EdgeDepthScaleScale();
        edgeDepthScaleScale->setVertex(0, optimizer.vertex(k-1));
        edgeDepthScaleScale->setVertex(1, optimizer.vertex(k));
        edgeDepthScaleScale->setInformation(Eigen::Matrix2d::Identity());
        edgeDepthScaleScale->setId(idEdge++);
        optimizer.addEdge(edgeDepthScaleScale);
    }

    for (int j = 0; j < keyframes.size(); ++j) {
        KeyFrame *kf = keyframes[j];
        for (int i = 0; i < kf->mvDepthVec.size(); ++i) {
            g2o::VertexDepthVector *vertexDepthVector = new g2o::VertexDepthVector();
            vertexDepthVector->setEstimate(kf->mvDepthVec[i].cast<double>());
            vertexDepthVector->setId(idVertex++);
            vertexDepthVector->setFixed(true);
            optimizer.addVertex(vertexDepthVector);

            g2o::EdgeDepthScalePoint *edgeDepthScalePoint = new g2o::EdgeDepthScalePoint();
            edgeDepthScalePoint->setVertex(0, optimizer.vertex(j));
            edgeDepthScalePoint->setVertex(1, vertexDepthVector);
            edgeDepthScalePoint->setInformation(Eigen::Matrix2d::Identity());
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            edgeDepthScalePoint->setRobustKernel(rk);
            rk->setDelta(thHuber2D);
            optimizer.addEdge(edgeDepthScalePoint);
        }
    }
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    for (int k = 0; k < keyframes.size(); ++k) {
        const g2o::VertexDepthScale *vertexDepthScale = static_cast<const g2o::VertexDepthScale *>(optimizer.vertex(
                keyframes[k]->mnId));
        keyframes[k]->mDepthScale = vertexDepthScale->estimate().cast<float>();
    }

}

pcl::PointCloud<PointT>::Ptr
VoxbloxMap::generatePointCloud(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, cv::Mat plane) {
    PointCloud::Ptr cloud(new PointCloud);

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

            cloud->points.push_back(p);
        }
    }
    cloud->is_dense = false;
    return cloud;
}

void VoxbloxMap::viewer() {

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
            optimize();
            PointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i], segmentImgs[i]);
            Eigen::Isometry3d Tf = ORB_SLAM2::Converter::toSE3Quat(keyframes[i]->GetPose()).inverse();
            Eigen::Quaterniond q(Tf.rotation());
            Rotation R(q.w(), q.x(), q.y(), q.z());
            Position pos(Tf.matrix().block<3, 1>(0, 3).cast<float>());
            Transformation T(R, pos);
            processPointCloudAndInsert(*p, T, use_freespace_pointcloud_);
            if (generate_esdf_)
                esdf_integrator_->updateFromTsdfLayer(true);
//          if (generate_occupancy_)
//              occupancy_integrator_->updateOccupancyVoxel(true,);
        }
        lastKeyframeSize = N;
    }

    saveMap(mesh_filename_);
}

void VoxbloxMap::saveMap(const std::string &filename) {
    constexpr bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = false;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
    voxblox::outputMeshLayerAsPly(filename, *mesh_layer_);
//    for (int i = 0; i < keyframes.size(); ++i) {
//        std::cout << keyframes[i]->mDepthScale.transpose() << std::endl;
//    }
}

