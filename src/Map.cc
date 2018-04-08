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

#include "Map.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include<mutex>
#include <Converter.h>

namespace ORB_SLAM2 {
    Surface::Surface() {

    }

    void Surface::AddSurfacePoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexSurface);
        mspMapPoints.insert(pMP);
    }


    ///
    bool Surface::FitPlaneFromMapPoints(KeyFrame* pKF) {
        size_t num_pts = mspMapPoints.size();
        if (num_pts < 10)
            return false;

        vector<Eigen::Vector3d> pts;
        pts.resize(num_pts);
        int idx = 0;
        vector<MapPoint*> vpMapPoints;
        for (auto pMP : mspMapPoints) {
            cv::Mat position = pMP->GetWorldPos();
            pts[idx++] = Eigen::Vector3d(position.at<float>(0, 0), position.at<float>(1, 0), position.at<float>(2, 0));
            vpMapPoints.push_back(pMP);
        }
        int plane_num = mspSurfacePieceWise.size();
        SurfacePieceWise *surfacePieceWise = new SurfacePieceWise(plane_num);
        if (surfacePieceWise->FitPlane(true, pts)) {
            surfacePieceWise->AddPiecePointByIndex(vpMapPoints);
            mspMapPoints.clear();
            cv::Mat Twc = pKF->GetPoseInverse();
            Eigen::Matrix3d Rcw = Converter::toMatrix3d(Twc.rowRange(0, 3).colRange(0, 3));
            Eigen::Vector3d Ocw = Converter::toVector3d(Twc.col(3));
            surfacePieceWise->SetAxis(Rcw, Ocw);
            surfacePieceWise->CalcPlaneWHT();

            if (plane_num == 0) {
                mspSurfacePieceWise.push_back(surfacePieceWise);
            } else {
                double min_dis = std::numeric_limits<double>::infinity();
                int idx = -1;
                for (int i = 0; i < mspSurfacePieceWise.size(); ++i) {

                    auto pPiece = mspSurfacePieceWise[i];
                    double distance = (pPiece->GetNormal() - surfacePieceWise->GetNormal()).norm();
                    if (min_dis > distance) {
                        min_dis = distance;
                        idx = i;
                    }
                }
                if (min_dis < 0.02) {
                    mspSurfacePieceWise[idx]->AddPiecePointByIndex(vpMapPoints, surfacePieceWise->GetInlier());
                    mspSurfacePieceWise[idx]->Update();
                } else {
                    mspSurfacePieceWise.push_back(surfacePieceWise);
                }
            }
        }
        return true;
    }

    SurfacePieceWise *Surface::GetMaxPieceWise(int &maxNum) {
        SurfacePieceWise *maxPiece;
        maxNum = 0;
        cout << "piece size " << mspSurfacePieceWise.size() << endl;
        if (mspSurfacePieceWise.size() > 0) {
            for (auto pPiece : mspSurfacePieceWise) {
                if (pPiece->mNumPoints > maxNum) {
                    maxNum = pPiece->mNumPoints;
                    maxPiece = pPiece;
                }
            }
        }
        return maxPiece;
    }

    vector<SurfacePieceWise *> Surface::GetAllPiecewisePlane() {
        unique_lock<mutex> lock(mMutexSurface);
        return vector<SurfacePieceWise *>(mspSurfacePieceWise.begin(), mspSurfacePieceWise.end());
    }


    Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0), mbUsePlane(false) {
    }

    void Map::AddKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if (pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    void Map::AddMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::EraseMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::EraseKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    void Map::InformNewBigChange() {
        unique_lock<mutex> lock(mMutexMap);
        mnBigChangeIdx++;
    }

    int Map::GetLastBigChangeIdx() {
        unique_lock<mutex> lock(mMutexMap);
        return mnBigChangeIdx;
    }

    vector<KeyFrame *> Map::GetAllKeyFrames() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    vector<MapPoint *> Map::GetAllMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::MapPointsInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    vector<MapPoint *> Map::GetReferenceMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    long unsigned int Map::GetMaxKFid() {
        unique_lock<mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    void Map::clear() {
        for (set<MapPoint *>::iterator sit = mspMapPoints.begin(), send = mspMapPoints.end(); sit != send; sit++)
            delete *sit;

        for (set<KeyFrame *>::iterator sit = mspKeyFrames.begin(), send = mspKeyFrames.end(); sit != send; sit++)
            delete *sit;

        mspMapPoints.clear();
        mspKeyFrames.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpKeyFrameOrigins.clear();
    }

} //namespace ORB_SLAM
