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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"

#include<opencv2/core/core.hpp>
#include<mutex>
#include "Eigen/Core"

namespace ORB_SLAM2 {

    class KeyFrame;

    class Map;

    class Frame;

    typedef Eigen::Matrix<double, 3, Eigen::Dynamic> PointSetType;
    typedef Eigen::Matrix<double, 1, Eigen::Dynamic> ModelErrorType;


    class MapPoint {
    public:
        MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);

        MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);

        void SetWorldPos(const cv::Mat &Pos);

        cv::Mat GetWorldPos();

        cv::Mat GetNormal();

        KeyFrame *GetReferenceKeyFrame();

        std::map<KeyFrame *, size_t> GetObservations();

        int Observations();

        void AddObservation(KeyFrame *pKF, size_t idx);

        void EraseObservation(KeyFrame *pKF);

        int GetIndexInKeyFrame(KeyFrame *pKF);

        bool IsInKeyFrame(KeyFrame *pKF);

        void SetBadFlag();

        bool isBad();

        void Replace(MapPoint *pMP);

        MapPoint *GetReplaced();

        void IncreaseVisible(int n = 1);

        void IncreaseFound(int n = 1);

        float GetFoundRatio();

        inline int GetFound() {
            return mnFound;
        }

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateNormalAndDepth();

        float GetMinDistanceInvariance();

        float GetMaxDistanceInvariance();

        int PredictScale(const float &currentDist, KeyFrame *pKF);

        int PredictScale(const float &currentDist, Frame *pF);

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        // Variables used by the tracking
        float mTrackProjX;
        float mTrackProjY;
        float mTrackProjXR;
        bool mbTrackInView;
        int mnTrackScaleLevel;
        float mTrackViewCos;

        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnLastFrameSeen;

        // Variables used by local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopPointForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;
        long unsigned int mnBAGlobalForKF;


        static std::mutex mGlobalMutex;

        bool mbInSurface;

    protected:

        // Position in absolute coordinates
        cv::Mat mWorldPos;

        // Keyframes observing the point and associated index in keyframe
        std::map<KeyFrame *, size_t> mObservations;

        // Mean viewing direction
        cv::Mat mNormalVector;

        // Best descriptor to fast matching
        cv::Mat mDescriptor;

        // Reference KeyFrame
        KeyFrame *mpRefKF;

        // Tracking counters
        int mnVisible;
        int mnFound;

        // Bad flag (we do not currently erase MapPoint from memory)
        bool mbBad;
        MapPoint *mpReplaced;

        // Scale invariance distances
        float mfMinDistance;
        float mfMaxDistance;

        Map *mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
    };

    class SurfacePieceWise {
    public:

        SurfacePieceWise(int nId);

        SurfacePieceWise(int nId, const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin,
                         const Eigen::Vector4d &params);

        int mnId;

        int mNumPoints;

        vector<int> mvInPieceIdx;

        Eigen::Vector3d GetNormal();

        Eigen::Vector3d GetXAxis();

        Eigen::Vector3d GetYAxis();

        Eigen::Matrix3d GetPose();

        Eigen::Vector4d GetParams();

        Eigen::Vector3d GetCentroid();

        double GetPieceThick();

        double GetPieceHeight();

        double GetInlierRate();

        double GetPieceWidth();

        bool FitPlane(bool UseRansac, const std::vector<Eigen::Vector3d> &points);

        vector<int> GetInlier();

        void SetAxis(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin, const Eigen::Vector4d &params);

        void SetAxis(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin);

        void SetAxis(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin, const Eigen::Vector3d &normal,
                     const Eigen::Vector3d &centroid);

        void SetOrigin(const Eigen::Vector3d &origin);

        void AddPiecePoint(const vector<MapPoint *> &vpPoints);
        vector<MapPoint *> GetAllPiecePoint();

        void AddPiecePointByIndex(const vector<MapPoint *> &vpPoints);
        void AddPiecePointByIndex(const vector<MapPoint *> &vpPoints,  const vector<int> inlier);


        void Update();
        void CalcPlaneWHT();

        static bool lnum(SurfacePieceWise* pP1, SurfacePieceWise* pP2){
            return pP1->mNumPoints > pP2->mNumPoints;
        }

    private:
        void best_plane_from_points(PointSetType &coord, Eigen::Vector3d &normvec, Eigen::Vector3d &centroid);

        bool best_plane_ransac(const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &normvec,
                               Eigen::Vector3d &centroid, std::vector<int> &inlier, int nIter = 5, float th = 0.05,
                               float inlier_rate = 0.6);

        void random_partition(int num, vector<int> &all_idxs);

        void FitPlaneRansac(Eigen::Vector4f &params, const std::vector<Eigen::Vector3f> &coord);

        void FitPlaneError(const PointSetType &data, const Eigen::Vector3d &normal, const Eigen::Vector3d &centoid,
                           ModelErrorType &error);

        void UpdateParams();

        Eigen::Matrix3d GetHomoMatrix();

        Eigen::Vector4d mParams;

        Eigen::Vector3d mNormal;
        Eigen::Vector3d mCentoid;

        Eigen::Matrix3d mHomoMatrix;
        Eigen::Vector3d mOrigin;
        Eigen::Matrix3d mRwp;

        double mPlaneThick;
        double mPlaneHeight;
        double mPlaneWidth;

        vector<MapPoint *> mspPointSet;
        double  mInlierRate;
    };


} //namespace ORB_SLAM

#endif // MAPPOINT_H
