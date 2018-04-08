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

#include "MapPoint.h"
#include "ORBmatcher.h"
#include <Converter.h>
#include<mutex>
#include <limits>

namespace ORB_SLAM2 {

    long unsigned int MapPoint::nNextId = 0;
    mutex MapPoint::mGlobalMutex;

    SurfacePieceWise::SurfacePieceWise(int nId) {
        mnId = nId;
        mNumPoints = 0;
        mInlierRate = 0.0;
        mOrigin = Eigen::Vector3d(0, 0, 0);
        mRwp = Eigen::Matrix3d::Identity();
        mNormal = Eigen::Vector3d(0, 0, 1);
        mCentoid = Eigen::Vector3d(0, 0, 0);
        UpdateParams();

    }

    SurfacePieceWise::SurfacePieceWise(int nId, const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin,
                                       const Eigen::Vector4d &params) {
        mnId = nId;
        mNumPoints = 0;
        mInlierRate = 0.0;
        mNormal = params.head<3>();
        mParams = params;
        SetAxis(Rwc, origin, params);
    }

    void SurfacePieceWise::UpdateParams() {
        mParams.head<3>() = mNormal;
        mParams(3) = -mNormal.dot(mCentoid);
    }

    void SurfacePieceWise::AddPiecePoint(const vector<MapPoint*>&vpPoints){
        mspPointSet.resize(vpPoints.size());
        for (int i = 0; i < vpPoints.size(); ++i) {
            mspPointSet[i] = vpPoints[i];
        }
    }

    vector<MapPoint *> SurfacePieceWise::GetAllPiecePoint() {
        return mspPointSet;
    }

    void SurfacePieceWise::AddPiecePointByIndex(const vector<MapPoint *> &vpPoints) {
        mspPointSet.resize(mvInPieceIdx.size());
        for (int i = 0; i < mvInPieceIdx.size(); ++i) {
            int idx = mvInPieceIdx[i];
            if (idx > vpPoints.size())
                continue;
            mspPointSet[i] = vpPoints[idx];
        }
    }

    void SurfacePieceWise::AddPiecePointByIndex(const vector<MapPoint *> &vpPoints, const vector<int> inlier) {
        if (inlier.size() <=0)
            return;
        int num = mspPointSet.size();
        assert(inlier.size() <= vpPoints.size());
        mspPointSet.resize(num + inlier.size());
        for (int i = 0; i < inlier.size(); ++i) {
            mspPointSet[num + i] = vpPoints[i];
        }
    }

    void SurfacePieceWise::Update() {
        int num_pts = mspPointSet.size();
        vector<Eigen::Vector3d> pts;
        pts.resize(num_pts);
        int idx = 0;
        vector<MapPoint*> vpMapPoints;
        for (auto pMP : mspPointSet) {
            cv::Mat position = pMP->GetWorldPos();
            pts[idx++] = Eigen::Vector3d(position.at<float>(0, 0), position.at<float>(1, 0), position.at<float>(2, 0));
            vpMapPoints.push_back(pMP);
        }
        if(FitPlane(true, pts)){
            CalcPlaneWHT();
        }
    }

    void SurfacePieceWise::CalcPlaneWHT() {
        int num = mspPointSet.size();
        if (num <=0)
            return;
        vector<double> ThickVec;
        vector<double> HeightVec;
        vector<double> WidthVec;
        ThickVec.resize(num);
        HeightVec.resize(num);
        WidthVec.resize(num);
        for (int i = 0; i < num; ++i) {
            MapPoint *pMp = mspPointSet[i];
            Eigen::Vector3d pos = Converter::toVector3d(pMp->GetWorldPos());
            ThickVec[i] = mNormal.dot(pos-mCentoid);
            Eigen::Vector3d YAxis = mRwp.col(1);
            Eigen::Vector3d XAxis = mRwp.col(0);
            HeightVec[i] = YAxis.dot(pos-mCentoid);
            WidthVec[i] = XAxis.dot(pos-mCentoid);
        }

        sort(ThickVec.begin(), ThickVec.end());
        int start = 0.1 * num;
        int end = 0.9 * num;
        mPlaneThick = ThickVec[end] - ThickVec[start];
        mPlaneHeight = HeightVec[end] - HeightVec[start];
        mPlaneWidth = WidthVec[end] - WidthVec[start];
    }

    double SurfacePieceWise::GetPieceHeight() {
        return mPlaneHeight;
    }

    double SurfacePieceWise::GetInlierRate() {
        return mInlierRate;
    }

    double SurfacePieceWise::GetPieceWidth() {
        return mPlaneWidth;
    }

    double SurfacePieceWise::GetPieceThick() {
        return mPlaneThick;
    }

    Eigen::Matrix3d SurfacePieceWise::GetHomoMatrix() {
        return mHomoMatrix;
    }

    Eigen::Vector3d SurfacePieceWise::GetCentroid() {
        return mCentoid;
    }

    Eigen::Vector3d SurfacePieceWise::GetNormal() {
        return mNormal;
    }

    Eigen::Matrix3d SurfacePieceWise::GetPose() {
        return mRwp;
    }

    Eigen::Vector3d SurfacePieceWise::GetXAxis() {
        return mRwp.col(0);
    }

    Eigen::Vector3d SurfacePieceWise::GetYAxis() {
        return mRwp.col(1);
    }

    Eigen::Vector4d SurfacePieceWise::GetParams() {
        UpdateParams();
        return mParams;
    }

    void
    SurfacePieceWise::best_plane_from_points(PointSetType &coord, Eigen::Vector3d &normvec, Eigen::Vector3d &centroid) {

        // calculate centroid
        Eigen::Vector3d centr(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

        // subtract centr
        coord.row(0).array() -= centr(0);
        coord.row(1).array() -= centr(1);
        coord.row(2).array() -= centr(2);

        // we only need the left-singular matrix here
        //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
        auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Vector3d plane_normal = svd.matrixU().rightCols<1>();
        normvec = plane_normal;
        centroid = centr;
    }

    bool SurfacePieceWise::best_plane_ransac(const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &normvec,
                                             Eigen::Vector3d &centroid, std::vector<int> &inliers, int nIter, float th,
                                             float inlier_rate) {
        Eigen::Vector3d best_norm;
        Eigen::Vector3d best_centoid;
        double best_err = std::numeric_limits<double>::infinity();
        vector<int> best_inlier_idxs;
        int iter_num = 10;
        int in_num = 20;

        int num = points.size();
        for (int i = 0; i < nIter; ++i) {
            vector<int> rand_idxs;
            random_partition(num, rand_idxs);
            PointSetType maybe(3, iter_num);
            PointSetType test(3, num);
            Eigen::Vector3d maybe_norm;
            Eigen::Vector3d maybe_centoid;
            ModelErrorType test_error(1, num);
            vector<int> inlier_idxs;
            std::cout << i << std::endl;
            for (int j = 0; j < iter_num; ++j) {
                maybe.col(j) = points[rand_idxs[j]];
            }

            for (int k = 0; k < num; ++k) {
                test.col(k) = points[rand_idxs[k]];
            }

            best_plane_from_points(maybe, maybe_norm, maybe_centoid);
            FitPlaneError(test, maybe_norm, maybe_centoid, test_error);

            for (int l = 0; l < num; ++l) {
                if (test_error(l) < th) {
                    inlier_idxs.push_back(rand_idxs[l]);
                }
            }
            if (inlier_idxs.size() > in_num) {
                Eigen::Vector3d better_norm;
                Eigen::Vector3d better_centoid;
                ModelErrorType better_err;
                int better_num = inlier_idxs.size();
                PointSetType better(3, better_num);
                for (int j = 0; j < better_num; ++j) {
                    better.col(j) = points[inlier_idxs[j]];
                }
                best_plane_from_points(better, better_norm, better_centoid);
                FitPlaneError(better, better_norm, better_centoid, better_err);
                double better_mean = better_err.sum() / better_err.cols();
                if (better_mean < best_err) {
                    best_err = better_mean;
                    best_centoid = better_centoid;
                    best_norm = better_norm;

                    FitPlaneError(test, maybe_norm, maybe_centoid, test_error);
                    best_inlier_idxs.clear();
                    for (int l = 0; l < num; ++l) {
                        if (test_error(l) < th) {
                            best_inlier_idxs.push_back(rand_idxs[l]);
                        }
                    }
                    float rate_in = 1.0 * best_inlier_idxs.size() / num;
                    cout << "inlier rate" << rate_in << endl;
                    if (rate_in > inlier_rate)
                        break;
                }
            }


        }
        if (best_err < th) {
            normvec = best_norm;
            centroid = best_centoid;
            for (auto idx : best_inlier_idxs)
                inliers.push_back(idx);
            return true;
        }
        return false;
    }

    vector<int> SurfacePieceWise::GetInlier() {
        return vector<int >(mvInPieceIdx.begin(), mvInPieceIdx.end());
    }

    bool SurfacePieceWise::FitPlane(bool UseRansac, const std::vector<Eigen::Vector3d> &points) {
        bool bOk = false;
        vector<int> inliers;
        double th = 0.07;
        float inlier_rate = 0.6;
        if (UseRansac) {
            bOk = best_plane_ransac(points, mNormal, mCentoid, inliers, 5, th, inlier_rate);
        } else {
            int num = points.size();

            PointSetType coord(3, points.size());
            for (int i = 0; i < points.size(); ++i) {
                coord.col(i) << points[i][0], points[i][1], points[i][2];
            }
            ModelErrorType test_error;
            best_plane_from_points(coord, mNormal, mCentoid);
            FitPlaneError(coord, mNormal, mCentoid, test_error);

            for (int l = 0; l < num; ++l) {
                if (test_error(l) < th) {
                    inliers.push_back(l);
                }
            }
            float rate_in = 1.0 * inliers.size() / num;
            cout << "inlier rate" << rate_in << endl;
            if (rate_in > inlier_rate)
                bOk = true;


        }

        if (bOk == true) {
            mNumPoints = inliers.size();
            mvInPieceIdx.resize(mNumPoints);
            for (int i = 0; i < mNumPoints; ++i) {
                mvInPieceIdx[i] = inliers[i];
            }
            UpdateParams();
            mInlierRate = inliers.size() / points.size();

        }
        return bOk;
    }

    void SurfacePieceWise::SetAxis(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin,
                                   const Eigen::Vector3d &normal, const Eigen::Vector3d &centroid) {
        Eigen::Vector4d params;
        params.head<3>() = normal;
        params(3) = -normal.dot(centroid);
        mCentoid = centroid;
        SetAxis(Rwc, origin, params);
    }

    void SurfacePieceWise::SetAxis(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin){

        double d = mParams(3);
        Eigen::Vector3d ZAxis;
        Eigen::Vector3d XAxis;
        if (mNormal[2] < 0)
            mNormal *= -1;
        ZAxis = mNormal;
        XAxis = Rwc.col(2);
        XAxis -= XAxis.dot(ZAxis) * ZAxis;
        XAxis.normalize();
        Eigen::Vector3d YAxis;
        YAxis = -ZAxis.cross(XAxis);
        mRwp.col(0).array() = XAxis;
        mRwp.col(1).array() = YAxis;
        mRwp.col(2).array() = ZAxis;

        Eigen::Matrix3d Rcw = Rwc;
        Eigen::Vector3d tcw = -Rcw * origin;
        mHomoMatrix = Rcw - tcw * mNormal.transpose() / d;
        mOrigin = origin;
    }

    void SurfacePieceWise::SetAxis(const Eigen::Matrix3d &Rwc, const Eigen::Vector3d &origin,
                                   const Eigen::Vector4d &params) {
        Eigen::Vector3d normal = params.head<3>();
        SetAxis(Rwc, origin);
        mNormal = normal;
        mParams  = params;
    }

    void SurfacePieceWise::SetOrigin(const Eigen::Vector3d& origin) {
        mOrigin = origin;
    }


    void SurfacePieceWise::FitPlaneError(const PointSetType &data, const Eigen::Vector3d &normal,
                                         const Eigen::Vector3d &centoid, ModelErrorType &error) {
        PointSetType pts = data;
        pts.row(0).array() -= centoid(0);
        pts.row(1).array() -= centoid(1);
        pts.row(2).array() -= centoid(2);
        error = normal.transpose() * pts;
    }

    void SurfacePieceWise::random_partition(int num, vector<int> &all_idxs) {
        all_idxs.resize(num);
        for (int i = 1; i < num; ++i)
            all_idxs[i] = i;

        std::random_shuffle(all_idxs.begin(), all_idxs.end());
    }



    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap) :
            mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
            mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
            mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap), mbInSurface(
            false) {
        Pos.copyTo(mWorldPos);
        mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    MapPoint::MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF) :
            mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
            mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame *>(NULL)), mnVisible(1),
            mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), mbInSurface(false) {
        Pos.copyTo(mWorldPos);
        cv::Mat Ow = pFrame->GetCameraCenter();
        mNormalVector = mWorldPos - Ow;
        mNormalVector = mNormalVector / cv::norm(mNormalVector);

        cv::Mat PC = Pos - Ow;
        const float dist = cv::norm(PC);
        const int level = pFrame->mvKeysUn[idxF].octave;
        const float levelScaleFactor = pFrame->mvScaleFactors[level];
        const int nLevels = pFrame->mnScaleLevels;

        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

        pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    void MapPoint::SetWorldPos(const cv::Mat &Pos) {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);
    }

    cv::Mat MapPoint::GetWorldPos() {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    cv::Mat MapPoint::GetNormal() {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    KeyFrame *MapPoint::GetReferenceKeyFrame() {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapPoint::AddObservation(KeyFrame *pKF, size_t idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;

        if (pKF->mvuRight[idx] >= 0)
            nObs += 2;
        else
            nObs++;
    }

    void MapPoint::EraseObservation(KeyFrame *pKF) {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF)) {
                int idx = mObservations[pKF];
                if (pKF->mvuRight[idx] >= 0)
                    nObs -= 2;
                else
                    nObs--;

                mObservations.erase(pKF);

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                // If only 2 observations or less, discard point
                if (nObs <= 2)
                    bBad = true;
            }
        }

        if (bBad)
            SetBadFlag();
    }

    map<KeyFrame *, size_t> MapPoint::GetObservations() {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapPoint::Observations() {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    void MapPoint::SetBadFlag() {
        map<KeyFrame *, size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
        }
        for (map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) {
            KeyFrame *pKF = mit->first;
            pKF->EraseMapPointMatch(mit->second);
        }

        mpMap->EraseMapPoint(this);
    }

    MapPoint *MapPoint::GetReplaced() {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapPoint::Replace(MapPoint *pMP) {
        if (pMP->mnId == this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame *, size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs = mObservations;
            mObservations.clear();
            mbBad = true;
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMP;
        }

        for (map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++) {
            // Replace measurement in keyframe
            KeyFrame *pKF = mit->first;

            if (!pMP->IsInKeyFrame(pKF)) {
                pKF->ReplaceMapPointMatch(mit->second, pMP);
                pMP->AddObservation(pKF, mit->second);
            } else {
                pKF->EraseMapPointMatch(mit->second);
            }
        }
        pMP->IncreaseFound(nfound);
        pMP->IncreaseVisible(nvisible);
        pMP->ComputeDistinctiveDescriptors();

        mpMap->EraseMapPoint(this);
    }

    bool MapPoint::isBad() {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void MapPoint::IncreaseVisible(int n) {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapPoint::IncreaseFound(int n) {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapPoint::GetFoundRatio() {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    void MapPoint::ComputeDistinctiveDescriptors() {
        // Retrieve all observed descriptors
        vector<cv::Mat> vDescriptors;

        map<KeyFrame *, size_t> observations;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if (mbBad)
                return;
            observations = mObservations;
        }

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            KeyFrame *pKF = mit->first;

            if (!pKF->isBad())
                vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for (size_t i = 0; i < N; i++) {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++) {
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++) {
            vector<int> vDists(Distances[i], Distances[i] + N);
            sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian) {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);
            mDescriptor = vDescriptors[BestIdx].clone();
        }
    }

    cv::Mat MapPoint::GetDescriptor() {
        unique_lock<mutex> lock(mMutexFeatures);
        return mDescriptor.clone();
    }

    int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    bool MapPoint::IsInKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    void MapPoint::UpdateNormalAndDepth() {
        map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        cv::Mat Pos;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if (mbBad)
                return;
            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos.clone();
        }

        if (observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int n = 0;
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end();
             mit != mend; mit++) {
            KeyFrame *pKF = mit->first;
            cv::Mat Owi = pKF->GetCameraCenter();
            cv::Mat normali = mWorldPos - Owi;
            normal = normal + normali / cv::norm(normali);
            n++;
        }

        cv::Mat PC = Pos - pRefKF->GetCameraCenter();
        const float dist = cv::norm(PC);
        const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels;

        {
            unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance = dist * levelScaleFactor;
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
            mNormalVector = normal / n;
        }
    }

    float MapPoint::GetMinDistanceInvariance() {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance() {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF) {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pKF->mnScaleLevels)
            nScale = pKF->mnScaleLevels - 1;

        return nScale;
    }

    int MapPoint::PredictScale(const float &currentDist, Frame *pF) {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }


} //namespace ORB_SLAM
