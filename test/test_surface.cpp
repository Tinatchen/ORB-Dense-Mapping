//
// Created by bobin on 17-12-13.
//
#include <MapPoint.h>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <chrono>

using namespace ORB_SLAM2;
using namespace std;
using namespace Eigen;

int main(int argc, char **argv){
    int point_num = 100;
    vector<Vector3d> points;
    points.resize(point_num);

    for (int i = 0; i < point_num; ++i) {
        double x,y,z;
        x =  std::rand() % 100;
        y = std::rand() % 100;
        z = 0.1 * (1 - 5 * x - 10 * y) + std::rand() % 100 / 100.0 ;
        points[i] = Vector3d(x, y, z)  / 10.0;
    }

    SurfacePieceWise surfacePieceWise(0);
    surfacePieceWise.FitPlane(true, points);
    std::cout << surfacePieceWise.GetNormal() << std::endl;
    std::cout << surfacePieceWise.GetCentroid() << std::endl;
}