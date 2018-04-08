/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file sgm_stereo.h
 * @brief SGM based stereo for autonomous driving platform
 *        recursive filtering based post-refinement
 * 1, this implementation is not intended for realtime applications
 * 2, the algorithm performance is tuned on KITTI 2012 data sets
 */

#ifndef DEEPROB_CV_SGM_STEREO_H_
#define DEEPROB_CV_SGM_STEREO_H_

#include <nmmintrin.h>
#include <vector>
#include "utility.h"
#include "ctmf.h"
#include "geodesic_filtering.h"

#define NUM_OF_DISPARITY_LEVEL        256 /*should be multiples of 8 in this implementation, see SGM code*/
#define RADIUS_CENSUS                 4
#define THRES_CENSUS                  20
#define RADIUS_BOXFILTER              1
#define THRES_ERR_CROSS_CHECK         2
#define DELTA_DISPARITY_RATIO         (0.1f)
#define DELTA_DISPARITY_MAXIMUM       (7)
#define RADIUS_MEDIAN                 3
#define SIGMA_S                       8.f /*spatial*/
#define SIGMA_R                       4.f /*range*/

typedef unsigned short COSTTYPE_;              /*only support unsigned short*/
typedef unsigned char  DISPARITYTYPE_;
typedef float          GUIDANCETYPE_;      /*float or double*/

namespace deeprob {
namespace cv {

class SGMStereo {
	static const int _nr_buffer_int = 4;

public:
	SGMStereo(int width, int height, int nr_planes = NUM_OF_DISPARITY_LEVEL,
            int c_guidance = 1, int c_input = 1)
            : _width(width), _height(height), _nr_planes(nr_planes)
            , _c_guidance(c_guidance), _c_input(c_input) {
		_cost_vol_buffer = alloc_4<COSTTYPE_>(3, _height, _width, _nr_planes, 0);
		_cost_vol = _cost_vol_buffer[0];
		_cost_vol_right = _cost_vol_buffer[1];
		_cost_vol_backup = _cost_vol_buffer[2];
		//_cost_vol_backup_2 = _cost_vol_buffer[3];

		_disparity_buffer = alloc_3<DISPARITYTYPE_>(3, _height, _width, 0);
		_disparity = _disparity_buffer[0];
		_disparity_r = _disparity_buffer[1];
		_disparity_backup = _disparity_buffer[2];

		int size = _width *_height;
		_gray_left = new unsigned char[size];
		_gray_right = new unsigned char[size];

		_buffer_int = new int[_nr_buffer_int * size];
		_buffer_int_0 = _buffer_int;
		_buffer_int_1 = _buffer_int_0 + size;
		_buffer_int_2 = _buffer_int_1 + size;
		_buffer_int_3 = _buffer_int_2 + size;

		_gf.initialize(SIGMA_S, SIGMA_R, (size_t)width, (size_t)height);

		//_vol_buffer_f = new GUIDANCETYPE_[nr_planes*size];
		//_guidance_left = new GUIDANCETYPE_[_width*_height];
		//_guidance_right = new GUIDANCETYPE_[_width*_height];
	}

	~SGMStereo() {
		free_4<COSTTYPE_>(_cost_vol_buffer);
		free_3<DISPARITYTYPE_>(_disparity_buffer);
		delete[]_gray_left;
		delete[]_gray_right;
		delete[]_buffer_int;
		//delete[]_vol_buffer_f;
		//if (_guidance_left != NULL) delete[]_guidance_left;
		//if (_guidance_right != NULL) delete[]_guidance_right;
	}

	void setImage(const unsigned char* left, const unsigned char* right, int c) {
		_left_input = left;
		_right_input = right;
		_c_input = c;
	}

	void setGuidance(const GUIDANCETYPE_* guidance_left, const GUIDANCETYPE_* guidance_right, int c) {
		//memcpy(_guidance_left, guidance_left, c*_width*_height);
		//memcpy(_guidance_right, guidance_right, c*_width*_height);
		_guidance_left = guidance_left;
		_guidance_right = guidance_right;
		_c_guidance = c;
	}

public:
	void calDisparity();/*main interface, on the reference(left) view*/

	void performSGM(COSTTYPE_* costImage, unsigned char* disparityImage, int width, int height,
		      int nr_planes, int smoothnessPenaltySmall_ = 1 * 7, int smoothnessPenaltyLarge_ = 16 * 7);/*the semi-global matcher, optimized with sse*/

private:
	void calMatchingCost(const unsigned char* left, const unsigned char*right, COSTTYPE_***cost_vol, int nr_planes, int width, int height);

	void stereo_flip_cost_vol(COSTTYPE_***&cost_vol_right, COSTTYPE_***cost_vol, int height, int width, int nr_planes);

	void calGradient_sobel(int*gradient, unsigned char*gray, int width, int height);

	void wta(DISPARITYTYPE_** disparity, COSTTYPE_*** cost_vol);

	void fillHoles(unsigned char* disparity, int width, int height);

	void speckleFilter(unsigned char* image, int width, int height, const int maxSpeckleSize = 200, const int maxDifference = 3);

	void boxfilter(COSTTYPE_* pSrc, COSTTYPE_* pDst, int radius_x, int radius_y, int w, int h, int c);

	void calCensusTransform(unsigned char* gray, int* census, int width, int height, int r_x, int r_y) {
#pragma omp parallel for
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				int census_code = 0, gray_center = gray[y*width + x];
				int y_min = y - r_y, y_max = y + r_y,
					x_min = x - r_x, x_max = x + r_x;
				for (int y_ = y_min; y_ <= y_max; y_ += 2) for (int x_ = x_min; x_ <= x_max; x_ += 2) /*sparsely sampling*/
					//for (int y_ = y_min; y_ <= y_max; y_ ++) for (int x_ = x_min; x_ <= x_max; x_ ++)
				{
					if (y_ >= 0 && y_ < height&&x_ >= 0 && x_ < width&&gray[y_*width + x_] >= gray_center) census_code += 1;
					census_code = census_code << 1;
				}
				census[y*width + x] = census_code;
			}
		}
	}

	inline int calHammingDistance(int leftCencusCode, int rightCensusCode) {
		return  static_cast<int>(_mm_popcnt_u32(static_cast<size_t>(leftCencusCode^rightCensusCode)));
	}

	void genProposalLR(DISPARITYTYPE_** disparity_left, DISPARITYTYPE_** disparity_right);

	void buildNewVol(DISPARITYTYPE_** disparity, int nr_planes, COSTTYPE_***cost_vol);

public:
	DISPARITYTYPE_*** _disparity_buffer, ** _disparity, **_disparity_r, **_disparity_backup;

private:
	int _nr_planes, _width, _height;
	COSTTYPE_**** _cost_vol_buffer, ***_cost_vol, ***_cost_vol_right, ***_cost_vol_backup;
	const GUIDANCETYPE_* _guidance_left, *_guidance_right;

	const unsigned char* _left_input, *_right_input; /*assigned from outside*/
	int _c_guidance, _c_input;
	unsigned char* _gray_left, *_gray_right;
	int* _buffer_int, *_buffer_int_0, *_buffer_int_1, *_buffer_int_2, *_buffer_int_3;
	GeodesicFiltering<GUIDANCETYPE_> _gf;

	//GUIDANCETYPE_* _vol_buffer_f;
};

}   // namespace cv
}   // namespace deeprob

#endif // DEEPROB_CV_SGM_STEREO_H_
