/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file local_stereo.h
 * @brief multi-scale + multi box agggre + disparity propagation
 */

#ifndef DEEPROB_CV_LOCAL_STEREO_H_
#define DEEPROB_CV_LOCAL_STEREO_H_

#include <nmmintrin.h>
#include <vector>
#include "utility.h"
#include "ctmf.h"
#include "geodesic_filtering.h"

typedef unsigned char COSTTYPE;
typedef unsigned char DISPARITYTYPE;
typedef float         GUIDANCETYPE;

#define RADIUS_CENSUS               4
//#define THRES_CENSUS              20
//#define THRES_AD                  20
//#define P_SMOTHNESS               15
#define THRES_ERR_CROSS_CHECK       1
#define DELTA_DISPARITY_RATIO       (0.05f)
#define DELTA_DISPARITY_MAXIMUM     (3)
#define RADIUS_MEDIAN               2
#define SIGMA_S                     (30.f)
#define SIGMA_R                     (18.f)
#define WEIGHT_VERTICAL             0.85 //0.85f~1, trick for running KITTI data sets
#define BASE_SPECKLESIZE            70
#define BASE_WIDTH                  320

//#define PROFILE_RUNTIME

/*V1: fuse two proposals*/
const int radius_x_boxfilter[2] = { 3, 1};
const int radius_y_boxfilter[2] = { 1, 3};

namespace deeprob {
namespace cv {

class LocalStereo {

public:
	static const int _nr_buffer_int = 4;
	static const int _downsample_factor = 2; /*default*/

	LocalStereo(int width, int height, int nr_planes, int c_guidance, int c_input)
            : _width(width), _height(height), _nr_planes(nr_planes)
            , _c_guidance(c_guidance), _c_input(c_input) {
		_disparity_output = new DISPARITYTYPE[_width*_height];
		memset(_disparity_output, 0, sizeof(DISPARITYTYPE)*_width*_height);
		_disparity_output_buffer = new DISPARITYTYPE[_width*_height];
		memset(_disparity_output_buffer, 0, sizeof(DISPARITYTYPE)*_width*_height);
		_width_high = _width;
		_height_high = _height;
		_width /= _downsample_factor;
		_height /= _downsample_factor;
		_nr_planes /= _downsample_factor;

		_cost_vol_buffer = alloc_4<COSTTYPE>(3, _height, _width, _nr_planes, 0);
		_cost_vol = _cost_vol_buffer[0];
		_cost_vol_right = _cost_vol_buffer[1];
		_cost_vol_backup = _cost_vol_buffer[2];
		//_cost_vol_backup_2 = _cost_vol_buffer[3];

		_disparity_buffer = alloc_3<DISPARITYTYPE>(4, _height, _width, 0);
		_disparity = _disparity_buffer[0];
		_disparity_r = _disparity_buffer[1];
		_disparity_backup = _disparity_buffer[2];
		_disparity_backup_2 = _disparity_buffer[3];

		int size = _width *_height;
		_gray_left = new unsigned char[size];
		_gray_right = new unsigned char[size];

		_buffer_int = new int[_nr_buffer_int * size];
		_buffer_int_0 = _buffer_int;
		_buffer_int_1 = _buffer_int_0 + size;
		_buffer_int_2 = _buffer_int_1 + size;
		_buffer_int_3 = _buffer_int_2 + size;

		_gf.initialize(SIGMA_S, SIGMA_R, (size_t)_width, (size_t)_height, WEIGHT_VERTICAL);

		_vol_buffer_f = new GUIDANCETYPE[nr_planes*size];
		_d_buffer = new DISPARITYTYPE[size * (_downsample_factor *_downsample_factor + 1)];

#ifdef PROFILE_RUNTIME
		_timecost_cost = _timecost_proposal = _timecost_propogation = _timecost_upsampling = 0;
#endif

		//_guidance_left = new GUIDANCETYPE[_width*_height];
		//_guidance_right = new GUIDANCETYPE[_width*_height];
	}

	~LocalStereo() {
		free_4<COSTTYPE>(_cost_vol_buffer);
		free_3<DISPARITYTYPE>(_disparity_buffer);
		delete[]_disparity_output;
		delete[]_disparity_output_buffer;
		delete[]_gray_left;
		delete[]_gray_right;
		delete[]_buffer_int;
		delete[]_vol_buffer_f;
		delete[]_d_buffer;
		//if (_guidance_left != NULL) delete[]_guidance_left;
		//if (_guidance_right != NULL) delete[]_guidance_right;
	}

	/*size should be the same as in constructor function, in low resolution!*/
	void setImage(const unsigned char* left, const unsigned char* right, int c) {
		_left_input = left;
		_right_input = right;
		_c_input = c;
	}

	void setGuidance(const GUIDANCETYPE* guidance_left, const GUIDANCETYPE* guidance_left_high, const GUIDANCETYPE* guidance_right, int c) {
		//memcpy(_guidance_left, guidance_left, c*_width*_height);
		//memcpy(_guidance_right, guidance_right, c*_width*_height);
		_guidance_left = guidance_left;
		_guidance_right = guidance_right;
		_c_guidance = c;

		_guidance_left_high = guidance_left_high;/*used for upsampling*/
	}

public:
	DISPARITYTYPE*** _disparity_buffer, ** _disparity, **_disparity_r, **_disparity_backup, **_disparity_backup_2;
	DISPARITYTYPE*_disparity_output, *_disparity_output_buffer;

	/*timecost analysis*/
	double _timecost_cost, _timecost_proposal, _timecost_propogation, _timecost_upsampling;

private:
	int _nr_planes, _width, _height, _width_high, _height_high;
	COSTTYPE**** _cost_vol_buffer, ***_cost_vol, ***_cost_vol_right, ***_cost_vol_backup;
	const GUIDANCETYPE* _guidance_left, *_guidance_right;
	const GUIDANCETYPE* _guidance_left_high;/*used for upsampling*/

	const unsigned char* _left_input, *_right_input; /*assigned from outside*/
	int _c_guidance, _c_input;
	unsigned char* _gray_left, *_gray_right;
	int* _buffer_int, *_buffer_int_0, *_buffer_int_1, *_buffer_int_2, *_buffer_int_3;
	GeodesicFiltering<GUIDANCETYPE> _gf;

	GUIDANCETYPE* _vol_buffer_f;
	DISPARITYTYPE* _d_buffer;/*used when upsampling*/

public:
	void calDisparity();/*on the reference(left) view*/

private:
	void calMatchingCost(const unsigned char* left, const unsigned char*right, COSTTYPE***cost_vol, int nr_planes, int width, int height);

	void stereo_flip_cost_vol(COSTTYPE***&cost_vol_right, COSTTYPE***cost_vol, int height, int width, int nr_planes);

	void calGradient_sobel(int*gradient, unsigned char*gray, int width, int height);

	void wta(DISPARITYTYPE** disparity, COSTTYPE*** cost_vol);
	void wta(DISPARITYTYPE** disparity, COSTTYPE*** cost_vol, COSTTYPE* cost_min);

	void fillHoles(unsigned char* disparity, int width, int height);

	void speckleFilter(unsigned char* image, int width, int height, const int maxSpeckleSize, const int maxDifference = 3);

	void boxfilter(COSTTYPE* pSrc, COSTTYPE* pDst, int radius_x, int radius_y, int w, int h, int c);

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
					//for (int y_ = y_min; y_ <= y_max; ++y_) for (int x_ = x_min; x_ <= x_max; ++x_)
				{
					if (y_ >= 0 && y_ < height&&x_ >= 0 && x_ < width&&gray[y_*width + x_] >= gray_center)
					{
						census_code += 1;
					}
					census_code = (census_code << 1);
				}
				census[y*width + x] = census_code;
			}
		}
	}

	inline int calHammingDistance(int leftCencusCode, int rightCensusCode) {
		return  static_cast<int>(_mm_popcnt_u32(static_cast<size_t>(leftCencusCode^rightCensusCode)));
	}

	void genProposalLR(DISPARITYTYPE** disparity_left, DISPARITYTYPE** disparity_right);

	void fuseProposals(std::vector<DISPARITYTYPE**>& proposals, DISPARITYTYPE** disparity);
	void fuseProposals(std::vector<DISPARITYTYPE**>& proposals, DISPARITYTYPE** disparity, COSTTYPE* cost_min);

	void buildNewVol(DISPARITYTYPE** disparity, int nr_planes, COSTTYPE***cost_vol);

public:
	void upsamplingDisparity(DISPARITYTYPE** disparity, DISPARITYTYPE* disparity_result/*allocated from outside*/,
		    int width_high, int height_high, int upsamling_factor, const GUIDANCETYPE* guidance, int c_guidance/*, int step = 5, int n = 5*/);
};

}   // namespace cv
}   // namespace deeprob

#endif // DEEPROB_CV_LOCAL_STEREO_H_
