/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file local_stereo.cpp
 * @brief local_stereo.cpp
 */

#include "local_stereo.h"

#include <stack>
#include <algorithm>
#include <iostream>
#include <nmmintrin.h>

/*tools for time cost analysis*/
#ifdef PROFILE_RUNTIME
extern LARGE_INTEGER g_nFreq;
extern LARGE_INTEGER g_nStartTime;
extern LARGE_INTEGER g_nEndTime;
extern void startTimeCal();
extern int endTimeCal();
#endif

namespace deeprob {
namespace cv {

void LocalStereo::calDisparity() {
	//matching cost initialization
#ifdef PROFILE_RUNTIME
	startTimeCal();
#endif
	calMatchingCost(_left_input, _right_input, _cost_vol, _nr_planes, _width, _height);
#ifdef PROFILE_RUNTIME
	int timecost_cost = endTimeCal();
	_timecost_cost += timecost_cost;
#endif

#if 0
	//single proposal

	/*test on the idea of relaxed fronto-parallel assumption*/
	COSTTYPE P1 = P_SMOTHNESS; int start_x = _nr_planes / 10;
#pragma omp parallel for
	for (int y = 0; y < _height; ++y)
	{
		for (int x = start_x; x < _width; ++x)
		{
			COSTTYPE costs_per_pixel[256];
			costs_per_pixel[0] = _cost_vol[y][x][0];
			costs_per_pixel[_nr_planes - 1] = _cost_vol[y][x][_nr_planes - 1];
			for (int d_ = 1; d_ < _nr_planes - 1; ++d_)
			{
				costs_per_pixel[d_] = __min(__min(_cost_vol[y][x][d_], _cost_vol[y][x][d_ + 1] + P1), _cost_vol[y][x][d_ - 1] + P1);
			}
			memcpy(_cost_vol[y][x], costs_per_pixel, sizeof(COSTTYPE)*_nr_planes);
		}
	}

	boxfilter(_cost_vol[0][0], _cost_vol_backup[0][0], radius_x_boxfilter[0], radius_x_boxfilter[0], _width, _height, _nr_planes);
	stereo_flip_cost_vol(_cost_vol_right, _cost_vol_backup, _height, _width, _nr_planes);
	wta(_disparity, _cost_vol_backup);
	wta(_disparity_r, _cost_vol_right);

	ctmf(_disparity[0], _disparity_backup[0], _width, _height, _width, _width, RADIUS_MEDIAN - 1, 1, _height*_width);
	std::memcpy(_disparity[0], _disparity_backup[0], sizeof(unsigned char)*_width*_height);
	ctmf(_disparity_r[0], _disparity_backup[0], _width, _height, _width, _width, RADIUS_MEDIAN - 1, 1, _height*_width);
	std::memcpy(_disparity_r[0], _disparity_backup[0], sizeof(unsigned char)*_width*_height);

	genProposalLR(_disparity, _disparity_r);
	//std::memcpy(_disparity_backup[0], _disparity[0], sizeof(unsigned char)*_width*_height);
	//ctmf(_disparity_backup[0], _disparity[0], _width, _height, _width, _width, RADIUS_MEDIAN - 2, 1, _height*_width);
#else
	//box filtering cost aggregation & disparity proposal generation
#ifdef PROFILE_RUNTIME
	startTimeCal();
#endif
	std::vector<DISPARITYTYPE**> proposals;

//	/*test on the idea of relaxed fronto-parallel assumption*/
//	COSTTYPE P1 = P_SMOTHNESS; int start_x = _nr_planes / 5;
//#pragma omp parallel for
//	for (int y = 0; y < _height; ++y)
//	{
//		for (int x = start_x; x < _width; ++x)
//		{
//			COSTTYPE costs_per_pixel[256];
//			costs_per_pixel[0] = _cost_vol[y][x][0];
//			costs_per_pixel[_nr_planes - 1] = _cost_vol[y][x][_nr_planes - 1];
//			for (int d_ = 1; d_ < _nr_planes - 1; ++d_)
//			{
//				costs_per_pixel[d_] = __min(__min(_cost_vol[y][x][d_], _cost_vol[y][x][d_ + 1] + P1), _cost_vol[y][x][d_ - 1] + P1);
//			}
//			std::memcpy(_cost_vol[y][x], costs_per_pixel, sizeof(COSTTYPE)*_nr_planes);
//		}
//	}

	//COSTTYPE* cost_min = new COSTTYPE[_width*_height * 3];/*hardcode 3 layers*/
	boxfilter(_cost_vol[0][0], _cost_vol_backup[0][0], radius_x_boxfilter[0], radius_y_boxfilter[0], _width, _height, _nr_planes);
	stereo_flip_cost_vol(_cost_vol_right, _cost_vol_backup, _height, _width, _nr_planes);
	wta(_disparity, _cost_vol_backup/*, cost_min*/);
	wta(_disparity_r, _cost_vol_right);

	//ctmf(_disparity[0], _disparity_backup[0], _width, _height, _width, _width, RADIUS_MEDIAN - 1, 1, _height*_width);
	//std::memcpy(_disparity[0], _disparity_backup[0], sizeof(unsigned char)*_width*_height);
	//ctmf(_disparity_r[0], _disparity_backup[0], _width, _height, _width, _width, RADIUS_MEDIAN - 1, 1, _height*_width);
	//std::memcpy(_disparity_r[0], _disparity_backup[0], sizeof(unsigned char)*_width*_height);

	genProposalLR(_disparity, _disparity_r);
	proposals.push_back(_disparity);

	boxfilter(_cost_vol[0][0], _cost_vol_backup[0][0], radius_x_boxfilter[1], radius_y_boxfilter[1], _width, _height, _nr_planes);
	stereo_flip_cost_vol(_cost_vol_right, _cost_vol_backup, _height, _width, _nr_planes);
	wta(_disparity_backup, _cost_vol_backup/*, cost_min + _width*_height*/);
	wta(_disparity_r, _cost_vol_right);

	//ctmf(_disparity_backup[0], _disparity_backup_2[0], _width, _height, _width, _width, RADIUS_MEDIAN - 1, 1, _height*_width);
	//std::memcpy(_disparity_backup[0], _disparity_backup_2[0], sizeof(unsigned char)*_width*_height);
	//ctmf(_disparity_r[0], _disparity_backup_2[0], _width, _height, _width, _width, RADIUS_MEDIAN - 1, 1, _height*_width);
	//std::memcpy(_disparity_r[0], _disparity_backup_2[0], sizeof(unsigned char)*_width*_height);

	genProposalLR(_disparity_backup, _disparity_r);
	proposals.push_back(_disparity_backup);

	/*extend to 3 proposals!*/
	//boxfilter(_cost_vol[0][0], _cost_vol_backup[0][0], radius_x_boxfilter[2], radius_y_boxfilter[2], _width, _height, _nr_planes);
	//stereo_flip_cost_vol(_cost_vol_right, _cost_vol_backup, _height, _width, _nr_planes);
	//wta(_disparity_backup_2, _cost_vol_backup, cost_min + 2*_width*_height);
	//wta(_disparity_r, _cost_vol_right);
	//genProposalLR(_disparity_backup_2, _disparity_r);
	//proposals.push_back(_disparity_backup_2);

	fuseProposals(proposals, _disparity);
	//imdebug("lum b=8  w=%d h=%d %p", _width, _height, _disparity[0]);

	//delete[]cost_min;
	//std::memcpy(_disparity_backup[0], _disparity[0], sizeof(unsigned char)*_width*_height);
	//ctmf(_disparity_backup[0], _disparity[0], _width, _height, _width, _width, RADIUS_MEDIAN, 1, _height*_width);
#ifdef PROFILE_RUNTIME
	int timecost_proposal = endTimeCal();
	_timecost_proposal += timecost_proposal;
#endif
#endif

	//ctmf(_disparity_backup[0], _disparity[0], _width, _height, _width, _width, RADIUS_MEDIAN - 1, 1, _height*_width);
	//fillHoles(_disparity[0], _width, _height);
	//return;

	//disparity propagation
	//buildNewVol(_disparity, _nr_planes, _cost_vol);
	//_gf.runFilteringRecursive<COSTTYPE>(_cost_vol[0][0], _nr_planes, _guidance_left, _c_guidance, _disparity[0], 2);
	//wta(_disparity, _cost_vol);

#ifdef PROFILE_RUNTIME
	startTimeCal();
#endif
	GUIDANCETYPE* data = _vol_buffer_f;// new GUIDANCETYPE[_nr_planes*_width*_height];
	int delta_disparity = __min(DELTA_DISPARITY_RATIO*_nr_planes, DELTA_DISPARITY_MAXIMUM);

#pragma omp parallel for
	for (int y = 0; y < _height; y++){
		for (int x = 0; x < _width; x++){
			int d = _disparity[y][x];
			int idx_pts = y*_width + x;
			if (d>0)
			{
				for (int d_ = 0; d_ < _nr_planes; ++d_)
				{
					GUIDANCETYPE delta = __min(i_abs(d - d_), delta_disparity);
					data[_nr_planes*(idx_pts)+d_] = delta*delta;
				}
			}
			else
			{
				memset(&data[_nr_planes*(idx_pts)], 0, sizeof(GUIDANCETYPE)*_nr_planes);
			}
		}
	}
	//GUIDANCETYPE* data_cpy = new GUIDANCETYPE[_nr_planes*_width*_height];
	//memcpy(data_cpy, data, sizeof(GUIDANCETYPE)*_nr_planes*_width*_height);
	//BoxFilter(data_cpy, data, 2, 2, _width, _height, _nr_planes);
	//delete[]data_cpy;
	_gf.setDataGuidence(data, _nr_planes, _guidance_left, _c_guidance);
	if (_c_guidance == 3)
		_gf.genWeightsColor();
	else
		_gf.genWeightsGray();

	//_gf.runFilteringExact(2);
	_gf.runFilteringExact(1);
#pragma omp parallel for
	for (int y = 0; y < _height; y++) {
		for (int x = 0; x < _width; x++) {
			//if (_disparity[y][x]>0) continue;
			int d = 0, idx_pts = y*_width + x;
			vec_min_pos(d, &data[_nr_planes*(idx_pts)], _nr_planes);
			_disparity[y][x] = d;
			//if (d>0 && d < _nr_planes - 1)//sub-pixel interpolation
			//{
			//	int centerCostValue = data[_nr_planes*(idx_pts)+d];
			//	int leftCostValue = data[_nr_planes*(idx_pts)+d - 1];
			//	int rightCostValue = data[_nr_planes*(idx_pts)+d + 1];
			//	double delta_d = static_cast<double>(rightCostValue - leftCostValue) / (2 * (leftCostValue + rightCostValue - 2 * centerCostValue));
			//	if (i_abs(delta_d) <= 1)  _disparity[y][x] = static_cast<int>(d - delta_d);
			//}
		}
	}
	//delete[]data;

	int factor_size = __max(i_round((float)_width / BASE_WIDTH), 1);
	int maxSpeckleSize = factor_size * factor_size * BASE_SPECKLESIZE;
	speckleFilter(_disparity[0], _width, _height, maxSpeckleSize);
	fillHoles(_disparity[0], _width, _height);
	//imdebug("lum *auto b=8 w=%d h=%d %p", _width, _height, _disparity[0]);

	memcpy(_disparity_backup[0], _disparity[0], sizeof(unsigned char)*_width*_height);
	ctmf(_disparity_backup[0], _disparity[0], _width, _height, _width, _width, RADIUS_MEDIAN + 1, 1, _height*_width);

#ifdef PROFILE_RUNTIME
	int timecost_propagation = endTimeCal();
	_timecost_propogation += timecost_propagation;
#endif

	/*final step: upsample the result to original resolution*/
	upsamplingDisparity(_disparity, _disparity_output, _width_high, _height_high, _downsample_factor, _guidance_left_high, _c_guidance);
}

void LocalStereo::genProposalLR(DISPARITYTYPE** disparity_left, DISPARITYTYPE** disparity_right) {
#pragma omp parallel for
	for (int y = 0; y < _height; ++y)
	{
		for (int x = 0; x < _width; ++x)
		{
			int d = disparity_left[y][x];
			int d_cor = x>d&&d>0 ? disparity_right[y][x - d] : 0;
			bool passed_LRC = (d > 0 && i_abs(d - d_cor) < THRES_ERR_CROSS_CHECK) ? true : false;
			if (!passed_LRC)
			{
				disparity_left[y][x] = 0;
			}
		}
	}
}

void LocalStereo::fuseProposals(std::vector<DISPARITYTYPE**>& proposals, DISPARITYTYPE** disparity) {
	//first version, hardcode the case of 2 proposals
#pragma omp parallel for
	for (int y = 0; y < _height; ++y)
	{
		for (int x = 0; x < _width; ++x)
		{
			DISPARITYTYPE d1 = proposals[0][y][x];
			DISPARITYTYPE d2 = proposals[1][y][x];
			if (d1 == 0 || d2 == 0) disparity[y][x] = d1 + d2;
			else if (i_abs(d1 - d2) < THRES_ERR_CROSS_CHECK * 3)
			{
				//disparity[y][x] = d1;/*from smaller scale thus supposed to have better details*/
				//disparity[y][x] = __min(d1, d2);
				disparity[y][x] = (d1 + d2) / 2;
			}
			else
			{
				disparity[y][x] = 0;
			}
		}
	}
}

void LocalStereo::fuseProposals(std::vector<DISPARITYTYPE**>& proposals, DISPARITYTYPE** disparity, COSTTYPE* cost_min) {
	//hardcode the case of 2 or 3 proposals
#pragma omp parallel for
	for (int y = 0; y < _height; ++y)
	{
		for (int x = 0; x < _width; ++x)
		{
			int idx_pt = y*_width + x;
			DISPARITYTYPE d1 = proposals[0][y][x];
			DISPARITYTYPE d2 = proposals[1][y][x];
			//DISPARITYTYPE d3 = proposals[2][y][x];
			COSTTYPE cost1 = cost_min[idx_pt];
			COSTTYPE cost2 = cost_min[idx_pt + _width*_height];
			//COSTTYPE cost3 = cost_min[idx_pt + 2*_width*_height];
			if (d1 == 0 || d2 == 0)
			{
				disparity[y][x] = d1 + d2;
			}
			else if (i_abs(d1 - d2) <= THRES_ERR_CROSS_CHECK * 2)
			{
				//disparity[y][x] = d1;/*from smaller scale thus supposed to have better details*/
				disparity[y][x] = (d1 + d2) / 2;
				//disparity[y][x] = __min(d1, d2);

				//disparity[y][x] = cost1 <= cost2 ? d1 : d2;
			}
			else
			{
				//disparity[y][x] = 0;
				disparity[y][x] = cost1 <= cost2 ? d1 : d2;
			}

			//if (d1 + d2 + d3 > 0)
			//{
			//	std::vector<std::pair<int, COSTTYPE>> ds;
			//	if (d1 > 0) ds.push_back(std::make_pair(d1, cost1));
			//	if (d2 > 0) ds.push_back(std::make_pair(d2, cost2));
			//	if (d3 > 0) ds.push_back(std::make_pair(d3, cost3));
			//	if (ds.size() == 1)
			//	{
			//		disparity[y][x] = ds[0].first;
			//	}
			//	else if (ds.size() > 1)
			//	{
			//		int d_res = ds[0].first;
			//		COSTTYPE cost_minimum = ds[0].second;
			//		for (int i = 1; i < ds.size(); ++i)
			//		{
			//			if (ds[i].second < cost_minimum){ d_res = ds[i].first; cost_minimum = ds[i].second; }
			//		}
			//	}
			//}
			//else
			//	disparity[y][x] = 0;
		}
	}
}

void LocalStereo::upsamplingDisparity(DISPARITYTYPE** disparity, DISPARITYTYPE* disparity_result/*allocated from outside*/,
	   int width_high, int height_high, int upsamling_factor, const GUIDANCETYPE* guidance, int c_guidance/*,int step = 5, int n = 5*/) {
	int width_h = width_high, height_h = height_high;
	GUIDANCETYPE* disparity_high = new GUIDANCETYPE[height_h*width_h];
	//DISPARITYTYPE* mask_reliable = _d_buffer;// new DISPARITYTYPE[height_h*width_h];
	//memset(mask_reliable, 0, sizeof(unsigned char)*height_h*width_h);
	//int nr_levels = 0;

#ifdef PROFILE_RUNTIME
	startTimeCal();
#endif

//	//nn sampling
//#pragma omp parallel for
//	for (int y = 0; y < height_h; ++y) for (int x = 0; x < width_h; ++x)
//	{
//		int idx_pts_h = y*width_h + x;
//		int x_low = i_round((float)x / upsamling_factor), y_low = i_round((float)y / upsamling_factor);
//		if (x_low<0 || x_low>_width - 1 || y_low <0 || y_low>_height - 1)
//		{
//			disparity_result[y*width_h + x] = 0;
//			disparity_high[y*width_h + x] = 0;
//			continue;
//		}
//		disparity_result[idx_pts_h] = disparity_high[y*width_h + x] = upsamling_factor * disparity[y_low][x_low];
//		//if (disparity_result[idx_pts_h] > nr_levels)nr_levels = disparity_result[idx_pts_h];
//		if (n * disparity_result[idx_pts_h] > (_nr_planes * upsamling_factor))//BG pixels
//			mask_reliable[idx_pts_h] = 255;
//	}

	//bilinear sampling
	float factor_high_to_low = 1.f / upsamling_factor;
#pragma omp parallel for
	for (int y = 0; y < height_h; ++y) for (int x = 0; x < width_h; ++x)
	{
		int idx_pts_h = y*width_h + x;
		float x_low = x*factor_high_to_low, y_low = y*factor_high_to_low;
		if (x_low<0 || x_low>_width - 1 || y_low <0 || y_low>_height - 1)
		{
			disparity_result[y*width_h + x] = 0;
			disparity_high[y*width_h + x] = 0;
			continue;
		}
		DISPARITYTYPE sampling_value = 0;
		getDataBilinearGray(x_low, y_low, disparity[0], sampling_value, _width, _height);
		disparity_result[idx_pts_h] = disparity_high[idx_pts_h] = i_round(upsamling_factor * sampling_value);
		//if (disparity_result[idx_pts_h] > nr_levels)nr_levels = disparity_result[idx_pts_h];
		/*
		if (n * disparity_result[idx_pts_h] > (_nr_planes * upsamling_factor))//BG pixels
			mask_reliable[idx_pts_h] = 255;
		*/
	}

	int nr_planes_high = _nr_planes*upsamling_factor;
	//	nr_levels += 1;
	//	DISPARITYTYPE* disparity_vol = new DISPARITYTYPE[width_high*height_high*nr_levels];
	//#pragma omp parallel for
	//	for (int y = 0; y < height_h; ++y)
	//	{
	//		for (int x = 0; x < width_h; ++x)
	//		{
	//			int idx_pts_h = y*width_h + x;
	//			int idx_pt_levels = nr_levels*idx_pts_h;
	//			int d_res = disparity_result[idx_pts_h];
	//			for (int d = 0; d < nr_levels; ++d)
	//			{
	//				disparity_vol[idx_pt_levels + d] = __min(abs(d_res - d), DELTA_DISPARITY_MAXIMUM);
	//			}
	//		}
	//	}

	//#pragma omp parallel for
	//	for (int y = 0; y < height_h; ++y) for (int x = 0; x < width_h; ++x)
	//	{
	//		//mask out occluded region detected asymmetrically
	//		int idx_pts_h = y*width_h + x;
	//		int d = disparity_result[idx_pts_h], x_end = __min(width_h - 1, x + nr_planes_high);
	//		for (int x_ = x + 1; x_ <= x_end; x_ += step) // sparsely sampled
	//		{
	//			if (x_ + d <= x + disparity_result[y*width_h + x_])
	//			{
	//				mask_reliable[idx_pts_h] = 0;
	//				break;
	//			}
	//		}
	//	}
	fillHoles(disparity_result, width_h, height_h);

	//filtering
	//imdebug("lum *auto w=%d h=%d %p", width_h, height_h, mask_reliable);
	//imdebug("lum *auto b=32f w=%d h=%d %p", width_h, height_h, disparity_high);

	GeodesicFiltering<GUIDANCETYPE> gf;
	gf.initialize(SIGMA_S / 4, SIGMA_R / 4, (size_t)width_h, (size_t)height_h);
	gf.setDataGuidence(disparity_high, 1, guidance, c_guidance);
	if (c_guidance == 1)
	{
		gf.genWeightsGray();
	}
	else
	{
		gf.genWeightsColor();
	}
	gf.runFilteringExact(1);

	//gf.runFilteringRecursive<DISPARITYTYPE>(disparity_result, 1, guidance, c_guidance, mask_reliable, 1);
	//gf.runFilteringRecursive<DISPARITYTYPE>(disparity_vol, nr_levels, guidance, c_guidance, mask_reliable, 1);
	//gf.runFilteringRecursive<DISPARITYTYPE>(disparity_result, 1, guidance, c_guidance, 1);
	//imdebug("lum *auto b=32f w=%d h=%d %p", width_h, height_h, disparity_high);

#pragma omp parallel for
	for (int y = 0; y < height_h; ++y) for (int x = 0; x < width_h; ++x)
	{
		int idx_pts_h = y*width_h + x;
		int d_filtered = __min((int)(disparity_high[idx_pts_h] + 0.5f), 255);
#if 0
		/*better visual quality*/
		disparity_result[idx_pts_h] = d_filtered;
#else
		/*KITTI, minor adjustment*/
		int d_ori = disparity_result[idx_pts_h];
		disparity_result[idx_pts_h] = abs(d_ori - d_filtered) < DELTA_DISPARITY_MAXIMUM ? d_filtered : d_ori;
#endif
	}

	/*
	DISPARITYTYPE* disparity_result_tmp = _disparity_output_buffer;
	memcpy(disparity_result_tmp, disparity_result, sizeof(DISPARITYTYPE)*width_h*height_h);
	ctmf(disparity_result_tmp, disparity_result, width_h, height_h, width_h, width_h, RADIUS_MEDIAN, 1, height_h*width_h);
	*/
#ifdef PROFILE_RUNTIME
	int timecost_upsampling = endTimeCal();
	_timecost_upsampling += timecost_upsampling;
#endif

	//delete[]disparity_vol;
	//delete[]mask_reliable;
	delete[]disparity_high;
}
void LocalStereo::buildNewVol(DISPARITYTYPE** disparity, int nr_planes, COSTTYPE***cost_vol)
{
	int delta_disparity = __min(DELTA_DISPARITY_RATIO*_nr_planes, DELTA_DISPARITY_MAXIMUM);

#pragma omp parallel for
	for (int y = 0; y < _height; y++)
	{
		for (int x = 0; x < _width; x++)
		{
			//left view (reference view only)
			int d = disparity[y][x];
			if (d>0)
			{
				for (int d_ = 0; d_ < _nr_planes; ++d_)
				{
					COSTTYPE delta = __min(i_abs(d - d_), delta_disparity);
					cost_vol[y][x][d_] = delta*delta;
				}
			}
			else
			{
				memset(cost_vol[y][x], 0, sizeof(COSTTYPE)*_nr_planes);
			}
		}
	}
}
void LocalStereo::calMatchingCost(const unsigned char* left, const unsigned char*right, COSTTYPE***cost_vol, int nr_planes, int width, int height)
{
#if 1
	unsigned char* gray_left = _gray_left, *gray_right = _gray_right;
	int	*census_left = _buffer_int_0, *census_right = _buffer_int_1;
	int	*gradient_left = _buffer_int_2, *gradient_right = _buffer_int_3;

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int idx_pt = y*width + x;
			gray_left[idx_pt] = _c_input == 3 ? rgb_2_gray(&left[idx_pt * 3]) : left[idx_pt];
			gray_right[idx_pt] = _c_input == 3 ? rgb_2_gray(&right[idx_pt * 3]) : right[idx_pt];
		}
	}

	//imdebug("lum *auto b=8 w=%d h=%d %p", width, height, gray_left);
	//imdebug("lum *auto b=8 w=%d h=%d %p", width, height, gray_right);
	calGradient_sobel(gradient_left, gray_left, width, height);
	calGradient_sobel(gradient_right, gray_right, width, height);
	calCensusTransform(gray_left, census_left, width, height, RADIUS_CENSUS, RADIUS_CENSUS);
	calCensusTransform(gray_right, census_right, width, height, RADIUS_CENSUS, RADIUS_CENSUS);

//#pragma omp parallel for
//	for (int y = 0; y < height; ++y)
//	{
//		for (int x = 0; x < width; ++x)
//		{
//			int idx_pt = y*width + x;
//			gray_left[idx_pt] = (gray_left[idx_pt] >> 2);
//			gray_right[idx_pt] = (gray_right[idx_pt] >> 2);
//		}
//	}

#pragma omp parallel for
	for (int y = 0; y < height; ++y){
		COSTTYPE* pCostRow = cost_vol[y][0];
		for (int x = 0; x < width; ++x){
			int idx_pt = y*width + x;
			COSTTYPE* pCost = pCostRow;
			for (int i = 0; i < nr_planes; ++i)
			{
				if (x >= i)
				{
					COSTTYPE cost_gradient = (COSTTYPE)i_abs(gradient_left[idx_pt] - gradient_right[idx_pt - i]);
					COSTTYPE cost_census = (COSTTYPE)calHammingDistance(census_left[idx_pt], census_right[idx_pt - i]);
					//if (cost_census>THRES_CENSUS) cost_census = THRES_CENSUS;
					//cost_vol[y][x][i] = (cost_census << 1) + cost_gradient;
					COSTTYPE cost_ad = (COSTTYPE)i_abs((int)gray_left[idx_pt] - (int)gray_right[idx_pt - i]);
					pCost[i] = ((cost_ad >> 2) + cost_gradient + (cost_census << 1));
				}
				else
				{
					pCost[i] = pCost[i - 1];
				}
			}
			pCostRow += nr_planes;
		}
	}
#else //version optimized with sse
	/*
	in this implementation, the disparity levels should be multiples of 16
	*/
	unsigned char* gray_left = _gray_left, *gray_right = _gray_right;
	//int	*census_left = _buffer_int_0, *census_right = _buffer_int_1;
	int	*gradient_left = _buffer_int_2, *gradient_right = _buffer_int_3;

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int idx_pt = y*width + x;
			gray_left[idx_pt] = _c_input == 3 ? rgb_2_gray(&left[idx_pt * 3]) : left[idx_pt];
			gray_right[idx_pt] = _c_input == 3 ? rgb_2_gray(&right[idx_pt * 3]) : right[idx_pt];
		}
	}
	calGradient_sobel(gradient_left, gray_left, width, height);
	calGradient_sobel(gradient_right, gray_right, width, height);

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int idx_pt = y*width + x;
			gray_left[idx_pt] /= 8;
			gray_right[idx_pt] /= 8;
		}
	}

	//modified with sse
	const int nr_loop = nr_planes >> 4; /*calculate cost for every 16 disparity levels*/
	COSTTYPE* cost_volume = cost_vol[0][0];

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		int image_offset = y*width;
		auto ptr_cost = cost_volume + image_offset * nr_planes;
		for (int x = 0; x < width; ++x)
		{
			int idx_pt = y*width + x;

			char ref = (char)gradient_left[idx_pt];
			//__m128i r0 = _mm_set_epi8(ref, ref, ref, ref, ref, ref, ref, ref, ref, ref, ref, ref, ref, ref, ref, ref);
			__m128i r0 = _mm_set1_epi8(ref);

			char ref_2 = (char)gray_left[idx_pt];
			//__m128i r0_2 = _mm_set_epi8(ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2, ref_2);
			__m128i r0_2 = _mm_set1_epi8(ref_2);

			char p_tar[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
			char p_tar_2[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
			int d = 0, d_cpy;
			for (int j = 0; j < nr_loop; ++j)
			{
				d_cpy = d;
				fillData_sse_16char(p_tar, gradient_right, idx_pt, x, _nr_planes, d);
				fillData_sse_16char(p_tar_2, gray_right, idx_pt, x, _nr_planes, d_cpy);

				//gradient cost
				__m128i r1 = _mm_setr_epi8(p_tar[0], p_tar[1], p_tar[2], p_tar[3], p_tar[4], p_tar[5], p_tar[6], p_tar[7],
					p_tar[8], p_tar[9], p_tar[10], p_tar[11], p_tar[12], p_tar[13], p_tar[14], p_tar[15]);
				__m128i r2 = _mm_sub_epi8(r0, r1);
				__m128i r3 = _mm_abs_epi8(r2);

				//ad
				__m128i r1_2 = _mm_setr_epi8(p_tar_2[0], p_tar_2[1], p_tar_2[2], p_tar_2[3], p_tar_2[4], p_tar_2[5], p_tar_2[6], p_tar_2[7],
					p_tar_2[8], p_tar_2[9], p_tar_2[10], p_tar_2[11], p_tar_2[12], p_tar_2[13], p_tar_2[14], p_tar_2[15]);
				__m128i r2_2 = _mm_sub_epi8(r0_2, r1_2);
				__m128i r3_2 = _mm_abs_epi8(r2_2);

				__m128i r_fill = _mm_add_epi8(r3, r3_2);

				//_mm_store_si128((__m128i*)ptr_cost, r3);
				_mm_store_si128((__m128i*)ptr_cost, r_fill);
				ptr_cost += 16;
			}
		}
	}
#endif
}

void LocalStereo::stereo_flip_cost_vol(COSTTYPE***&cost_vol_right, COSTTYPE***cost_vol, int height, int width, int nr_planes)
{
#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width - nr_planes; x++) for (int d = 0; d < nr_planes; d++) cost_vol_right[y][x][d] = cost_vol[y][x + d][d];
		for (int x = width - nr_planes; x < width; x++) for (int d = 0; d < nr_planes; d++)
		{
			if ((x + d) < width)
				cost_vol_right[y][x][d] = cost_vol[y][x + d][d];
			else
				cost_vol_right[y][x][d] = cost_vol_right[y][x][d - 1];
		}
	}
}
void LocalStereo::calGradient_sobel(int*gradient, unsigned char*gray, int width, int height)
{
	//memset(gradient, 0, sizeof(int)*height*width);
	int p1 = (height - 1)*width, p2 = width - 1;
	for (int x = 0; x < width; ++x) gradient[x] = gradient[p1 + x] = 0;
	for (int y = 0; y < height; ++y)
	{
		int ywidth = y*width;
		gradient[ywidth] = gradient[ywidth + p2] = 0;
	}
#pragma omp parallel for
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			int sobelValue = ((int)gray[width*(y - 1) + x + 1] + 2 * (int)gray[width*y + x + 1] + (int)gray[width*(y + 1) + x + 1])
				- ((int)gray[width*(y - 1) + x - 1] + 2 * (int)gray[width*y + x - 1] + (int)gray[width*(y + 1) + x - 1]);
			gradient[width*y + x] = (sobelValue>15 ? 30 : (sobelValue < -15 ? 0 : sobelValue + 15));/*using the same mapping range as Yamaguchi*/
		}
	}
}

void LocalStereo::wta(DISPARITYTYPE** disparity, COSTTYPE*** cost_vol)
{
	int w = _width, h = _height, nr_planes = _nr_planes;
#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int d = 0;
			vec_min_pos(d, cost_vol[y][x], nr_planes);
			disparity[y][x] = d;
		}
	}
}
void LocalStereo::wta(DISPARITYTYPE** disparity, COSTTYPE*** cost_vol, COSTTYPE* cost_min)
{
	int w = _width, h = _height, nr_planes = _nr_planes;
#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			int d = 0;
			vec_min_pos(d, cost_vol[y][x], nr_planes);
			disparity[y][x] = d;
			cost_min[y*w + x] = cost_vol[y][x][d];
		}
	}
}
void LocalStereo::speckleFilter(unsigned char* image, int width, int height, const int maxSpeckleSize, const int maxDifference)
{
	int width_ = width, height_ = height;
	std::vector<int> labels(width_*height_, 0);
	std::vector<bool> regionTypes(1);
	regionTypes[0] = false;
	int currentLabelIndex = 0;

	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			int pixelIndex = width_*y + x;
			if (image[pixelIndex] != 0) {
				if (labels[pixelIndex] > 0) {
					image[width_*y + x] = regionTypes[labels[pixelIndex]] ? 0 : image[width_*y + x];
				}
				else {
					std::stack<int> wavefrontIndices;
					wavefrontIndices.push(pixelIndex);
					++currentLabelIndex;
					regionTypes.push_back(false);
					int regionPixelTotal = 0;
					labels[pixelIndex] = currentLabelIndex;

					while (!wavefrontIndices.empty()) {
						int currentPixelIndex = wavefrontIndices.top();
						wavefrontIndices.pop();
						//int currentX = currentPixelIndex%width_;
						int currentY = currentPixelIndex / width_;
						int currentX = currentPixelIndex - currentY*width_;
						++regionPixelTotal;
						short pixelValue = image[width_*currentY + currentX];

						if (currentX < width_ - 1 && labels[currentPixelIndex + 1] == 0
							&& image[width_*currentY + currentX + 1] != 0
							&& i_abs(pixelValue - image[width_*currentY + currentX + 1]) <= maxDifference)
						{
							labels[currentPixelIndex + 1] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex + 1);
						}

						if (currentX > 0 && labels[currentPixelIndex - 1] == 0
							&& image[width_*currentY + currentX - 1] != 0
							&& i_abs(pixelValue - image[width_*currentY + currentX - 1]) <= maxDifference)
						{
							labels[currentPixelIndex - 1] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex - 1);
						}

						if (currentY < height_ - 1 && labels[currentPixelIndex + width_] == 0
							&& image[width_*(currentY + 1) + currentX] != 0
							&& i_abs(pixelValue - image[width_*(currentY + 1) + currentX]) <= maxDifference)
						{
							labels[currentPixelIndex + width_] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex + width_);
						}

						if (currentY > 0 && labels[currentPixelIndex - width_] == 0
							&& image[width_*(currentY - 1) + currentX] != 0
							&& std::abs(pixelValue - image[width_*(currentY - 1) + currentX]) <= maxDifference)
						{
							labels[currentPixelIndex - width_] = currentLabelIndex;
							wavefrontIndices.push(currentPixelIndex - width_);
						}
					}

					if (regionPixelTotal <= maxSpeckleSize) {
						regionTypes[currentLabelIndex] = true;
						image[width_*y + x] = 0;
					}
				}
				//
			}
		}
	}
}
void LocalStereo::fillHoles(unsigned char* disparity, int width, int height)
{
	/*horizontal*/
#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int ywidth = y*width;
			if (disparity[ywidth + x] == 0)
			{
				int x_l, x_r;
				int val_l = 0;
				for (x_l = __max(x - 1, 0); x_l >= 0 && val_l == 0; x_l--) val_l = disparity[ywidth + x_l];

				int val_r = 0;
				for (x_r = __min(x + 1, width - 1); x_r<width&&val_r == 0; x_r++) val_r = disparity[ywidth + x_r];

				unsigned char d_ipol = val_l*val_r == 0 ? (val_l + val_r) : __min(val_l, val_r);
				if (x_r>x_l)
				{
					for (int x_in = x_l + 1; x_in <= x_r - 1; ++x_in)
					{
						disparity[ywidth + x_in] = d_ipol;
					}
				}
			}
		}
	}

	/*vertical*/
#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (disparity[y*width + x] == 0)
			{
				int val_t = 0; int y_t, y_b;
				for (y_t = __max(y - 1, 0); y_t >= 0 && val_t == 0; y_t--) val_t = disparity[y_t*width + x];

				int val_b = 0;
				for (y_b = __min(y + 1, height - 1); y_b<height&&val_b == 0; y_b++) val_b = disparity[y_b*width + x];

				unsigned char d_ipol = val_t*val_b == 0 ? (val_t + val_b) : __min(val_t, val_b);
				if (y_b>y_t)
				{
					for (int y_in = y_t + 1; y_in <= y_b - 1; ++y_in)
					{
						disparity[y_in*width + x] = d_ipol;
					}
				}
			}
		}
	}
}
void LocalStereo::boxfilter(COSTTYPE* pSrc, COSTTYPE* pDst, int radius_x, int radius_y, int w, int h, int c)
{
	//base-line version
#if 0
	int max_length = w;
	if (h > w) max_length = h;
	float* pSum = new float[(max_length + 1)*c];
	memset(pSum, 0, sizeof(float)*(max_length + 1)*c);
	float nor_x = 1.0f / (2 * radius_x + 1), nor_y = 1.0f / (2 * radius_y + 1);

	/*x direction*/
	for (int y = 0; y < h; y++)
	{
		for (int n = 0; n < c; ++n) pSum[n] = 0;
		for (int x = 0; x < w; ++x)for (int n = 0; n < c; ++n)
			pSum[(x + 1)*c + n] = pSum[x*c + n] + (float)pSrc[(y*w + x)*c + n];
		for (int x = 0; x < radius_x; ++x)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE)(pSum[(x + radius_x + 1)*c + n] / (float)(x + 1 + radius_x));
		for (int x = radius_x; x < w - radius_x - 1; ++x)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE)(nor_x*(pSum[(x + radius_x + 1)*c + n] - pSum[(x - radius_x)*c + n]));
		for (int x = w - radius_x - 1; x < w; ++x) for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE)((pSum[w*c + n] - pSum[(x - radius_x)*c + n]) / (float)(w - x + radius_x));
	}
	/*y direction*/
	for (int x = 0; x < w; x++)
	{
		for (int n = 0; n < c; ++n) pSum[n] = 0;
		for (int y = 0; y < h; y++)for (int n = 0; n < c; ++n)
			pSum[(y + 1)*c + n] = pSum[y*c + n] + (float)pDst[(y*w + x)*c + n];
		for (int y = 0; y < radius_y; y++)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE)(pSum[(y + radius_y + 1)*c + n] / (float)(y + 1 + radius_y));
		for (int y = radius_y; y < h - radius_y - 1; y++)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE)(nor_y*(pSum[(y + radius_y + 1)*c + n] - pSum[(y - radius_y)*c + n]));
		for (int y = h - radius_y - 1; y < h; ++y)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE)((pSum[h*c + n] - pSum[(y - radius_y)*c + n]) / (float)(h - y + radius_y));
	}

	delete[]pSum;

#else
	/*openmp optimized version*/
	int max_length = w;
	if (h > w) max_length = h;
	int coreNum = omp_get_num_procs();   /*get the number of processors*/
	GUIDANCETYPE* pSumFull = _vol_buffer_f;// new GUIDANCETYPE[coreNum*(max_length + 1)*c];
	memset(pSumFull, 0, sizeof(GUIDANCETYPE)*coreNum*(max_length + 1)*c);
	float nor_x = 1.0f / (2 * radius_x + 1), nor_y = 1.0f / (2 * radius_y + 1);
	int k, step = (int)ceil((GUIDANCETYPE)h / coreNum);

	//x direction
#pragma omp parallel for
	for (k = 0; k < coreNum; k++)
	{
		float* pSum = pSumFull + k*(max_length + 1)*c;
		for (int y = k*step; y < __min(h, (k + 1)*step); y++)
		{
			for (int n = 0; n < c; ++n)pSum[n] = 0;
			for (int x = 0; x < w; ++x)for (int n = 0; n < c; ++n)
				pSum[(x + 1)*c + n] = pSum[x*c + n] + pSrc[(y*w + x)*c + n];
			for (int x = 0; x < radius_x; ++x)for (int n = 0; n < c; ++n)
				pDst[(y*w + x)*c + n] = pSum[(x + radius_x + 1)*c + n] / (x + 1 + radius_x);
			for (int x = radius_x; x < w - radius_x - 1; ++x)for (int n = 0; n < c; ++n)
				pDst[(y*w + x)*c + n] = nor_x*(pSum[(x + radius_x + 1)*c + n] - pSum[(x - radius_x)*c + n]);
			for (int x = w - radius_x - 1; x < w; ++x) for (int n = 0; n < c; ++n)
				pDst[(y*w + x)*c + n] = (pSum[w*c + n] - pSum[(x - radius_x)*c + n]) / (w - x + radius_x);
		}
	}

	step = (int)ceil((GUIDANCETYPE)w / coreNum);

	//y direction
#pragma omp parallel for
	for (k = 0; k < coreNum; k++)
	{
		float* pSum = pSumFull + k*(max_length + 1)*c;
		for (int x = k*step; x < __min(w, (k + 1)*step); x++)
		{
			for (int n = 0; n < c; ++n)pSum[n] = 0;
			for (int y = 0; y < h; y++)for (int n = 0; n < c; ++n)
				pSum[(y + 1)*c + n] = pSum[y*c + n] + pDst[(y*w + x)*c + n];
			for (int y = 0; y < radius_y; y++)for (int n = 0; n < c; ++n)
				pDst[(y*w + x)*c + n] = pSum[(y + radius_y + 1)*c + n] / (y + 1 + radius_y);
			for (int y = radius_y; y < h - radius_y - 1; y++)for (int n = 0; n < c; ++n)
				pDst[(y*w + x)*c + n] = nor_y*(pSum[(y + radius_y + 1)*c + n] - pSum[(y - radius_y)*c + n]);
			for (int y = h - radius_y - 1; y < h; ++y)for (int n = 0; n < c; ++n)
				pDst[(y*w + x)*c + n] = (pSum[h*c + n] - pSum[(y - radius_y)*c + n]) / (h - y + radius_y);
		}
	}

	//delete[] pSumFull; pSumFull = NULL;
#endif
}

}   // namespace cv
}   // namespace deeprob
