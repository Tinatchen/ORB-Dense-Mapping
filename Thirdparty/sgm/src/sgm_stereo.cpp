/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file sgm_stereo.cpp
 * @brief sgm_stereo.cpp
 */

#include "sgm_stereo.h"

#include <emmintrin.h>
#include <stack>
#include <algorithm>
#include <limits>

namespace deeprob {
namespace cv {

void SGMStereo::performSGM(COSTTYPE_* costImage, unsigned char* disparityImage, int width, int height,
	int nr_planes, int smoothnessPenaltySmall_, int smoothnessPenaltyLarge_)
{
	int width_ = width, height_ = height, disparityTotal_ = nr_planes;

	/*default*/
	const int pathRowBufferTotal_ = 2;
	const int disparitySize_ = disparityTotal_ + 16;
	const int pathTotal_ = 8;
	const int pathDisparitySize_ = pathTotal_*disparitySize_;

	int costSumBufferRowSize_ = width_*disparityTotal_;
	int costSumBufferSize_ = costSumBufferRowSize_*height_;
	int pathMinCostBufferSize_ = (width_ + 2)*pathTotal_;
	int pathCostBufferSize_ = pathMinCostBufferSize_*disparitySize_;
	int totalBufferSize_ = (pathMinCostBufferSize_ + pathCostBufferSize_)*pathRowBufferTotal_ + costSumBufferSize_ + 16;
	short* sgmBuffer_ = reinterpret_cast<short*>(_mm_malloc(totalBufferSize_*sizeof(short), 16));

	const short costMax = std::numeric_limits<short>::max();
	int widthStepCostImage = width_*disparityTotal_;

	short* costSums = sgmBuffer_;
	memset(costSums, 0, costSumBufferSize_*sizeof(short));/*save the updated cost*/

	short** pathCosts = new short*[pathRowBufferTotal_];
	short** pathMinCosts = new short*[pathRowBufferTotal_];

	const int processPassTotal = 2;
	for (int processPassCount = 0; processPassCount < processPassTotal; ++processPassCount)
	{
		int startX, endX, stepX;
		int startY, endY, stepY;
		if (processPassCount == 0) /*first pass: from top-left to bottom-right*/
		{
			startX = 0; endX = width_; stepX = 1;
			startY = 0; endY = height_; stepY = 1;
		}
		else /*second pass: from bottom-right to top-left*/
		{
			startX = width_ - 1; endX = -1; stepX = -1;
			startY = height_ - 1; endY = -1; stepY = -1;
		}

		for (int i = 0; i < pathRowBufferTotal_; ++i)
		{
			/*
			0 - current scanline
			1 - previous scanline
			*/
			pathCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*i + pathDisparitySize_ + 8;
			memset(pathCosts[i] - pathDisparitySize_ - 8, 0, pathCostBufferSize_*sizeof(short));
			pathMinCosts[i] = costSums + costSumBufferSize_ + pathCostBufferSize_*pathRowBufferTotal_ + pathMinCostBufferSize_*i + pathTotal_ * 2;
			memset(pathMinCosts[i] - pathTotal_, 0, pathMinCostBufferSize_*sizeof(short));
		}

		for (int y = startY; y != endY; y += stepY)
		{
			unsigned short* pixelCostRow = costImage + widthStepCostImage*y;
			short* costSumRow = costSums + costSumBufferRowSize_*y;

			/*re-set the auxilliary memory*/
			memset(pathCosts[0] - pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
			memset(pathCosts[0] + width_*pathDisparitySize_ - 8, 0, pathDisparitySize_*sizeof(short));
			memset(pathMinCosts[0] - pathTotal_, 0, pathTotal_*sizeof(short));
			memset(pathMinCosts[0] + width_*pathTotal_, 0, pathTotal_*sizeof(short));

			for (int x = startX; x != endX; x += stepX)
			{
				int pathMinX = x*pathTotal_;
				int pathX = pathMinX*disparitySize_;
				/*
				0->horizontal
				1->vertical
				*/
				int previousPathMin0 = pathMinCosts[0][pathMinX - stepX*pathTotal_] + smoothnessPenaltyLarge_;
				int previousPathMin2 = pathMinCosts[1][pathMinX + 2] + smoothnessPenaltyLarge_;/*begin with direction 0! hence + 2 corresponds to 1*/

				short* previousPathCosts0 = pathCosts[0] + pathX - stepX*pathDisparitySize_;
				short* previousPathCosts2 = pathCosts[1] + pathX + disparitySize_ * 2;

				previousPathCosts0[-1] = previousPathCosts0[disparityTotal_] = costMax;
				previousPathCosts2[-1] = previousPathCosts2[disparityTotal_] = costMax;

				short* pathCostCurrent = pathCosts[0] + pathX;
				const unsigned short* pixelCostCurrent = pixelCostRow + disparityTotal_*x;
				short* costSumCurrent = costSumRow + disparityTotal_*x;

				__m128i regPenaltySmall = _mm_set1_epi16(static_cast<short>(smoothnessPenaltySmall_));
				__m128i regPathMin0 = _mm_set1_epi16(static_cast<short>(previousPathMin0));
				__m128i regPathMin2 = _mm_set1_epi16(static_cast<short>(previousPathMin2));
				__m128i regNewPathMin = _mm_set1_epi16(costMax);

				for (int d = 0; d < disparityTotal_; d += 8)
				{
					__m128i regPixelCost = _mm_load_si128(reinterpret_cast<const __m128i*>(pixelCostCurrent + d));
					__m128i regPathCost0 = _mm_load_si128(reinterpret_cast<const __m128i*>(previousPathCosts0 + d));
					__m128i regPathCost2 = _mm_load_si128(reinterpret_cast<const __m128i*>(previousPathCosts2 + d));

					regPathCost0 = _mm_min_epi16(regPathCost0,
						_mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts0 + d - 1)), regPenaltySmall));
					regPathCost0 = _mm_min_epi16(regPathCost0,
						_mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts0 + d + 1)), regPenaltySmall));
					regPathCost2 = _mm_min_epi16(regPathCost2,
						_mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts2 + d - 1)), regPenaltySmall));
					regPathCost2 = _mm_min_epi16(regPathCost2,
						_mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(previousPathCosts2 + d + 1)), regPenaltySmall));

					regPathCost0 = _mm_min_epi16(regPathCost0, regPathMin0);
					regPathCost0 = _mm_adds_epi16(_mm_subs_epi16(regPathCost0, regPathMin0), regPixelCost);
					regPathCost2 = _mm_min_epi16(regPathCost2, regPathMin2);
					regPathCost2 = _mm_adds_epi16(_mm_subs_epi16(regPathCost2, regPathMin2), regPixelCost);

					_mm_store_si128(reinterpret_cast<__m128i*>(pathCostCurrent + d), regPathCost0);
					_mm_store_si128(reinterpret_cast<__m128i*>(pathCostCurrent + d + disparitySize_ * 2), regPathCost2);/*packed in format: path 0, blank, path 1, ...*/

					__m128i regCostSum = _mm_load_si128(reinterpret_cast<const __m128i*>(costSumCurrent + d));
					regCostSum = _mm_adds_epi16(regCostSum, regPathCost0);
					regCostSum = _mm_adds_epi16(regCostSum, regPathCost2);
					_mm_store_si128(reinterpret_cast<__m128i*>(costSumCurrent + d), regCostSum);

					/*calculate the minimum of path cost with every 8 disparities for scanline 0*/
					__m128i regMin02 = _mm_min_epi16(_mm_unpacklo_epi16(regPathCost0, regPathCost2), _mm_unpackhi_epi16(regPathCost0, regPathCost2));
					regMin02 = _mm_min_epi16(_mm_unpacklo_epi16(regMin02, regMin02), _mm_unpackhi_epi16(regMin02, regMin02));
					regNewPathMin = _mm_min_epi16(regNewPathMin, regMin02);
				}

				/*update the minimum of path cost for scanline 0*/
				regNewPathMin = _mm_min_epi16(regNewPathMin, _mm_srli_si128(regNewPathMin, 8));
				_mm_storel_epi64(reinterpret_cast<__m128i*>(&pathMinCosts[0][pathMinX]), regNewPathMin);
			}

			if (processPassCount == processPassTotal - 1)/*calculate the disparity map results using the aggregated path costs*/
			{
				unsigned char* disparityRow = disparityImage + width_*y;

				for (int x = 0; x < width_; ++x)
				{
					short* costSumCurrent = costSumRow + disparityTotal_*x;
					int bestSumCost = costSumCurrent[0];
					int bestDisparity = 0;
					for (int d = 1; d < disparityTotal_; ++d)
					{
						if (costSumCurrent[d] < bestSumCost)
						{
							bestSumCost = costSumCurrent[d];
							bestDisparity = d;
						}
					}

					if (bestDisparity > 0 && bestDisparity < disparityTotal_ - 1)/*quadratic interpolation*/
					{
						int centerCostValue = costSumCurrent[bestDisparity];
						int leftCostValue = costSumCurrent[bestDisparity - 1];
						int rightCostValue = costSumCurrent[bestDisparity + 1];
						int delta = (int)(static_cast<double>(rightCostValue - leftCostValue) / (2.0 * (leftCostValue + rightCostValue - 2.0 * centerCostValue)) + 0.5);
						if (i_abs(delta)<=1) bestDisparity = static_cast<int>(bestDisparity - delta);
					}
					else
					{
						bestDisparity = static_cast<int>(bestDisparity);
					}

					disparityRow[x] = static_cast<unsigned short>(bestDisparity);
				}
			}

			/*0->1 for the next scanline*/
			std::swap(pathCosts[0], pathCosts[1]);
			std::swap(pathMinCosts[0], pathMinCosts[1]);
		}
	}
	delete[] pathCosts;
	delete[] pathMinCosts;
	_mm_free(sgmBuffer_);
}

void SGMStereo::calDisparity()
{
	//step 1. matching cost initialization
	calMatchingCost(_left_input, _right_input, _cost_vol, _nr_planes, _width, _height);
	boxfilter(_cost_vol[0][0], _cost_vol_backup[0][0], RADIUS_BOXFILTER + 1, RADIUS_BOXFILTER, _width, _height, _nr_planes);/*box filtering cost aggregation*/
	stereo_flip_cost_vol(_cost_vol_right, _cost_vol_backup, _height, _width, _nr_planes);/*flip the cost volume from left to right*/

	//step 2. semi-global matcher on both views
	performSGM(_cost_vol_backup[0][0], _disparity[0], _width, _height, _nr_planes);
	speckleFilter(_disparity[0], _width, _height);
	//memcpy(_disparity_backup[0], _disparity[0], sizeof(unsigned char)*_width*_height);
	//ctmf(_disparity_backup[0], _disparity[0], _width, _height, _width, _width, RADIUS_MEDIAN + 1, 1, _height*_width);

	performSGM(_cost_vol_right[0][0], _disparity_r[0], _width, _height, _nr_planes);
	speckleFilter(_disparity_r[0], _width, _height);
	//memcpy(_disparity_backup[0], _disparity_r[0], sizeof(unsigned char)*_width*_height);
	//ctmf(_disparity_backup[0], _disparity_r[0], _width, _height, _width, _width, RADIUS_MEDIAN + 1, 1, _height*_width);

	genProposalLR(_disparity, _disparity_r);/*cross-check to generate a disparity proposal*/

	//step 3. disparity re-propagation based post processing
	/*first pass: fix small noises in 3*3 neighborhood*/
	buildNewVol(_disparity, _nr_planes, _cost_vol_backup);
	boxfilter(_cost_vol_backup[0][0], _cost_vol[0][0], RADIUS_BOXFILTER, RADIUS_BOXFILTER, _width, _height, _nr_planes);
	wta(_disparity, _cost_vol);
#pragma omp parallel for
	for (int y_ = 0; y_ < _height; ++y_)
	{
		for (int x_ = 0; x_ < _width; ++x_)
		{
			int bestDisparity = _disparity[y_][x_];
			if (bestDisparity>1 && bestDisparity < _nr_planes - 1)/*sub-pixel interpolation*/
			{
				int centerCostValue = _cost_vol[y_][x_][bestDisparity];
				int leftCostValue = _cost_vol[y_][x_][bestDisparity - 1];
				int rightCostValue = _cost_vol[y_][x_][bestDisparity + 1];
				int delta_d = (int)(static_cast<double>(rightCostValue - leftCostValue) / (2 * (leftCostValue + rightCostValue - 2 * centerCostValue)) + 0.5);
				if (i_abs(delta_d) <= 1)  _disparity[y_][x_] = static_cast<int>(bestDisparity - delta_d);
			}
		}
	}

	/*second pass: reliable disparity propagation for remaining holes*/
	memcpy(_disparity_backup[0], _disparity[0], sizeof(unsigned char)*_width*_height);
	fillHoles(_disparity[0], _width, _height);
	buildNewVol(_disparity, _nr_planes, _cost_vol);
	//boxfilter(_cost_vol_backup[0][0], _cost_vol[0][0], RADIUS_BOXFILTER, RADIUS_BOXFILTER, _width, _height, _nr_planes);

	_gf.runFilteringRecursive<COSTTYPE_>(_cost_vol[0][0], _nr_planes, _guidance_left, _c_guidance, 1);
	wta(_disparity, _cost_vol);
#pragma omp parallel for
	for (int y_ = 0; y_ < _height; ++y_)
	{
		for (int x_ = 0; x_ < _width; ++x_)
		{
			if (_disparity_backup[y_][x_]>0)
			{
				_disparity[y_][x_] = _disparity_backup[y_][x_];
				continue;
			}

			int bestDisparity = _disparity[y_][x_];
			if (bestDisparity>1 && bestDisparity < _nr_planes - 1)/*sub-pixel interpolation*/
			{
				int centerCostValue = _cost_vol[y_][x_][bestDisparity];
				int leftCostValue = _cost_vol[y_][x_][bestDisparity - 1];
				int rightCostValue = _cost_vol[y_][x_][bestDisparity + 1];
				int delta_d = (int)(static_cast<double>(rightCostValue - leftCostValue) / (2 * (leftCostValue + rightCostValue - 2 * centerCostValue)) + 0.5);
				if (i_abs(delta_d) <= 1)  _disparity[y_][x_] = static_cast<int>(bestDisparity - delta_d);
			}
		}
	}
	//imdebug("lum *auto b=8 w=%d h=%d %p", _width, _height, _disparity[0]);

	fillHoles(_disparity[0], _width, _height);
	memcpy(_disparity_backup[0], _disparity[0], sizeof(unsigned char)*_width*_height);
	ctmf(_disparity_backup[0], _disparity[0], _width, _height, _width, _width, RADIUS_MEDIAN, 1, _height*_width);
}
void SGMStereo::genProposalLR(DISPARITYTYPE_** disparity_left, DISPARITYTYPE_** disparity_right)
{
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

	//	int  step = 2;
	//#pragma omp parallel for
	//	for (int y = 0; y < _height; ++y) for (int x = 0; x < _width; ++x)
	//	{
	//		//mask out occluded region detected asymmetrically
	//		int d = disparity_left[y][x], x_end = __min(_width - 1, x + _nr_planes);
	//		if (d == 0) continue;
	//		for (int x_ = x + 1; x_ <= x_end; x_ += step) // sparsely sampled
	//		{
	//			if (x_ + d <= x + disparity_left[y][x_])
	//			{
	//				disparity_left[y][x] = 0;
	//				break;
	//			}
	//		}
	//	}
}

void SGMStereo::buildNewVol(DISPARITYTYPE_** disparity, int nr_planes, COSTTYPE_***cost_vol)
{
	int delta_disparity = __min(DELTA_DISPARITY_RATIO*_nr_planes, DELTA_DISPARITY_MAXIMUM);

#pragma omp parallel for
	for (int y = 0; y < _height; y++)
	{
		for (int x = 0; x < _width; x++)
		{
			int d = disparity[y][x]; /*disparity values from left view (reference)*/
			if (d>0)
			{
				for (int d_ = 0; d_ < _nr_planes; ++d_)
				{
					COSTTYPE_ delta = __min(i_abs(d - d_), delta_disparity);
					cost_vol[y][x][d_] = delta*delta;
				}
			}
			else
			{
				memset(cost_vol[y][x], 0, sizeof(COSTTYPE_)*_nr_planes);
			}
		}
	}
}
void SGMStereo::calMatchingCost(const unsigned char* left, const unsigned char*right, COSTTYPE_***cost_vol, int nr_planes, int width, int height)
{
# if 0
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

#pragma omp parallel for
	for (int y = 0; y < height; ++y){
		for (int x = 0; x < width; ++x){
			int idx_pt = y*width + x;
			for (int i = 0; i < nr_planes; ++i)
			{
				if (x >= i)
				{
					COSTTYPE_ cost_gradient = (COSTTYPE_)i_abs(gradient_left[idx_pt] - gradient_right[idx_pt - i]);
					COSTTYPE_ cost_census = (COSTTYPE_)calHammingDistance(census_left[idx_pt], census_right[idx_pt - i]);
					if (cost_census>THRES_CENSUS) cost_census = THRES_CENSUS;
					cost_vol[y][x][i] = (COSTTYPE_)(cost_census/2 + cost_gradient);
				}
				else
				{
					cost_vol[y][x][i] = cost_vol[y][x][i - 1];
				}
			}
		}
	}

#else
	/*Birchfield and Tomasi(BT) cost measure on gradient*/
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

	int size = width * height;
	int* buffer = new int[size * 4];
	memset(buffer, 0, sizeof(int)*size * 4);
	int* min_left = buffer, *max_left = buffer + size;
	int* min_right = buffer + 2 * size, *max_right = buffer + 3 * size;

#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int I_minus, I_plus, idx_pt = y*width + x;

			//left
			I_minus = x >= 1 ? ((gradient_left[idx_pt - 1] + gradient_left[idx_pt]) / 2) : gradient_left[idx_pt];
			I_plus = (x < width - 1) ? ((gradient_left[idx_pt + 1] + gradient_left[idx_pt]) / 2) : gradient_left[idx_pt];

			min_left[idx_pt] = __min(__min(I_minus, I_plus), gradient_left[idx_pt]);
			max_left[idx_pt] = __max(__max(I_minus, I_plus), gradient_left[idx_pt]);

			//right
			I_minus = x >= 1 ? ((gradient_right[idx_pt - 1] + gradient_right[idx_pt]) / 2) : gradient_right[idx_pt];
			I_plus = (x < width - 1) ? ((gradient_right[idx_pt + 1] + gradient_right[idx_pt]) / 2) : gradient_right[idx_pt];

			min_right[idx_pt] = __min(__min(I_minus, I_plus), gradient_right[idx_pt]);
			max_right[idx_pt] = __max(__max(I_minus, I_plus), gradient_right[idx_pt]);
		}
	}


#pragma omp parallel for
	for (int y = 0; y < height; ++y){
		for (int x = 0; x < width; ++x){
			int idx_pt = y*width + x;
			for (int i = 0; i < nr_planes; ++i)
			{
				if (x >= i)
				{
					COSTTYPE_ A = __max(__max(gradient_left[idx_pt] - max_right[idx_pt - i], min_right[idx_pt - i] - gradient_left[idx_pt]), 0);
					COSTTYPE_ B = __max(__max(gradient_right[idx_pt - i] - max_left[idx_pt], min_left[idx_pt] - gradient_right[idx_pt - i]), 0);
					COSTTYPE_ cost_gradient = __min(A, B); /*BT cost measure*/
					COSTTYPE_ cost_census = (COSTTYPE_)calHammingDistance(census_left[idx_pt], census_right[idx_pt - i]);
					if (cost_census>THRES_CENSUS) cost_census = THRES_CENSUS;
					cost_vol[y][x][i] = (COSTTYPE_)(cost_census / 2 + cost_gradient);
				}
				else
				{
					cost_vol[y][x][i] = cost_vol[y][x][i - 1];
				}
			}
		}
	}

	delete[]buffer;
#endif

}
void SGMStereo::stereo_flip_cost_vol(COSTTYPE_***&cost_vol_right, COSTTYPE_***cost_vol, int height, int width, int nr_planes)
{
#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width - nr_planes; x++) for (int d = 0; d < nr_planes; d++) cost_vol_right[y][x][d] = cost_vol[y][x + d][d];
		for (int x = width - nr_planes; x < width; x++) for (int d = 0; d < nr_planes; d++)
		{
			if ((x + d) < width)
			{
				cost_vol_right[y][x][d] = cost_vol[y][x + d][d];
			}
			else
			{
				cost_vol_right[y][x][d] = cost_vol_right[y][x][d - 1];
			}
		}
	}
}
void SGMStereo::calGradient_sobel(int*gradient, unsigned char*gray, int width, int height)
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
			gradient[width*y + x] = (sobelValue>15 ? 30 : (sobelValue < -15 ? 0 : sobelValue + 15));//using the same mapping range as Yamaguchi
		}
	}
}
void SGMStereo::wta(DISPARITYTYPE_** disparity, COSTTYPE_*** cost_vol)
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
void SGMStereo::speckleFilter(unsigned char* image, int width, int height, const int maxSpeckleSize, const int maxDifference)
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
void SGMStereo::fillHoles(unsigned char* disparity, int width, int height)
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
void SGMStereo::boxfilter(COSTTYPE_* pSrc, COSTTYPE_* pDst, int radius_x, int radius_y, int w, int h, int c)
{

#if 0 /*baseline version*/
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
			pDst[(y*w + x)*c + n] = (COSTTYPE_)(pSum[(x + radius_x + 1)*c + n] / (float)(x + 1 + radius_x));
		for (int x = radius_x; x < w - radius_x - 1; ++x)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE_)(nor_x*(pSum[(x + radius_x + 1)*c + n] - pSum[(x - radius_x)*c + n]));
		for (int x = w - radius_x - 1; x < w; ++x) for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE_)((pSum[w*c + n] - pSum[(x - radius_x)*c + n]) / (float)(w - x + radius_x));
	}
	/*y direction*/
	for (int x = 0; x < w; x++)
	{
		for (int n = 0; n < c; ++n) pSum[n] = 0;
		for (int y = 0; y < h; y++)for (int n = 0; n < c; ++n)
			pSum[(y + 1)*c + n] = pSum[y*c + n] + (float)pDst[(y*w + x)*c + n];
		for (int y = 0; y < radius_y; y++)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE_)(pSum[(y + radius_y + 1)*c + n] / (float)(y + 1 + radius_y));
		for (int y = radius_y; y < h - radius_y - 1; y++)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE_)(nor_y*(pSum[(y + radius_y + 1)*c + n] - pSum[(y - radius_y)*c + n]));
		for (int y = h - radius_y - 1; y < h; ++y)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (COSTTYPE_)((pSum[h*c + n] - pSum[(y - radius_y)*c + n]) / (float)(h - y + radius_y));
	}

	delete[]pSum;

#else
	/*openmp optimized version*/
	int max_length = w;
	if (h > w) max_length = h;
	int coreNum = omp_get_num_procs();   /*get the number of processors*/
	GUIDANCETYPE_* pSumFull = new GUIDANCETYPE_[coreNum*(max_length + 1)*c];
	memset(pSumFull, 0, sizeof(GUIDANCETYPE_)*coreNum*(max_length + 1)*c);
	float nor_x = 1.0f / (2 * radius_x + 1), nor_y = 1.0f / (2 * radius_y + 1);
	int k, step = (int)ceil((GUIDANCETYPE_)h / coreNum);

	/*x direction*/
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

	step = (int)ceil((GUIDANCETYPE_)w / coreNum);

	/*y direction*/
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
	delete[] pSumFull; pSumFull = NULL;
#endif
}

}   // namespace cv
}   // namespace deeprob
