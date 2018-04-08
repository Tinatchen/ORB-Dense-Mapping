/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file utility.h
 * @brief utility.h
 */

#ifndef DEEPROB_CV_STEREO_UTILITY_H_
#define DEEPROB_CV_STEREO_UTILITY_H_

#include <ctime>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>

// #include "filtering.h"

// #ifndef _DEBUG
// #include <omp.h>
// #endif
//
#define DEF_PADDING    (10)

namespace deeprob {
namespace cv {

inline float  i_abs(float a){ return a < 0.f ? -a : a; }
inline   int  i_abs(int a){ return a < 0 ? -a : a; }
inline double i_abs(double a){ return a < 0.0 ? -a : a; }

inline int i_round(int a){ return (a); }
inline int i_round(float a){ return((a >= 0.f) ? ((int)(a + 0.5f)) : ((int)(a - 0.5f))); }
inline int i_round(double a){ return((a >= 0.0) ? ((int)(a + 0.5)) : ((int)(a - 0.5))); }

inline double __min(double a, double b) {return (a > b) ? b : a;}
inline double __max(double a, double b) {return (a > b) ? a : b;}

template <typename T>
inline void vec_min_pos(int &min_pos, T *in, int len) {
	T min_val = in[0]; min_pos = 0;
	for (int i = 1; i < len; ++i) if (in[i] < min_val) {
		min_val = in[i]; min_pos = i;
	}
}

template<typename T>
bool saveData2D(const char* file_name, const size_t width, const size_t height,
	   const T* const data) {
	if (data == NULL){ return false; }
	FILE* fp = fopen(file_name, "wb");
	if (fp == NULL)
	{
		fprintf(stderr, "Error: fail to save data\n");
		return false;
	}

	if (fwrite(&width, sizeof(size_t), 1, fp) == EOF || fwrite(&height, sizeof(size_t), 1, fp) == EOF)
	{
		fprintf(stderr, "Error: fail to save data\n");
		return false;
	}

	if (fwrite(data, sizeof(T), width*height, fp) == EOF)
	{
		fprintf(stderr, "Error: fail to save data\n");
		return false;
	}

	fclose(fp);
	return true;
}

template<typename T>
bool loadData2D(const char* file_name, size_t& width, size_t& height, T* &data) {
	FILE* fp = fopen(file_name, "rb");
	if (!fp)
	{
		fprintf(stderr, "Error: fail to load data\n");
		return false;
	}

	if (fread((void*)&width, sizeof(size_t), 1, fp) == EOF || fread((void*)&height, sizeof(size_t), 1, fp) == EOF)
	{
		fprintf(stderr, "Error: fail to load data\n");
		return false;
	}

	if (data != NULL)
	{
		delete[] data;
		data = NULL;
	}

	data = new T[width*height];
	if (data == NULL)
	{
		fprintf(stderr, "Error: fail to alloc enough memory\n");
		return false;
	}

	if (fread((void*)data, sizeof(T), width*height, fp) == EOF)
	{
		fprintf(stderr, "Error: fail to load data\n");
		return false;
	}
	fclose(fp);
	return true;
}

inline int rgb_2_gray(const unsigned char*in) {
	int gray_val = int(0.299*in[0] + 0.587*in[1] + 0.114*in[2] + 0.5);
	return(__min(__max(gray_val, 0), 255));
}

template<typename T>
void displayincolor(unsigned char* canvas, const T *data, int width, int height,
        unsigned char bg_lum = 64) {
	float dmin = FLT_MAX;
	float dmax = FLT_MIN;
	int num_pt = width*height;
	for (int i = 0; i<num_pt; ++i)
	{
		T val = data[i];
		if (val>(T)0)
		{
			dmin = __min(dmin, val);
			dmax = __max(dmax, val);
		}
	}
	if (dmax - dmin <= (T)1e-5)
	{
		printf("Bad data range for displaying image!\n");
		return;
	}
	float scale = 1.0f / (dmax - dmin);

	for (int i = 0; i < num_pt * 3; ++i)
	{
		canvas[i] = bg_lum;
	}

	// color map
	const float map[8][4] = { { 0, 0, 0, 114 }, { 0, 0, 1, 185 }, { 1, 0, 0, 114 }, { 1, 0, 1, 174 },
	{ 0, 1, 0, 114 }, { 0, 1, 1, 185 }, { 1, 1, 0, 114 }, { 1, 1, 1, 0 } };
	float sum = 0;
	for (int i = 0; i < 8; i++) sum += map[i][3];

	float weights[8]; // relative weights
	float cumsum[8];  // cumulative weights
	cumsum[0] = 0;
	for (int i = 0; i < 7; i++)
	{
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	unsigned char max_r = 0, max_g = 0, max_b = 0;
	unsigned char min_r = 255, min_g = 255, min_b = 255;

	// for all pixels do
	for (int v = 0; v < height; v++) {
		int vw = v*width;
		for (int u = 0; u < width; u++) {
			int id = 3 * (vw + u);

			if (data[vw + u] == (T)0)
			{
				canvas[id] = bg_lum;
				canvas[id + 1] = bg_lum;
				canvas[id + 2] = bg_lum;
				continue;
			}

			// get normalized value
			float val = __min(__max(((float)(data[v*width + u]) - dmin)*scale, 0.0f), 1.0f);

			// find bin
			int i;
			for (i = 0; i < 7; i++)
			if (val < cumsum[i + 1])
				break;

			// compute rgb values
			float   w = 1.0f - (val - cumsum[i])*weights[i];

			unsigned char r = (unsigned char)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.f + 0.5f);
			unsigned char g = (unsigned char)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.f + 0.5f);
			unsigned char b = (unsigned char)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.f + 0.5f);

			// set pixel
			canvas[id] = r;//R
			canvas[id + 1] = g;//G
			canvas[id + 2] = b;//B

			max_r = __max(r, max_r);
			max_g = __max(g, max_g);
			max_b = __max(b, max_b);

			min_r = __min(r, min_r);
			min_g = __min(g, min_g);
			min_b = __min(b, min_b);
		}
	}

	float scale_r = max_r > min_r ? 255.f / ((float)max_r - (float)min_r) : 1.f;
	float scale_g = max_g > min_g ? 255.f / ((float)max_g - (float)min_g) : 1.f;
	float scale_b = max_b > min_b ? 255.f / ((float)max_b - (float)min_b) : 1.f;

	for (int v = 0; v < height; v++) {
		int vw = v*width;
		for (int u = 0; u < width; u++) {
			int id = 3 * (vw + u);
			if (canvas[id] != bg_lum &&
				canvas[id + 1] != bg_lum &&
				canvas[id + 2] != bg_lum)
			{
				canvas[id] = (unsigned char)((float)(canvas[id] - min_r) * scale_r);
				canvas[id + 1] = (unsigned char)((float)(canvas[id + 1] - min_g) * scale_g);
				canvas[id + 2] = (unsigned char)((float)(canvas[id + 2] - min_b) * scale_b);
			}
		}
	}
}

template <typename T>
inline T *** alloc_3(int n, int r, int c, int padding = DEF_PADDING) {
	unsigned char *a, **p, ***pp;
	int rc = r*c;
	int i, j;
	a = (T*)malloc(sizeof(T)*(n*rc + padding));
	if (a == NULL) { printf("alloc_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p = (T**)malloc(sizeof(T*)*n*r);
	pp = (T***)malloc(sizeof(T**)*n);
	for (i = 0; i < n; i++)
	for (j = 0; j < r; j++)
		p[i*r + j] = &a[i*rc + j*c];
	for (i = 0; i < n; i++)
		pp[i] = &p[i*r];
	return(pp);
}

template <typename T>
inline void free_3(T ***p) {
	if (p != NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p = NULL;
	}
}

template <typename T>
inline T**** alloc_4(int t, int n, int r, int c, int padding = DEF_PADDING) {
	T *a, **p, ***pp, ****ppp;
	int nrc = n*r*c, nr = n*r, rc = r*c;
	int i, j, k;
	a = (T*)malloc(sizeof(T)*(t*nrc + padding));
	if (a == NULL) { printf("alloc_4() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p = (T**)malloc(sizeof(T*)*t*nr);
	pp = (T***)malloc(sizeof(T**)*t*n);
	ppp = (T****)malloc(sizeof(T***)*t);
	for (k = 0; k < t; k++)
	for (i = 0; i < n; i++)
	for (j = 0; j < r; j++)
		p[k*nr + i*r + j] = &a[k*nrc + i*rc + j*c];
	for (k = 0; k < t; k++)
	for (i = 0; i < n; i++)
		pp[k*n + i] = &p[k*nr + i*r];
	for (k = 0; k < t; k++)
		ppp[k] = &pp[k*n];
	return(ppp);
}

template <typename T>
inline void free_4(T ****p) {
	if (p != NULL)
	{
		free(p[0][0][0]);
		free(p[0][0]);
		free(p[0]);
		free(p);
		p = NULL;
	}
}

template <typename T>
void BoxFilter(T* pSrc, T* pDst, int radius_x, int radius_y, int w, int h, int c) {
	int max_length = w;
	if (h > w)
		max_length = h;
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
			pDst[(y*w + x)*c + n] = (T)(pSum[(x + radius_x + 1)*c + n] / (float)(x + 1 + radius_x));
		for (int x = radius_x; x < w - radius_x - 1; ++x)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (T)(nor_x*(pSum[(x + radius_x + 1)*c + n] - pSum[(x - radius_x)*c + n]));
		for (int x = w - radius_x - 1; x < w; ++x) for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (T)((pSum[w*c + n] - pSum[(x - radius_x)*c + n]) / (float)(w - x + radius_x));
	}
	/*y direction*/
	for (int x = 0; x < w; x++)
	{
		for (int n = 0; n < c; ++n) pSum[n] = 0;
		for (int y = 0; y < h; y++)for (int n = 0; n < c; ++n)
			pSum[(y + 1)*c + n] = pSum[y*c + n] + (float)pDst[(y*w + x)*c + n];
		for (int y = 0; y < radius_y; y++)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (T)(pSum[(y + radius_y + 1)*c + n] / (float)(y + 1 + radius_y));
		for (int y = radius_y; y < h - radius_y - 1; y++)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (T)(nor_y*(pSum[(y + radius_y + 1)*c + n] - pSum[(y - radius_y)*c + n]));
		for (int y = h - radius_y - 1; y < h; ++y)for (int n = 0; n < c; ++n)
			pDst[(y*w + x)*c + n] = (T)((pSum[h*c + n] - pSum[(y - radius_y)*c + n]) / (float)(h - y + radius_y));
	}
	delete[]pSum;
}

template <typename T>
inline void getDataBilinearGray(float u, float v, const T* data, T& val, const int w, const int h) {
	int x = (int)u, y = (int)v;
	int idx_pt = y*w + x;
	if (x == w - 1 || y == h - 1)
	{
		val = data[idx_pt];
		return;
	}
	float Q11, Q12, Q21, Q22, val1, val2;
	float ratio_x = (u - x);
	Q11 = data[idx_pt],
		Q12 = data[idx_pt + 1],
		Q21 = data[idx_pt + w],
		Q22 = data[idx_pt + w + 1];
	val1 = Q11 + (Q12 - Q11)*ratio_x;
	val2 = Q21 + (Q22 - Q21)*ratio_x;
	val = i_round(val1 + (val2 - val1)*(v - y));
	if (val > 255) val = 255;
}

template <typename T>
void downsamplingImage(T* image_high, T* image_low, int nr_levels, int width_high, int height_high, int width_low, int height_low, int factor) {
    #pragma omp parallel for
	for (int y = 0; y < height_low; ++y)
	{
		for (int x = 0; x < width_low; ++x)
		{
			int idx_pt_low = y*width_low + x;
			int y_h = factor*y, x_h = factor*x;
			if (y_h>height_high - 1 || x_h>width_high - 1)
			{
				for (int c = 0; c < nr_levels; ++c)image_low[idx_pt_low*nr_levels + c] = 0;
				continue;
			}
			int idx_pt_high = y_h*width_high + x_h;
			for (int c = 0; c < nr_levels; ++c)image_low[idx_pt_low*nr_levels + c] = image_high[idx_pt_high*nr_levels + c];
		}
	}
}

template <typename T, typename S>
void getDepthFromDisparity(S* disparity, T* depth, int width, int height, T len_baseline/*baseline length*/, T f/*focal length of stereo rig, in pixels*/) {
    #pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			int pixelIndex = y*width + x;
			depth[pixelIndex] = (T)(len_baseline*f/(T)(disparity[pixelIndex]));
		}
	}
}

template <typename T>
inline void fillData_sse_16char(char data[16], T* tar, int idx_pt, int x, int nr_planes, int &d) {
    if (x <= nr_planes)
    {
        int idx_row_start = idx_pt - x;
        unsigned char val_row_start = tar[idx_row_start];
        data[0] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[1] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[2] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[3] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[4] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[5] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[6] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[7] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[8] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[9] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[10] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[11] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[12] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[13] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[14] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
        data[15] = (x - d >= 0 ? (char)tar[idx_pt - d] : val_row_start); d++;
    }
    else
    {
        //data[0] = tar[idx_pt - d++];
        //data[1] = tar[idx_pt - d++];
        //data[2] = tar[idx_pt - d++];
        //data[3] = tar[idx_pt - d++];
        //data[4] = tar[idx_pt - d++];
        //data[5] = tar[idx_pt - d++];
        //data[6] = tar[idx_pt - d++];
        //data[7] = tar[idx_pt - d++];
        //data[8] = tar[idx_pt - d++];
        //data[9] = tar[idx_pt - d++];
        //data[10] = tar[idx_pt - d++];
        //data[11] = tar[idx_pt - d++];
        //data[12] = tar[idx_pt - d++];
        //data[13] = tar[idx_pt - d++];
        //data[14] = tar[idx_pt - d++];
        //data[15] = tar[idx_pt - d++];

        int i = idx_pt - d;
        data[0] = tar[i];
        data[1] = tar[--i];
        data[2] = tar[--i];
        data[3] = tar[--i];
        data[4] = tar[--i];
        data[5] = tar[--i];
        data[6] = tar[--i];
        data[7] = tar[--i];
        data[8] = tar[--i];
        data[9] = tar[--i];
        data[10] = tar[--i];
        data[11] = tar[--i];
        data[12] = tar[--i];
        data[13] = tar[--i];
        data[14] = tar[--i];
        data[15] = tar[--i];
        d += 16;
    }
}

}   // namespace cv
}   // namespace deeprob

#endif // DEEPROB_CV_STEREO_UTILITY_H_
