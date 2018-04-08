/**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
 * @author Degang Yang (degang.yang@horizon-robotics.com)
 *
 * @file geodesic_filtering.h
 * @brief geodesic_filtering.h
 */

#ifndef DEEPROB_CV_STEREO_GEODESIC_FILTERING_H_
#define DEEPROB_CV_STEREO_GEODESIC_FILTERING_H_

#include <algorithm>
#include <omp.h>

namespace deeprob {
namespace cv {

template <typename T>
class GeodesicFiltering {

public:
	GeodesicFiltering();

	GeodesicFiltering(T sigma_s, T sigma_r, size_t width, size_t height,
	       float weight_vertical = 1);

	GeodesicFiltering(T sigma_s, T sigma_r, T* data, int c_data,
	        unsigned char *guidance, int c_guidance,
	        size_t width, size_t height, float weight_vertical = 1);

	void initialize(T sigma_s, T sigma_r,
            size_t width, size_t height, float weight_vertical = 1);

	~GeodesicFiltering();

	void setParameters(T sigma_s, T sigma_r);

	void setDataGuidence(T *data, int levels_data, const T *guidance,
	        int levels_guidance);

    template<typename DT>
	void runFilteringRecursive(DT *data, int levels_data,
	        const T *guidance, int levels_guidance, int iter = 1);

    template<typename DT>
	void runFilteringRecursive(DT *data, int levels_data, const T *guidance,
            int levels_guidance, unsigned char *mask_non_holes, int iter = 1);

	void runFilteringExact(int iter = 1, bool outout_normalized = true);

	void genWeightsColor();

	void genWeightsGray();

private:
	void runGeoFilterHorizontal_singleplane(T *weight_h, T *weight_h_res);

	void runGeoFilterVertical_singleplane(T *weight_v, T *weight_v_res);

public:
	T *_data;
	const T *_guidance;

private:
	bool _copy_memory;
	T *_weights_h, *_weights_v;
	T _sigma_s, _sigma_r;
	size_t _width, _height, _levels_data, _levels_guidance;
	float _weight_vertical;  //special for KITTI data sets which have many
	                         //road surfaces in the lower part of images
					         //used for local method only
};

}   // namespace cv
}   // namespace deeprob

#include "impl/geodesic_filtering.hpp"

#endif // DEEPROB_CV_STEREO_GEODESIC_FILTERING_H_
