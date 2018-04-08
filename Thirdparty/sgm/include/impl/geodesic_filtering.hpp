 /**@copyright 2016 Horizon-Robotics Inc. All Rights Reserved.
  * @author Degang Yang (degang.yang@horizon-robotics.com)
  *
  * @file geodesic_filtering.hpp
  * @brief The implementation of the matrix.
  */

#include "utility.h"

namespace deeprob {
namespace cv {

template <typename T>
GeodesicFiltering<T>::GeodesicFiltering() {
	_sigma_s = 0;
	_sigma_r = 0;
	_width = 0;
	_height = 0;
	_copy_memory = false;
    _data = NULL;
	_guidance = NULL;
	_weights_h = NULL;
	_weights_v = NULL;
	_weight_vertical = 1;
}

template <typename T>
GeodesicFiltering<T>::GeodesicFiltering(T sigma_s, T sigma_r,
        size_t width, size_t height, float weight_vertical) {
	_sigma_s = sigma_s;
	_sigma_r = sigma_r;
	_width = width;
	_height = height;
	_weight_vertical = weight_vertical;

    // _levels_data = c_data;
    // _levels_guidance = c_guidance;

	_copy_memory = false;
	_guidance = _data = NULL;

	int e_num_h = height * (width - 1);
	int e_num_v = width * (height - 1);
	_weights_h = new T[e_num_h];
    memset(_weights_h, 0, sizeof(T) * e_num_h);
	_weights_v = new T[e_num_v];
    memset(_weights_v, 0, sizeof(T) * e_num_v);
}

template <typename T>
GeodesicFiltering<T>::GeodesicFiltering(T sigma_s, T sigma_r,
	    T *data, int c_data, unsigned char *guidance, int  c_guidance,
    	size_t width, size_t height, float weight_vertical) {
	_sigma_s = sigma_s;
	_sigma_r = sigma_r;
	_width = width;
	_height = height;
	_weight_vertical = weight_vertical;

	_levels_data = c_data;
	_levels_guidance = c_guidance;

	_copy_memory = true;
	_guidance = new T[_width * _height * c_guidance];
	_data = new T[_width * _height * c_data];
	for (int y = 0; y < _height; ++y) {
        for (int x = 0; x < _width; ++x) {
		    int idx_pts = y * _width + x;
            int idx_pts_c = c_guidance * idx_pts;
		    for (int c = 0; c < c_guidance; ++c) {
                _guidance[idx_pts_c + c] = (T)guidance[idx_pts_c + c];
		    }
            idx_pts_c = c_data * idx_pts;
            for (int c = 0; c < c_data; ++c) {
                _data[idx_pts_c + c] = (T)data[idx_pts_c + c];
            }
        }
    }
	int e_num_h = height * (width - 1);
	int e_num_v = width * (height - 1);
	_weights_h = new T[e_num_h];
    memset(_weights_h, 0, sizeof(T) * e_num_h);
	_weights_v = new T[e_num_v];
    memset(_weights_v, 0, sizeof(T) * e_num_v);
}

template <typename T>
void GeodesicFiltering<T>::initialize(T sigma_s, T sigma_r,
        size_t width, size_t height, float weight_vertical) {
    _sigma_s = sigma_s;
    _sigma_r = sigma_r;
    _width = width;
    _height = height;
    _weight_vertical = weight_vertical;
    _copy_memory = false;
    _guidance = _data = NULL;

    int e_num_h = height * (width - 1);
    int e_num_v = width * (height - 1);
    if (_weights_h != NULL) {
        delete []_weights_h;
        _weights_h = NULL;
    }
    if (_weights_v != NULL) {
        delete []_weights_v;
        _weights_v = NULL;
    }
    _weights_h = new T[e_num_h];
    memset(_weights_h, 0, sizeof(T) * e_num_h);
    _weights_v = new T[e_num_v];
    memset(_weights_v, 0, sizeof(T) * e_num_v);
}

template <typename T>
GeodesicFiltering<T>::~GeodesicFiltering() {
    if (_copy_memory && _guidance != NULL) {
        delete []_guidance;
        _guidance = NULL;
    }
    if (_copy_memory && _data != NULL) {
        delete []_data;
        _data = NULL;
    }
    if (_weights_h != NULL) {
        delete []_weights_h;
        _weights_h = NULL; }
    if (_weights_v != NULL) {
        delete []_weights_v;
        _weights_v = NULL;
    }
}

template <typename T>
void GeodesicFiltering<T>::setParameters(T sigma_s, T sigma_r) {
    _sigma_s = sigma_s;
    _sigma_r = sigma_r;
}

template <typename T>
void GeodesicFiltering<T>::setDataGuidence(T *data, int levels_data,
        const T *guidance, int levels_guidance) {
    _data = data;
    _levels_data = levels_data;
    _guidance = guidance;
    _levels_guidance = levels_guidance;
    _copy_memory = false;
}

template <typename T>
void GeodesicFiltering<T>::genWeightsColor() {
    const T *p_ref = _guidance;
    T sigma_s = _sigma_s, sigma_r = _sigma_r;
    T sigma = sigma_s / sigma_r;
    T sigma_h = sigma_s;
    int w = _width, h = _height;
    T lut[256];

#pragma omp parallel for
    for (int i = 0; i <= 255; ++i) {
        lut[i] = exp(-((T)i * sigma + (T)1) / sigma_h);
    }

    //weights along horizontal directions
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < (w - 1); ++x) {
            int idx_c = 3 * (y * w + x);
            int idx_n = idx_c + 3;
            int diff_0 = i_abs(p_ref[idx_c++] - p_ref[idx_n++]);
            int diff_1 = i_abs(p_ref[idx_c++] - p_ref[idx_n++]);
            int diff_2 = i_abs(p_ref[idx_c] - p_ref[idx_n]);
            if (diff_1 > diff_0) {
                diff_0 = diff_1;
            }
            if (diff_2 > diff_0) {
                diff_0 = diff_2;
            }
            int idx_h = y * (w - 1) + x;
            _weights_h[idx_h] = lut[diff_0];
        }
    }

    //weights along vertical directions
#pragma omp parallel for
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < (h - 1); ++y) {
            int idx_c = 3 * (y * w + x);
            int idx_n = idx_c + 3 * w;
            int diff_0 = i_abs(p_ref[idx_c++] - p_ref[idx_n++]);
            int diff_1 = i_abs(p_ref[idx_c++] - p_ref[idx_n++]);
            int diff_2 = i_abs(p_ref[idx_c] - p_ref[idx_n]);
            if (diff_1 > diff_0) {
                diff_0 = diff_1;
            }
            if (diff_2 > diff_0) {
                diff_0 = diff_2;
            }
            int idx_v = x * (h - 1) + y;
            _weights_v[idx_v] = ((2 * y) > h ? _weight_vertical : 1) * lut[diff_0];
        }
    }
}

template <typename T>
void GeodesicFiltering<T>::genWeightsGray() {
    const T *p_ref = _guidance;
    T sigma_s = _sigma_s, sigma_r = _sigma_r;
    T sigma = sigma_s / sigma_r;
    T sigma_h = sigma_s;
    int w = _width, h = _height;
    T lut[256];

#pragma omp parallel for
    for (int i = 0; i <= 255; ++i) {
        lut[i] = exp(-((T)i * sigma + (T)1) / sigma_h);
    }

    //weights along horizontal directions
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < (w - 1); ++x) {
            int idx_1 = (y * w + x);
            int idx_2 = idx_1 + 1;
            int idx_h = y * (w - 1) + x;
            _weights_h[idx_h] = lut[(int)i_abs(p_ref[idx_1] - p_ref[idx_2])];
        }
    }

    //weights along vertical directions
#pragma omp parallel for
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < (h - 1); ++y) {
            int idx_1 = (y * w + x);
            int idx_2 = idx_1 + w;
            int idx_v = x * (h - 1) + y;
            _weights_v[idx_v] = ((2 * y) > h ? _weight_vertical : 1)
                             * lut[(int)i_abs(p_ref[idx_1] - p_ref[idx_2])];
        }
    }
}

template <typename T>
void GeodesicFiltering<T>::runGeoFilterHorizontal_singleplane(T *weight_h,
        T *weight_h_res) {
    int w = _width, h = _height;

#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        //1st pass: from left to right
        for (int x = 1; x < w; ++x) {
            int idx_pt = y * w + x;
            int idx_e = y * w - y + x - 1;
            weight_h_res[idx_pt] = weight_h_res[idx_pt]
                                 + weight_h[idx_e] * weight_h_res[idx_pt - 1];
        }

        //2nd pass: from right to left
        for (int x = w - 2; x >= 0; --x) {
            int idx_pt = y * w + x;
            int idx_e = y * w - y + x;
            T inv_w_e2 = (T)1 - weight_h[idx_e] * weight_h[idx_e];
            weight_h_res[idx_pt] = inv_w_e2 * weight_h_res[idx_pt]
                                 + weight_h[idx_e] * weight_h_res[idx_pt + 1];
        }
    }
}

template <typename T>
void GeodesicFiltering<T>::runGeoFilterVertical_singleplane(T *weight_v,
        T *weight_v_res) {
    int w = _width, h = _height;

#pragma omp parallel for
    for (int x = 0; x < w; ++x) {
        //1st pass: from top to bottom
        for (int y = 1; y < h; ++y) {
            int idx_pt = y * w + x;
            int idx_e = x * h - x + y - 1;
            weight_v_res[idx_pt] = weight_v_res[idx_pt]
                                 + weight_v[idx_e] * weight_v_res[idx_pt - w];
        }

        //2nd pass: from bottom to top
        for (int y = h - 2; y >= 0; --y) {
            int idx_pt = y * w + x;
            int idx_e = x * h - x + y;
            T inv_w_e2 = (T)1 - weight_v[idx_e] * weight_v[idx_e];
            weight_v_res[idx_pt] = inv_w_e2 * weight_v_res[idx_pt]
                                 + weight_v[idx_e] * weight_v_res[idx_pt + w];
        }
    }
}

template <typename T>
template<typename DT>
void GeodesicFiltering<T>::runFilteringRecursive(DT *data, int levels_data,
        const T *guidance, int levels_guidance, int iter) {
    int nplane = levels_data, h = _height, w = _width;
    DT *p_data = data;
    _guidance = guidance;
    if (levels_guidance == 3) {
        genWeightsColor();
    } else {
        genWeightsGray();
    }
    _copy_memory = false;

    while (iter > 0) {
        --iter;

        //horizontal
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            //from left to right
            for (int x = 1; x < w; ++x) {
                int idx_pt = y * w + x;
                int idx_e = idx_pt - y - 1;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_h[idx_e];
                T w_e_inv = (T)1 - w_e;
                DT* p_data_cur = &p_data[idx_pt_tem];
                DT* p_data_pre = &p_data[idx_pt_tem - nplane];
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] =(DT)(w_e_inv * p_data[idx_pt_tem + i]
                                         + w_e * p_data[idx_pt_tem - nplane + i]);
                }
            }

            //from right to left
            for (int x = w - 2; x >= 0; --x) {
                int idx_pt = y * w + x;
                int idx_e = idx_pt - y;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_h[idx_e];
                T w_e_inv = (T)1 - w_e;
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] =(DT)(w_e_inv * p_data[idx_pt_tem + i]
                                         + w_e * p_data[idx_pt_tem + nplane + i]);
                }
            }
        }

        //vertical
#pragma omp parallel for
        for (int x = 0; x < w; ++x) {
            int w_nplane = w * nplane;
            //from top to bottom
            for (int y = 1; y < h; ++y) {
                int idx_pt = y * w + x;
                int idx_e = x * h - x + y - 1;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_v[idx_e];
                T w_e_inv = (T)1 - w_e;
                DT* p_data_cur = &p_data[idx_pt_tem];
                DT* p_data_pre = &p_data[idx_pt_tem - w_nplane];
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] =(DT)(w_e_inv * p_data[idx_pt_tem + i]
                                         + w_e * p_data[idx_pt_tem - w_nplane + i]);
                }
            }

            //from bottom to top
            for (int y = h - 2; y >= 0; --y) {
                int idx_pt = y * w + x;
                int idx_e = x * h - x + y;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_v[idx_e];
                T w_e_inv = (T)1 - w_e;
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] =(DT)(w_e_inv * p_data[idx_pt_tem + i]
                                         + w_e * p_data[idx_pt_tem + w_nplane + i]);
                }
            }
        }
    }
}

template <typename T>
template<typename DT>
void GeodesicFiltering<T>::runFilteringRecursive(DT *data,
        int levels_data, const T *guidance, int levels_guidance,
        unsigned char *mask_non_holes, int iter) {
    int nplane = levels_data, h = _height, w = _width;
    DT *p_data = data;
    _guidance = guidance;
    if (levels_guidance == 3) {
        genWeightsColor();
    } else {
        genWeightsGray();
    }
    _copy_memory = false;

    while (iter > 0) {
        --iter;

        //horizontal
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            //from left to right
            for (int x = 1; x < w; ++x) {
                int idx_pt = y * w + x;
                int idx_e = idx_pt - y - 1;
                int idx_pt_tem = nplane * idx_pt;
                T w_e, w_e_inv;

                if (mask_non_holes[idx_pt] > 0 && mask_non_holes[idx_pt - 1] == 0) {
                    w_e = (T)0;
                    w_e_inv = (T)1;
                } else if (mask_non_holes[idx_pt] == 0 && mask_non_holes[idx_pt - 1] > 0) {
                    w_e = (T)1;
                    w_e_inv = (T)0;
                } else {
                    w_e = _weights_h[idx_e];
                    w_e_inv = (T)1 - w_e;
                }

                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] =(DT)(w_e_inv * p_data[idx_pt_tem + i]
                                         + w_e * p_data[idx_pt_tem - nplane + i]);
                }
            }

            //from right to left
            for (int x = w - 2; x >= 0; --x) {
                int idx_pt = y * w + x;
                int idx_e = idx_pt - y;
                int idx_pt_tem = nplane * idx_pt;
                T w_e, w_e_inv;

                if (mask_non_holes[idx_pt] > 0 && mask_non_holes[idx_pt + 1] == 0) {
                    w_e = (T)0;
                    w_e_inv = (T)1;
                } else if (mask_non_holes[idx_pt] == 0 && mask_non_holes[idx_pt + 1] > 0) {
                    w_e = (T)1;
                    w_e_inv = (T)0;
                } else {
                    w_e = _weights_h[idx_e];
                    w_e_inv = (T)1 - w_e;
                }

                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] = (DT)(w_e_inv * p_data[idx_pt_tem + i]
                                       + w_e * p_data[idx_pt_tem + nplane + i]);
                }
            }
        }

        //vertical
        int w_nplane = w * nplane;
#pragma omp parallel for
        for (int x = 0; x < w; ++x) {
            //from top to bottom
            for (int y = 1; y < h; ++y) {
                int idx_pt = y * w + x;
                int idx_e = x * h - x + y - 1;
                int idx_pt_tem = nplane * idx_pt;
                T w_e, w_e_inv;

                if (mask_non_holes[idx_pt] > 0 && mask_non_holes[idx_pt - w] == 0) {
                    w_e = (T)0;
                    w_e_inv = (T)1;
                } else if (mask_non_holes[idx_pt] == 0 && mask_non_holes[idx_pt - w] > 0) {
                    w_e = (T)1;
                    w_e_inv = (T)0;
                } else {
                    w_e = _weights_v[idx_e];
                    w_e_inv = (T)1 - w_e;
                }

                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] = (DT)(w_e_inv * p_data[idx_pt_tem + i]
                                       + w_e * p_data[idx_pt_tem - w_nplane + i]);
                }
            }

            //from bottom to top
            for (int y = h - 2; y >= 0; --y) {
                int idx_pt = y * w + x;
                int idx_e = x * h - x + y;
                int idx_pt_tem = nplane * idx_pt;
                T w_e, w_e_inv;

                if (mask_non_holes[idx_pt] > 0 && mask_non_holes[idx_pt + w] == 0) {
                    w_e = (T)0;
                    w_e_inv = (T)1;
                } else if (mask_non_holes[idx_pt] == 0 && mask_non_holes[idx_pt + w] > 0) {
                    w_e = (T)1;
                    w_e_inv = (T)0;
                } else{
                    w_e = _weights_v[idx_e];
                    w_e_inv = (T)1 - w_e;
                }

                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] = (DT)(w_e_inv * p_data[idx_pt_tem + i]
                                       + w_e * p_data[idx_pt_tem + w_nplane + i]);
                }
            }
        }
    }
}

template <typename T>
void GeodesicFiltering<T>::runFilteringExact(int iter, bool outout_normalized) {
    int nplane = _levels_data, h = _height, w = _width;
    int size = h * w;
    T *weight_sum_h = new T[size];
    for (int j = 0; j < size; j++) {
        weight_sum_h[j] = (T)1;
    }
    runGeoFilterHorizontal_singleplane(_weights_h, weight_sum_h);
    T *weight_sum_v = NULL;
    if (iter >= 1) {
        weight_sum_v = new T[size];
        for (int j = 0; j< size; j++) {
            weight_sum_v[j] = (T)1;
        }
        runGeoFilterVertical_singleplane(_weights_v, weight_sum_v);
    }
    T *p_data = _data;

    while (iter>0) {
        --iter;

        //horizontal
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            //from left to right
            for (int x = 1; x < w; ++x) {
                int idx_pt = y * w + x;
                int idx_e = idx_pt - y - 1;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_h[idx_e];
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] = p_data[idx_pt_tem + i]
                                           + w_e * p_data[idx_pt_tem - nplane + i];
                }
            }

            //from right to left
            for (int x = w - 2; x >= 0; --x) {
                int idx_pt = y * w + x;
                int idx_e = idx_pt - y;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_h[idx_e];
                T inv_w_e2 = (T)1 - w_e * w_e;
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] = inv_w_e2 * p_data[idx_pt_tem + i]
                                           + w_e * p_data[idx_pt_tem + nplane + i];
                }
            }
        }

        //normalization
        T scale_nor = 0;
        for (int idx = 0; idx < size; ++idx) {
            scale_nor = ((T)1) / weight_sum_h[idx];
            int pt_st = idx * nplane;
            for (int d = 0; d < nplane; ++d) {
                p_data[pt_st + d] *= scale_nor;
            }
        }

        //vertical
        int w_nplane = w * nplane;
#pragma omp parallel for
        for (int x = 0; x < w; ++x) {
            //from top to bottom
            for (int y = 1; y < h; ++y) {
                int idx_pt = y * w + x;
                int idx_e = x * h - x + y - 1;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_v[idx_e];
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem] = p_data[idx_pt_tem + i]
                                       + w_e * p_data[idx_pt_tem - w_nplane + i];
                }
            }

            //from bottom to top
            for (int y = h - 2; y >= 0; --y) {
                int idx_pt = y * w + x;
                int idx_e = x * h - x + y;
                int idx_pt_tem = nplane * idx_pt;
                T w_e = _weights_v[idx_e];
                T inv_w_e2 = (T)1 - w_e * w_e;
                for (int i = 0; i < nplane; ++i) {
                    p_data[idx_pt_tem + i] = inv_w_e2 * p_data[idx_pt_tem + i]
                                           + w_e * p_data[idx_pt_tem + w_nplane + i];
                }
            }
        }

        //normalization
        if (iter>0 || outout_normalized) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int idx = y * w + x;
                    T scale_nor = (T)1 / weight_sum_v[idx];
                    for (int d = 0; d < nplane; ++d) {
                        p_data[idx*nplane + d] *= scale_nor;
                    }
                }
            }
        }
    }

    if (weight_sum_h != NULL) {
        delete []weight_sum_h;
        weight_sum_h = NULL;
    }
    if (weight_sum_v != NULL) {
        delete []weight_sum_v;
        weight_sum_v = NULL;
    }
}

}   // namespace cv
}   // namespace deeprob
