#ifndef DEEPROB_CV_CONFIG_H_IN_
#define DEEPROB_CV_CONFIG_H_IN_

#define DEEPROB_CV_VERSION_MAJOR 0
#define DEEPROB_CV_VERSION_MINOR 1
/* #undef USE_EIGEN */
#define USE_SSE2
#define USE_SSE3
#define USE_SSSE3
#define USE_SSE4_1
/* #undef USE_SSE4_2 */

#ifdef USE_EIGEN
#include "Eigen/Dense"
#include "Eigen/SVD"
#endif

#ifdef USE_SSE2
#include <emmintrin.h>
#endif

#ifndef _DEBUG
#include <omp.h>
#endif

#endif // DEEPROB_CV_CONFIG_H_IN_
