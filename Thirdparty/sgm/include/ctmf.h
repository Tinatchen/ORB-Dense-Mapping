#ifndef DEEPROB_CV_STEREO_CTMF_H_
#define DEEPROB_CV_STEREO_CTMF_H_

//this is an implementation for constant-time median filtering. See "ctmf.c" for more details.

#ifdef __cplusplus
extern "C" {
#endif

void ctmf(const unsigned char* src, unsigned char* dst,int width, int height,int src_step_row, int dst_step_row,int r, int channels, unsigned long memsize);

#ifdef __cplusplus
}
#endif

#endif // DEEPROB_CV_STEREO_CTMF_H_
