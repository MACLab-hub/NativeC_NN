#include "compute.h"

void conv2d_layer(float *data, const float *w, float *bias, float *data_out, int C_in, int C_out,
                  int H_in, int W_in, int H_out, int W_out, int H_f, int W_f, int stride, int padding);

void fc_layer(float *data, const float *w, float *bias, float *data_out, int H_in, int H_out);

void conv2d_quan_layer(float *data, const int8_t *w, float *bias, float *data_out, float *s_w,
                       int C_in, int C_out, int H_in, int W_in, int H_out, int W_out, int H_f, int W_f,
                       int stride, int padding);

void fc_quan_layer(float *data, const int8_t *w, float *bias, float *data_out, float *s_w,
                   int H_in, int H_out);

void conv2d_binary_layer(float *data, const int *w, float *bias, float *data_out,
                         int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
                         int H_f, int W_f, int stride, int padding);

void fc_binary_layer(float *data, const int *w, float *bias, float *data_out, int H_in, int H_out);

void conv2d_float_layer(float *data, const int *w, float *bias, float *data_out,
                         int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
                         int H_f, int W_f, int stride, int padding);
