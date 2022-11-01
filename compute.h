#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#define BIT 8

float max_v(float *data, int size);

float min_v(float *data, int size);

int Round(float v);

void quantization(float *data, int8_t *quan_data, int mat_size, bool mode, float *s_x, int8_t *z);

void de_quantization(int *quan_data, float *data, int mat_size, int C, float s_x, float *s_w, float *b);

void de_quantization_fc(int *quan_data, float *data, int mat_size, float s_x, float *s_w, float *b);

void conv2d(const float *data_in, const float *w, float *data_out, int C_in, int C_out, int H_in, int W_in,
            int H_out, int W_out, int H_f, int W_f, int stride, int padding);

void conv2d_quan(const int8_t *data_in, const int8_t *w, int *data_out, int C_in, int C_out, int H_in, int W_in,
                 int H_out, int W_out, int H_f, int W_f, int stride, int padding, int8_t z);

void average_pooling2d(const float *data_in, float *data_out, int C_out, int H_in, int W_in,
                       int H_out, int W_out, int H_f, int W_f, int stride);

void max_pooling2d(const float *data_in, float *data_out, int C_out, int H_in, int W_in,
                   int H_out, int W_out, int H_f, int W_f, int stride);

void bn(const float *data_in, const float *gamma, const float *beta, const float *moving_mean, const float *moving_var,
        float *data_out, int C, int H, int W, bool binary);

void fc(const float *data_in, const float *w, float *data_out, int H_in, int H_out);

void fc_quan(const int8_t *data_in, const int8_t *w, int *data_out, int H_in, int H_out, int8_t z);

void relu(float *data, const int size);

void relu_int(int *data, const int size);

void softmax(float *p, float *x, int num_class);

void hardtanh(float *data, const int size);

void binarize(float *data_in, int *data_out, const int size);

void conv2d_binary(const int *data_in, const int *w, int *data_out, int C_in, int C_out, int H_in, int W_in,
                   int H_out, int W_out, int H_f, int W_f, int stride, int padding);

void fc_binary(const int *data_in, const int *w, int *data_out, int H_in, int H_out);

void conv2d_float(const float *data_in, const int *w, float *data_out, int C_in, int C_out, int H_in, int W_in,
                   int H_out, int W_out, int H_f, int W_f, int stride, int padding);
