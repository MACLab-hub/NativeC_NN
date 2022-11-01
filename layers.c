#include "layers.h"

void conv2d_layer(float *data, const float *w, float *bias, float *data_out, int C_in, int C_out,
                  int H_in, int W_in, int H_out, int W_out, int H_f, int W_f, int stride, int padding)
{
    conv2d(data, w, data_out, C_in, C_out, H_in, W_in, H_out, W_out, H_f, W_f, stride, padding);

    int mat_size = H_out*W_out;
    for(int c = 0; c < C_out; c++)
        for(int k = 0; k < mat_size; k++)
            data_out[c*mat_size + k] = data_out[c*mat_size + k] + bias[c];
}

void fc_layer(float *data, const float *w, float *bias, float *data_out, int H_in, int H_out)
{
    fc(data, w, data_out, H_in, H_out);

    for(int k = 0; k < H_out; k++)
       data_out[k] = data_out[k] + bias[k];
}

void conv2d_quan_layer(float *data, const int8_t *w, float *bias, float *data_out, float *s_w,
                       int C_in, int C_out, int H_in, int W_in, int H_out, int W_out, int H_f, int W_f,
                       int stride, int padding)
{
    int8_t *quan_data = (int8_t*)malloc(sizeof(int8_t)*C_in*H_in*W_in);
    float s_x = 0;
    int8_t z = 0;
    quantization(data, quan_data, H_in*W_in*C_in, true, &s_x, &z);
    
    int *quan_data_out = (int*)malloc(sizeof(int)*C_out*H_out*W_out);
    conv2d_quan(quan_data, w, quan_data_out, C_in, C_out, H_in, W_in, H_out, W_out,
                H_f, W_f, stride, padding, z);

    de_quantization(quan_data_out, data_out, H_out*W_out, C_out, s_x, s_w, bias);

    free(quan_data);
    free(quan_data_out);
}

void fc_quan_layer(float *data, const int8_t *w, float *bias, float *data_out, float *s_w,
                   int H_in, int H_out)
{
    int8_t *quan_data = (int8_t*)malloc(sizeof(int8_t)*H_in);
    float s_x = 0;
    int8_t z = 0;
    quantization(data, quan_data, H_in, true, &s_x, &z);
    
    int *quan_data_out = (int*)malloc(sizeof(int)*H_out);
    fc_quan(quan_data, w, quan_data_out, H_in, H_out, z);
    
    de_quantization_fc(quan_data_out, data_out, H_out, s_x, s_w, bias);
    
    free(quan_data);
    free(quan_data_out);
}

void conv2d_binary_layer(float *data, const int *w, float *bias, float *data_out,
                         int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
                         int H_f, int W_f, int stride, int padding)
{
    int data_in_size = C_in*H_in*W_in;
    int data_out_size = C_out*H_out*W_out;

    int *binary_data = (int*)malloc(sizeof(int)*data_in_size);
    binarize(data, binary_data, C_in*H_in*W_in);
    
    int *binary_data_out = (int*)malloc(sizeof(int)*data_out_size);
    conv2d_binary(binary_data, w, binary_data_out, C_in, C_out, H_in, W_in, H_out, W_out,
                  H_f, W_f, stride, padding);

    int mat_size = H_out*W_out;
    for(int c = 0; c < C_out; c++)
        for(int k = 0; k < mat_size; k++)
            data_out[c*mat_size + k] = (float)binary_data_out[c*mat_size + k] + bias[c];

    free(binary_data);
    free(binary_data_out);
}

void fc_binary_layer(float *data, const int *w, float *bias, float *data_out, int H_in, int H_out)
{
   int *binary_data = (int*)malloc(sizeof(int)*H_in);
   binarize(data, binary_data, H_in);

   int *binary_data_out = (int*)malloc(sizeof(int)*H_out);
   fc_binary(binary_data, w, binary_data_out, H_in, H_out);

   for(int k = 0; k < H_out; k++)
       data_out[k] = (float)binary_data_out[k] + bias[k];

   free(binary_data);
   free(binary_data_out);
}

void conv2d_float_layer(float *data, const int *w, float *bias, float *data_out,
                         int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
                         int H_f, int W_f, int stride, int padding)
{
    conv2d_float(data, w, data_out, C_in, C_out, H_in, W_in, H_out, W_out, H_f, W_f,
                 stride, padding);

    int mat_size = H_out*W_out;
    for(int c = 0; c < C_out; c++)
        for(int k = 0; k < mat_size; k++)
            data_out[c*mat_size + k] = data_out[c*mat_size + k] + bias[c];
}
