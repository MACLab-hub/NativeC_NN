#include "compute.h"

float max_v(float *data, int size)
{
    float max = -1000;
    for(int i = 0; i < size; i++)
    {
        if(max < data[i])
            max = data[i];
    }
    return max;
}

float min_v(float *data, int size)
{
    float min = data[0];
    for(int i = 0; i < size; i++)
    {
        if(min > data[i])
            min = data[i];
    }
    return min;
}

int Round(float v)
{
    if((int)(v + 0.5) == (int)v)
        return v;
    else
        return v + 1;
}

void quantization(float *data, int8_t *quan_data, int mat_size, bool mode, float *s_x, int8_t *z)
{
    float s = 0;
    float a = 0;
    float b = 0;
    int max_num = pow(2, BIT - 1);

    a = max_v(data, mat_size);
    b = min_v(data, mat_size);

    if(mode == true)    //true = affine
    {
        s = (max_num*2 - 1) / (a - b);
        *z = (-1 * Round(b*s)) - max_num;
    }
    else                //false = scale
    {
        float alpha = 0;
        alpha = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
        s = (max_num - 1) / alpha;
    }

    *s_x = s;
    for(int i = 0; i < mat_size; i++)
    {
        int tmp_data = Round(data[i]*s + *z);
        if(tmp_data < -1*max_num + 1)
            quan_data[i] = -1*max_num + 1;
        else if(tmp_data > max_num - 1)
            quan_data[i] = max_num - 1;
        else
            quan_data[i] = (int8_t)tmp_data;
    }
}

void de_quantization(int *quan_data, float *data, int mat_size, int C, float s_x, float *s_w, float *b)
{
    for(int c = 0; c < C; c++)
        for(int k = 0; k < mat_size; k++)
        {
            int idx = c*mat_size + k;
            data[idx] = quan_data[idx] / (s_x*s_w[c]) + b[c];
        }
}

void de_quantization_fc(int *quan_data, float *data, int mat_size, float s_x, float *s_w, float *b)
{
    for(int k = 0; k < mat_size; k++)
    {
        data[k] = quan_data[k] / (s_x*s_w[k]) + b[k];
    }
}

void conv2d(const float *data_in, const float *w, float *data_out, int C_in, int C_out, int H_in, int W_in,
            int H_out, int W_out, int H_f, int W_f, int stride, int padding)
{
    int out_c_idx;
    int out_y_idx, out_x_idx;
    int in_c_idx;
    int f_y_idx, f_x_idx;

    // Padding
    float *data_padding;
    if(padding != 0)
    {
        H_in += padding*2;
        W_in += padding*2;
        data_padding = (float*)malloc(sizeof(float)*H_in*W_in*C_in);
        memset(data_padding, 0, sizeof(float)*H_in*W_in*C_in);
        for(int c = 0; c < C_in; c++)
            for(int h = padding; h < H_in - padding; h++)
                for(int w = padding; w < W_in - padding; w++)
                {
                    int idx = c*(H_in - padding*2)*(W_in - padding*2) + (h - padding)*(W_in - padding*2) + (w - padding);
                    data_padding[c*H_in*W_in + h*W_in + w] = data_in[idx];
                }
    }
    else
        data_padding = data_in;

    // Convolution
    for(out_c_idx = 0; out_c_idx < C_out; out_c_idx++)
    {
        int out_c_addr = out_c_idx * H_out * W_out;
        int f_out_c_addr = out_c_idx * C_in * H_f * W_f;
        for(out_y_idx = 0; out_y_idx < H_out; out_y_idx++)
        {
            int out_y_addr = out_y_idx * W_out;
            for(out_x_idx = 0; out_x_idx < W_out; out_x_idx++)
            {
                int out_idx = out_c_addr + out_y_addr + out_x_idx;
                float sum = 0;
                for(in_c_idx = 0; in_c_idx < C_in; in_c_idx++)
                {
                    int in_c_addr = in_c_idx * H_in * W_in;
                    int f_in_c_addr = in_c_idx * H_f * W_f;
                    for(f_y_idx = 0; f_y_idx < H_f; f_y_idx++)
                    {
                        int f_y_addr = f_y_idx * W_f;
                        for(f_x_idx = 0; f_x_idx < W_f; f_x_idx++)
                        {
                            int in_idx = in_c_addr + out_y_idx*stride*W_in + out_x_idx*stride + f_y_idx*W_in + f_x_idx;
                            int f_idx = f_out_c_addr + f_in_c_addr + f_y_addr + f_x_idx;
                            sum += data_padding[in_idx] * w[f_idx];
                        }
                    }
                }
                data_out[out_idx] = sum;
            }
        }
    }

    if(padding != 0)
        free(data_padding);
}

void conv2d_quan(const int8_t *data_in, const int8_t *w, int *data_out, int C_in, int C_out, int H_in, int W_in,
                 int H_out, int W_out, int H_f, int W_f, int stride, int padding, int8_t z)
{
    int out_c_idx;
    int out_y_idx, out_x_idx;
    int in_c_idx;
    int f_y_idx, f_x_idx;

    // Padding
    int8_t *data_padding;
    if(padding != 0)
    {
        H_in += padding*2;
        W_in += padding*2;
        data_padding = (int8_t*)malloc(sizeof(int8_t)*H_in*W_in*C_in);
        memset(data_padding, z, sizeof(int8_t)*H_in*W_in*C_in);
        for(int c = 0; c < C_in; c++)
            for(int h = padding; h < H_in - padding; h++)
                for(int w = padding; w < W_in - padding; w++)
                {
                    int idx = c*(H_in - padding*2)*(W_in - padding*2) + (h - padding)*(W_in - padding*2) + (w - padding);
                    data_padding[c*H_in*W_in + h*W_in + w] = data_in[idx];
                }
    }
    else
        data_padding = data_in;

    // Convolution
    for(out_c_idx = 0; out_c_idx < C_out; out_c_idx++)
    {
        int out_c_addr = out_c_idx * H_out * W_out;
        int f_out_c_addr = out_c_idx * C_in * H_f * W_f;
        for(out_y_idx = 0; out_y_idx < H_out; out_y_idx++)
        {
            int out_y_addr = out_y_idx * W_out;
            for(out_x_idx = 0; out_x_idx < W_out; out_x_idx++)
            {
                int out_idx = out_c_addr + out_y_addr + out_x_idx;
                int sum = 0;
                for(in_c_idx = 0; in_c_idx < C_in; in_c_idx++)
                {
                    int in_c_addr = in_c_idx * H_in * W_in;
                    int f_in_c_addr = in_c_idx * H_f * W_f;
                    for(f_y_idx = 0; f_y_idx < H_f; f_y_idx++)
                    {
                        int f_y_addr = f_y_idx * W_f;
                        for(f_x_idx = 0; f_x_idx < W_f; f_x_idx++)
                        {
                            int in_idx = in_c_addr + out_y_idx*stride*W_in + out_x_idx*stride + f_y_idx*W_in + f_x_idx;
                            int f_idx = f_out_c_addr + f_in_c_addr + f_y_addr + f_x_idx;
                            sum += (data_padding[in_idx] - z) * w[f_idx];
                        }
                    }
                }
                data_out[out_idx] = sum;
            }
        }
    }

    if(padding != 0)
        free(data_padding);
}

void average_pooling2d(const float *data_in, float *data_out, int C_out, int H_in, int W_in,
                       int H_out, int W_out, int H_f, int W_f, int stride)
{
    int out_c_idx;
    int out_y_idx, out_x_idx;
    int f_y_idx, f_x_idx;

    for(out_c_idx = 0; out_c_idx < C_out; out_c_idx++)
    {
        int out_c_addr = out_c_idx * H_out * W_out;
        for(out_y_idx = 0; out_y_idx < H_out; out_y_idx++)
        {
            int out_y_addr = out_y_idx * W_out;
            for(out_x_idx = 0; out_x_idx < W_out; out_x_idx++)
            {
                int out_idx = out_c_addr + out_y_addr + out_x_idx;
                float sum = 0;
                for(f_y_idx = 0; f_y_idx < H_f; f_y_idx++)
                    for(f_x_idx = 0; f_x_idx < W_f; f_x_idx++)
                    {
                        int in_idx = out_c_idx*H_in*W_in + out_y_idx*stride*W_in + f_y_idx*W_in + out_x_idx*stride + f_x_idx;
                        sum += data_in[in_idx];
                    }
                data_out[out_idx] = (float)sum / (H_f*W_f);
            }
        }
    }
}

void max_pooling2d(const float *data_in, float *data_out, int C_out, int H_in, int W_in,
                   int H_out, int W_out, int H_f, int W_f, int stride)
{
    int out_c_idx;
    int out_y_idx, out_x_idx;
    int f_y_idx, f_x_idx;

    for(out_c_idx = 0; out_c_idx < C_out; out_c_idx++)
    {
        int out_c_addr = out_c_idx * H_out * W_out;
        for(out_y_idx = 0; out_y_idx < H_out; out_y_idx++)
        {
            int out_y_addr = out_y_idx * W_out;
            for(out_x_idx = 0; out_x_idx < W_out; out_x_idx++)
            {
                int out_idx = out_c_addr + out_y_addr + out_x_idx;
                float max = -1000;
                for(f_y_idx = 0; f_y_idx < H_f; f_y_idx++)
                    for(f_x_idx = 0; f_x_idx < W_f; f_x_idx++)
                    {
                        int in_idx = out_c_idx*H_in*W_in + out_y_idx*stride*W_in + f_y_idx*W_in + out_x_idx*stride + f_x_idx;
                        if(max < data_in[in_idx])
                            max =  data_in[in_idx];
                    }
                data_out[out_idx] = max;
            }
        }
    }
}

void bn(const float *data_in, const float *gamma, const float *beta, const float *moving_mean, const float *moving_var,
        float *data_out, int C, int H, int W, bool binary)
{
    //float epsilon = 0.00001;
    float epsilon = 0.001;

    int c_idx, d_idx;
    int d_size = H*W;

    for(c_idx = 0; c_idx < C; c_idx++)
    {
        for(d_idx = 0; d_idx < d_size; d_idx++)
        {
            int index = c_idx*d_size + d_idx;
            float tmp1 = data_in[index] - moving_mean[c_idx];
            float tmp2 = sqrt(moving_var[c_idx] + epsilon);
            data_out[index] = gamma[c_idx] * tmp1 / tmp2 + beta[c_idx];
        
            if(binary == true)
            {
                if(data_out[index] >= 0)
                    data_out[index] = 1;
                else
                    data_out[index] = -1;
            }
        }
    }
}

void fc(const float *data_in, const float *w, float *data_out, int H_in, int H_out)
{
    int h_out, h_in;

    for(h_out = 0; h_out < H_out; h_out++)
    {
        float sum = 0;
        for(h_in = 0; h_in < H_in; h_in++)
            sum += data_in[h_in] * w[H_in*h_out + h_in];
        data_out[h_out] = sum;
    }
}

void fc_quan(const int8_t *data_in, const int8_t *w, int *data_out, int H_in, int H_out, int8_t z)
{
    int h_out, h_in;

    for(h_out = 0; h_out < H_out; h_out++)
    {
        int sum = 0;
        for(h_in = 0; h_in < H_in; h_in++)
            sum += (data_in[h_in] - z) * w[H_in*h_out + h_in];
        data_out[h_out] = sum;
    }
}

void relu(float *data, const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(data[i] < 0)
            data[i] = 0;
    }
}

void relu_int(int *data, const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(data[i] < 0)
            data[i] = 0;
    }
}

void softmax(float *p, float *x, int num_class)
{
    float max = -1000;
    float sum = 0;
    float *exp_x = (float*)malloc(sizeof(float)*10);

    for(int i = 0; i < num_class; i++)
    {
        if(max < x[i])
            max = x[i];
    }
    for(int i = 0; i < num_class; i++)
    {
        x[i] -= max;
        exp_x[i] = expf(x[i]);
        sum += exp_x[i];
    }
    for(int i = 0; i < num_class; i++)
        p[i] = exp_x[i] / sum;

    free(exp_x);
}

void hardtanh(float *data, const int size)
{
    for(int i = 0; i < size; i++)
    {
        if(data[i] >= 1)
            data[i] = 1;
        else if(data[i] <= -1)
            data[i] = -1;
    }
}

void binarize(float *data_in, int *data_out, const int size)
{
    // Sign Function
    for(int i = 0; i < size; i++)
    {
        if(data_in[i] >= 0)
            data_out[i] = 1;
        else
            data_out[i] = -1;
    }
}

void conv2d_binary(const int *data_in, const int *w, int *data_out, int C_in, int C_out, int H_in, int W_in,
                   int H_out, int W_out, int H_f, int W_f, int stride, int padding)
{
    int out_c_idx;
    int out_y_idx, out_x_idx;
    int in_c_idx;
    int f_y_idx, f_x_idx;

    // Padding
    int *data_padding;
    if(padding != 0)
    {
        H_in += padding*2;
        W_in += padding*2;
        data_padding = (int*)malloc(sizeof(int)*H_in*W_in*C_in);
        memset(data_padding, 0, sizeof(int)*H_in*W_in*C_in);
        for(int c = 0; c < C_in; c++)
            for(int h = padding; h < H_in - padding; h++)
                for(int w = padding; w < W_in - padding; w++)
                {
                    int idx = c*(H_in - padding*2)*(W_in - padding*2) + (h - padding)*(W_in - padding*2) + (w - padding);
                    data_padding[c*H_in*W_in + h*W_in + w] = data_in[idx];
                }
    }
    else
        data_padding = data_in;

    // Convolution
    for(out_c_idx = 0; out_c_idx < C_out; out_c_idx++)
    {
        int out_c_addr = out_c_idx * H_out * W_out;
        int f_out_c_addr = out_c_idx * C_in * H_f * W_f;
        for(out_y_idx = 0; out_y_idx < H_out; out_y_idx++)
        {
            int out_y_addr = out_y_idx * W_out;
            for(out_x_idx = 0; out_x_idx < W_out; out_x_idx++)
            {
                int out_idx = out_c_addr + out_y_addr + out_x_idx;
                int sum = 0;
                for(in_c_idx = 0; in_c_idx < C_in; in_c_idx++)
                {
                    int in_c_addr = in_c_idx * H_in * W_in;
                    int f_in_c_addr = in_c_idx * H_f * W_f;
                    for(f_y_idx = 0; f_y_idx < H_f; f_y_idx++)
                    {
                        int f_y_addr = f_y_idx * W_f;
                        for(f_x_idx = 0; f_x_idx < W_f; f_x_idx++)
                        {
                            int in_idx = in_c_addr + out_y_idx*stride*W_in + out_x_idx*stride + f_y_idx*W_in + f_x_idx;
                            int f_idx = f_out_c_addr + f_in_c_addr + f_y_addr + f_x_idx;
                            sum += data_padding[in_idx] * w[f_idx];
                        }
                    }
                }
                data_out[out_idx] = sum;
            }
        }
    }

    if(padding != 0)
        free(data_padding);
}

void fc_binary(const int *data_in, const int *w, int *data_out, int H_in, int H_out)
{
    int h_out, h_in;

    for(h_out = 0; h_out < H_out; h_out++)
    {
        int sum = 0;
        for(h_in = 0; h_in < H_in; h_in++)
            sum += data_in[h_in] * w[H_in*h_out + h_in];
        data_out[h_out] = sum;
    }
}

void conv2d_float(const float *data_in, const int *w, float *data_out, int C_in, int C_out, int H_in, int W_in,
                   int H_out, int W_out, int H_f, int W_f, int stride, int padding)
{
    int out_c_idx;
    int out_y_idx, out_x_idx;
    int in_c_idx;
    int f_y_idx, f_x_idx;

    // Padding
    float *data_padding;
    if(padding != 0)
    {
        H_in += padding*2;
        W_in += padding*2;
        data_padding = (float*)malloc(sizeof(float)*H_in*W_in*C_in);
        memset(data_padding, 0, sizeof(float)*H_in*W_in*C_in);
        for(int c = 0; c < C_in; c++)
            for(int h = padding; h < H_in - padding; h++)
                for(int w = padding; w < W_in - padding; w++)
                {
                    int idx = c*(H_in - padding*2)*(W_in - padding*2) + (h - padding)*(W_in - padding*2) + (w - padding);
                    data_padding[c*H_in*W_in + h*W_in + w] = data_in[idx];
                }
    }
    else
        data_padding = data_in;

    // Convolution
    for(out_c_idx = 0; out_c_idx < C_out; out_c_idx++)
    {
        int out_c_addr = out_c_idx * H_out * W_out;
        int f_out_c_addr = out_c_idx * C_in * H_f * W_f;
        for(out_y_idx = 0; out_y_idx < H_out; out_y_idx++)
        {
            int out_y_addr = out_y_idx * W_out;
            for(out_x_idx = 0; out_x_idx < W_out; out_x_idx++)
            {
                int out_idx = out_c_addr + out_y_addr + out_x_idx;
                float sum = 0;
                for(in_c_idx = 0; in_c_idx < C_in; in_c_idx++)
                {
                    int in_c_addr = in_c_idx * H_in * W_in;
                    int f_in_c_addr = in_c_idx * H_f * W_f;
                    for(f_y_idx = 0; f_y_idx < H_f; f_y_idx++)
                    {
                        int f_y_addr = f_y_idx * W_f;
                        for(f_x_idx = 0; f_x_idx < W_f; f_x_idx++)
                        {
                            int in_idx = in_c_addr + out_y_idx*stride*W_in + out_x_idx*stride + f_y_idx*W_in + f_x_idx;
                            int f_idx = f_out_c_addr + f_in_c_addr + f_y_addr + f_x_idx;
                            sum += data_padding[in_idx] * w[f_idx];
                        }
                    }
                }
                data_out[out_idx] = sum;
            }
        }
    }

    if(padding != 0)
        free(data_padding);
}
