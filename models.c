#include "models.h"

void printarray(float *Data, int size, bool fc)
{
    if(fc == true)
    {
        for(int i = 0; i < size; i++)
            printf("%.3f\n", Data[i]);
        return;
    }
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
            printf("%.3f\t", Data[i*size + j]);
        printf("\n");
    }
}

void vgg_cifar10_binary(float **Data, int **Weight, float **Bias, float **BN_G, float **BN_B, float **BN_RM, float **BN_RV, float *Result)
{
    // 128C3 - 128C3 - P2
    conv2d_float_layer(Data[0], Weight[0], Bias[0], Data[1], 3, 128, 32, 32, 32, 32, 3, 3, 1, 1);
    bn(Data[1], BN_G[0], BN_B[0], BN_RM[0], BN_RV[0], Data[1], 128, 32, 32, true);
    hardtanh(Data[1], 32*32*128);
    conv2d_binary_layer(Data[1], Weight[1], Bias[1], Data[2], 128, 128, 32, 32, 32, 32, 3, 3, 1, 1);
    max_pooling2d(Data[2], Data[3], 128, 32, 32, 16, 16, 2, 2, 2);
    bn(Data[3], BN_G[1], BN_B[1], BN_RM[1], BN_RV[1], Data[3], 128, 16, 16, true);
    hardtanh(Data[3], 16*16*128);

    // 256C3 - 256C3 - P2
    conv2d_binary_layer(Data[3], Weight[2], Bias[2], Data[4], 128, 256, 16, 16, 16, 16, 3, 3, 1, 1);
    bn(Data[4], BN_G[2], BN_B[2], BN_RM[2], BN_RV[2], Data[4], 256, 16, 16, true);
    hardtanh(Data[4], 16*16*256);
    conv2d_binary_layer(Data[4], Weight[3], Bias[3], Data[5], 256, 256, 16, 16, 16, 16, 3, 3, 1, 1);
    max_pooling2d(Data[5], Data[6], 256, 16, 16, 8, 8, 2, 2, 2);
    bn(Data[6], BN_G[3], BN_B[3], BN_RM[3], BN_RV[3], Data[6], 256, 8, 8, true);
    hardtanh(Data[6], 8*8*256);
    
    // 512C3 - 512C3 - P2
    conv2d_binary_layer(Data[6], Weight[4], Bias[4], Data[7], 256, 512, 8, 8, 8, 8, 3, 3, 1, 1);
    bn(Data[7], BN_G[4], BN_B[4], BN_RM[4], BN_RV[4], Data[7], 512, 8, 8, true);
    hardtanh(Data[7], 8*8*512);
    conv2d_binary_layer(Data[7], Weight[5], Bias[5], Data[8], 512, 512, 8, 8, 8, 8, 3, 3, 1, 1);
    max_pooling2d(Data[8], Data[9], 512, 8, 8, 4, 4, 2, 2, 2);
    bn(Data[9], BN_G[5], BN_B[5], BN_RM[5], BN_RV[5], Data[9], 512, 4, 4, true);
    hardtanh(Data[9], 4*4*512);

    // 1024FC - 1024FC - 10FC
    fc_binary_layer(Data[9], Weight[6], Bias[6], Data[10], 8192, 1024);
    bn(Data[10], BN_G[6], BN_B[6], BN_RM[6], BN_RV[6], Data[10], 1024, 1, 1, true);
    hardtanh(Data[10], 1024);

    fc_binary_layer(Data[10], Weight[7], Bias[7], Data[11], 1024, 1024);
    bn(Data[11], BN_G[7], BN_B[7], BN_RM[7], BN_RV[7], Data[11], 1024, 1, 1, true);
    hardtanh(Data[11], 1024);

    fc_binary_layer(Data[11], Weight[8], Bias[8], Data[12], 1024, 10);
    bn(Data[12], BN_G[8], BN_B[8], BN_RM[8], BN_RV[8], Data[12], 10, 1, 1, true);
    softmax(Result, Data[12], 10);
}

void vgg_cifar10(float **Data, float **Weight, float **Bias, float **BN_G, float **BN_B, float **BN_RM, float **BN_RV, float *Result)
{
    // 128C3 - 128C3 - P2
    conv2d_layer(Data[0], Weight[0], Bias[0], Data[1], 1, 128, 32, 32, 32, 32, 3, 3, 1, 1);
    bn(Data[1], BN_G[0], BN_B[0], BN_RM[0], BN_RV[0], Data[1], 128, 32, 32, false);
    relu(Data[1], 32*32*128);
    conv2d_layer(Data[1], Weight[1], Bias[1], Data[2], 128, 128, 32, 32, 32, 32, 3, 3, 1, 1);
    max_pooling2d(Data[2], Data[3], 128, 32, 32, 16, 16, 2, 2, 2);
    bn(Data[3], BN_G[1], BN_B[1], BN_RM[1], BN_RV[1], Data[3], 128, 16, 16, false);
    relu(Data[3], 16*16*128);

    // 256C3 - 256C3 - P2
    conv2d_layer(Data[3], Weight[2], Bias[2], Data[4], 128, 256, 16, 16, 16, 16, 3, 3, 1, 1);
    bn(Data[4], BN_G[2], BN_B[2], BN_RM[2], BN_RV[2], Data[4], 256, 16, 16, false);
    relu(Data[4], 16*16*256);
    conv2d_layer(Data[4], Weight[3], Bias[3], Data[5], 256, 256, 16, 16, 16, 16, 3, 3, 1, 1);
    max_pooling2d(Data[5], Data[6], 256, 16, 16, 8, 8, 2, 2, 2);
    bn(Data[6], BN_G[3], BN_B[3], BN_RM[3], BN_RV[3], Data[6], 256, 8, 8, false);
    relu(Data[6], 8*8*256);

    // 512C3 - 512C3 - P2
    conv2d_layer(Data[6], Weight[4], Bias[4], Data[7], 256, 512, 8, 8, 8, 8, 3, 3, 1, 1);
    bn(Data[7], BN_G[4], BN_B[4], BN_RM[4], BN_RV[4], Data[7], 512, 8, 8, false);
    relu(Data[7], 8*8*512);
    conv2d_layer(Data[7], Weight[5], Bias[5], Data[8], 512, 512, 8, 8, 8, 8, 3, 3, 1, 1);
    max_pooling2d(Data[8], Data[9], 512, 8, 8, 4, 4, 2, 2, 2);
    bn(Data[9], BN_G[5], BN_B[5], BN_RM[5], BN_RV[5], Data[9], 512, 4, 4, false);
    relu(Data[9], 4*4*512);

    // Flatten (Tensorflow ..)
    float Data_f[4*4*512] = { 0, };
    for(int i = 0; i < 512; i++)
        for(int j = 0; j < 4; j++)
            for(int z = 0; z < 4; z++)
                Data_f[(j*4 + z)*512 + i] = Data[9][i*16 + j*4 + z];

    // 1024FC - 1024FC - 10FC
    fc_layer(Data_f, Weight[6], Bias[6], Data[10], 8192, 1024);
    bn(Data[10], BN_G[6], BN_B[6], BN_RM[6], BN_RV[6], Data[10], 1024, 1, 1, false);
    relu(Data[10], 1024);
    
    fc_layer(Data[10], Weight[7], Bias[7], Data[11], 1024, 1024);
    bn(Data[11], BN_G[7], BN_B[7], BN_RM[7], BN_RV[7], Data[11], 1024, 1, 1, false);
    relu(Data[11], 1024);
    
    fc_layer(Data[11], Weight[8], Bias[8], Data[12], 1024, 10);
    bn(Data[12], BN_G[8], BN_B[8], BN_RM[8], BN_RV[8], Data[12], 10, 1, 1, false);
    softmax(Result, Data[12], 10);
}
