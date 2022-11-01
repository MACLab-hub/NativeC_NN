#include "layers.h"

void vgg_cifar10_binary(float **Data, int **Weight, float **Bias, float **BN_G, float **BN_B, float **BN_RM, float **BN_RV, float *Result);

void vgg_cifar10(float **Data, float **Weight, float **Bias, float **BN_G, float **BN_B, float **BN_RM, float **BN_RV, float *Result);
