#include "models.h"

void read_data(float *Original_Data, int *Class);
void read_weight(int **Weight, float **Bias, float **BN_G, float **BN_B, float **BN_RM, float **BN_RV);

int main()
{
    float *Original_Data = (float*)malloc(sizeof(float)*32*32*3*10000);
    int *Class = (int*)malloc(sizeof(int)*10000);

    float **Data = (float**)malloc(sizeof(float*)*13);
    int **Weight = (int**)malloc(sizeof(int*)*9);
    float **Bias = (float**)malloc(sizeof(float*)*9);

    float **BN_G = (float**)malloc(sizeof(float*)*9);
    float **BN_B = (float**)malloc(sizeof(float*)*9);
    float **BN_RM = (float**)malloc(sizeof(float*)*9);
    float **BN_RV = (float**)malloc(sizeof(float*)*9);

    Data[0] = (float*)malloc(sizeof(float)*32*32*3);    // Input
    Data[1] = (float*)malloc(sizeof(float)*32*32*128);  // Conv2d_128_1
    Data[2] = (float*)malloc(sizeof(float)*32*32*128);  // Conv2d_128_2
    Data[3] = (float*)malloc(sizeof(float)*16*16*128);  // MaxPool
    Data[4] = (float*)malloc(sizeof(float)*16*16*256);  // Conv2d_256_1
    Data[5] = (float*)malloc(sizeof(float)*16*16*256);  // Conv2d_256_2
    Data[6] = (float*)malloc(sizeof(float)*8*8*256);    // MaxPool
    Data[7] = (float*)malloc(sizeof(float)*8*8*512);    // Conv2d_512_1
    Data[8] = (float*)malloc(sizeof(float)*8*8*512);    // Conv2d_512_2
    Data[9] = (float*)malloc(sizeof(float)*4*4*512);    // MaxPool
    Data[10] = (float*)malloc(sizeof(float)*1024);      // FC_1
    Data[11] = (float*)malloc(sizeof(float)*1024);      // FC_2
    Data[12] = (float*)malloc(sizeof(float)*10);        // FC_3

    Weight[0] = (int*)malloc(sizeof(int)*3*3*3*128);
    Weight[1] = (int*)malloc(sizeof(int)*3*3*128*128);
    Weight[2] = (int*)malloc(sizeof(int)*3*3*128*256);
    Weight[3] = (int*)malloc(sizeof(int)*3*3*256*256);
    Weight[4] = (int*)malloc(sizeof(int)*3*3*256*512);
    Weight[5] = (int*)malloc(sizeof(int)*3*3*512*512);
    Weight[6] = (int*)malloc(sizeof(int)*8192*1024);
    Weight[7] = (int*)malloc(sizeof(int)*1024*1024);
    Weight[8] = (int*)malloc(sizeof(int)*1024*10);

    Bias[0] = (float*)malloc(sizeof(float)*128);
    Bias[1] = (float*)malloc(sizeof(float)*128);
    Bias[2] = (float*)malloc(sizeof(float)*256);
    Bias[3] = (float*)malloc(sizeof(float)*256);
    Bias[4] = (float*)malloc(sizeof(float)*512);
    Bias[5] = (float*)malloc(sizeof(float)*512);
    Bias[6] = (float*)malloc(sizeof(float)*1024);
    Bias[7] = (float*)malloc(sizeof(float)*1024);
    Bias[8] = (float*)malloc(sizeof(float)*10);

    BN_G[0] = (float*)malloc(sizeof(float)*128);
    BN_B[0] = (float*)malloc(sizeof(float)*128);
    BN_RM[0] = (float*)malloc(sizeof(float)*128);
    BN_RV[0] = (float*)malloc(sizeof(float)*128);
    BN_G[1] = (float*)malloc(sizeof(float)*128);
    BN_B[1] = (float*)malloc(sizeof(float)*128);
    BN_RM[1] = (float*)malloc(sizeof(float)*128);
    BN_RV[1] = (float*)malloc(sizeof(float)*128);
    BN_G[2] = (float*)malloc(sizeof(float)*256);
    BN_B[2] = (float*)malloc(sizeof(float)*256);
    BN_RM[2] = (float*)malloc(sizeof(float)*256);
    BN_RV[2] = (float*)malloc(sizeof(float)*256);
    BN_G[3] = (float*)malloc(sizeof(float)*256);
    BN_B[3] = (float*)malloc(sizeof(float)*256);
    BN_RM[3] = (float*)malloc(sizeof(float)*256);
    BN_RV[3] = (float*)malloc(sizeof(float)*256);
    BN_G[4] = (float*)malloc(sizeof(float)*512);
    BN_B[4] = (float*)malloc(sizeof(float)*512);
    BN_RM[4] = (float*)malloc(sizeof(float)*512);
    BN_RV[4] = (float*)malloc(sizeof(float)*512);
    BN_G[5] = (float*)malloc(sizeof(float)*512);
    BN_B[5] = (float*)malloc(sizeof(float)*512);
    BN_RM[5] = (float*)malloc(sizeof(float)*512);
    BN_RV[5] = (float*)malloc(sizeof(float)*512);
    BN_G[6] = (float*)malloc(sizeof(float)*1024);
    BN_B[6] = (float*)malloc(sizeof(float)*1024);
    BN_RM[6] = (float*)malloc(sizeof(float)*1024);
    BN_RV[6] = (float*)malloc(sizeof(float)*1024);
    BN_G[7] = (float*)malloc(sizeof(float)*1024);
    BN_B[7] = (float*)malloc(sizeof(float)*1024);
    BN_RM[7] = (float*)malloc(sizeof(float)*1024);
    BN_RV[7] = (float*)malloc(sizeof(float)*1024);
    BN_G[8] = (float*)malloc(sizeof(float)*10);
    BN_B[8] = (float*)malloc(sizeof(float)*10);
    BN_RM[8] = (float*)malloc(sizeof(float)*10);
    BN_RV[8] = (float*)malloc(sizeof(float)*10);

#ifdef GETRESULT
    //Result(Score) File
    FILE *fp = fopen("./Score_int.txt", "w");
    fprintf(fp, "-Data\t\t-Pred\t-Ans\n");
#endif

    printf("## Read Data...\n");
    read_data(Original_Data, Class);
    read_weight(Weight, Bias, BN_G, BN_B, BN_RM, BN_RV);
    printf("## Done\n");

    int score = 0;
    int data_num = 1;
    for(int i = 0; i < data_num; i++)
    {
        int idx = 0;
        float max = -10000;
        float Result[10] = { 0, };
        
        printf("Start: %d imgs\n", i);
        memcpy(Data[0], Original_Data + i*32*32*3, 32*32*3*sizeof(float));
        vgg_cifar10_binary(Data, Weight, Bias, BN_G, BN_B, BN_RM, BN_RV, Result);
        
        for(int j = 0; j < 10; j++)
        {
            if(max < Result[j])
            {
                max = Result[j];
                idx = j;
            }
        }
        if(idx == Class[i])
            score++;

#ifdef GETRESULT
        fprintf(fp, "%4d)\t\t[ %d ]\t[ %d ]\t\t", i, idx, Class[i]);
        if(idx == Class[i])
            fprintf(fp, "Right!\n");
        else
            fprintf(fp, "Wrong..\n");
#endif
    }
    //Result
    printf("## Score: %d\n## Accuracy: %.2f%%\n", score, (float)score/data_num*100);

#ifdef GETRESULT
    fprintf(fp, "## Score: %d\n## Accuracy: %.2f%%\n", score, (float)score/data_num*100);
    fclose(fp);
#endif

    free(Original_Data);
    free(Class);
    for(int i = 0; i < 13; i++)
        free(Data[i]);
    for(int i = 0; i < 9; i++)
    {
        free(Weight[i]);
        free(Bias[i]);
        free(BN_G[i]);
        free(BN_B[i]);
        free(BN_RM[i]);
        free(BN_RV[i]);
    }
    free(Data);
    free(Weight);
    free(Bias);
    free(BN_G);
    free(BN_B);
    free(BN_RM);
    free(BN_RV);

    return 0;
}

void read_data(float *Original_Data, int *Class)
{
    FILE *fp1 = fopen("./Data/cifar10-test-rgb-data.txt", "r");
    FILE *fp2 = fopen("./Data/cifar10-test-labels.txt", "r");

    char buf[100] = { 0, };

    for(int i = 0; fgets(buf, 100, fp1) != NULL; i++)
        Original_Data[i] = atof(buf);
    for(int i = 0; fgets(buf, 100, fp2) != NULL; i++)
        Class[i] = atoi(buf);
}

void read_weight(int **Weight, float **Bias, float **BN_G, float **BN_B, float **BN_RM, float **BN_RV)
{
    FILE *fp1 = fopen("./Weight_int/conv2d_3x3x3x128.txt", "r");
    FILE *fp2 = fopen("./Weight_int/conv2d_3x3x128x128.txt", "r");
    FILE *fp3 = fopen("./Weight_int/conv2d_3x3x128x256.txt", "r");
    FILE *fp4 = fopen("./Weight_int/conv2d_3x3x256x256.txt", "r");
    FILE *fp5 = fopen("./Weight_int/conv2d_3x3x256x512.txt", "r");
    FILE *fp6 = fopen("./Weight_int/conv2d_3x3x512x512.txt", "r");
    FILE *fp7 = fopen("./Weight_int/fc_8192x1024.txt", "r");
    FILE *fp8 = fopen("./Weight_int/fc_1024x1024.txt", "r");
    FILE *fp9 = fopen("./Weight_int/fc_1024x10.txt", "r");

    FILE *fp10 = fopen("./Weight_int/bias_conv2d_3x3x3x128.txt", "r");
    FILE *fp11 = fopen("./Weight_int/bias_conv2d_3x3x128x128.txt", "r");
    FILE *fp12 = fopen("./Weight_int/bias_conv2d_3x3x128x256.txt", "r");
    FILE *fp13 = fopen("./Weight_int/bias_conv2d_3x3x256x256.txt", "r");
    FILE *fp14 = fopen("./Weight_int/bias_conv2d_3x3x256x512.txt", "r");
    FILE *fp15 = fopen("./Weight_int/bias_conv2d_3x3x512x512.txt", "r");
    FILE *fp16 = fopen("./Weight_int/bias_fc_8192x1024.txt", "r");
    FILE *fp17 = fopen("./Weight_int/bias_fc_1024x1024.txt", "r");
    FILE *fp18 = fopen("./Weight_int/bias_fc_1024x10.txt", "r");

    FILE *fp19 = fopen("./Weight_int/bn_1_gamma.txt", "r");
    FILE *fp20 = fopen("./Weight_int/bn_1_beta.txt", "r");
    FILE *fp21 = fopen("./Weight_int/bn_1_running_mean.txt", "r");
    FILE *fp22 = fopen("./Weight_int/bn_1_running_var.txt", "r");
    FILE *fp23 = fopen("./Weight_int/bn_2_gamma.txt", "r");
    FILE *fp24 = fopen("./Weight_int/bn_2_beta.txt", "r");
    FILE *fp25 = fopen("./Weight_int/bn_2_running_mean.txt", "r");
    FILE *fp26 = fopen("./Weight_int/bn_2_running_var.txt", "r");
    FILE *fp27 = fopen("./Weight_int/bn_3_gamma.txt", "r");
    FILE *fp28 = fopen("./Weight_int/bn_3_beta.txt", "r");
    FILE *fp29 = fopen("./Weight_int/bn_3_running_mean.txt", "r");
    FILE *fp30 = fopen("./Weight_int/bn_3_running_var.txt", "r");
    FILE *fp31 = fopen("./Weight_int/bn_4_gamma.txt", "r");
    FILE *fp32 = fopen("./Weight_int/bn_4_beta.txt", "r");
    FILE *fp33 = fopen("./Weight_int/bn_4_running_mean.txt", "r");
    FILE *fp34 = fopen("./Weight_int/bn_4_running_var.txt", "r");
    FILE *fp35 = fopen("./Weight_int/bn_5_gamma.txt", "r");
    FILE *fp36 = fopen("./Weight_int/bn_5_beta.txt", "r");
    FILE *fp37 = fopen("./Weight_int/bn_5_running_mean.txt", "r");
    FILE *fp38 = fopen("./Weight_int/bn_5_running_var.txt", "r");
    FILE *fp39 = fopen("./Weight_int/bn_6_gamma.txt", "r");
    FILE *fp40 = fopen("./Weight_int/bn_6_beta.txt", "r");
    FILE *fp41 = fopen("./Weight_int/bn_6_running_mean.txt", "r");
    FILE *fp42 = fopen("./Weight_int/bn_6_running_var.txt", "r");
    FILE *fp43 = fopen("./Weight_int/bn_7_gamma.txt", "r");
    FILE *fp44 = fopen("./Weight_int/bn_7_beta.txt", "r");
    FILE *fp45 = fopen("./Weight_int/bn_7_running_mean.txt", "r");
    FILE *fp46 = fopen("./Weight_int/bn_7_running_var.txt", "r");
    FILE *fp47 = fopen("./Weight_int/bn_8_gamma.txt", "r");
    FILE *fp48 = fopen("./Weight_int/bn_8_beta.txt", "r");
    FILE *fp49 = fopen("./Weight_int/bn_8_running_mean.txt", "r");
    FILE *fp50 = fopen("./Weight_int/bn_8_running_var.txt", "r");
    FILE *fp51 = fopen("./Weight_int/bn_9_gamma.txt", "r");
    FILE *fp52 = fopen("./Weight_int/bn_9_beta.txt", "r");
    FILE *fp53 = fopen("./Weight_int/bn_9_running_mean.txt", "r");
    FILE *fp54 = fopen("./Weight_int/bn_9_running_var.txt", "r");

    if(fp1 == NULL)
        printf("Error: No Weight File\n");

    char s[100] = { 0, };

    for(int c = 0; c < 128; c++)
        for(int c_in = 0; c_in < 3; c_in++)
            for(int h = 0; h < 3; h++)
                for(int w = 0; w < 3; w++)
                {
                    fgets(s, 100, fp1);
                    Weight[0][c*3*3*3 + c_in*3*3 + h*3 + w] = atoi(s);
                }
    
    for(int c = 0; c < 128; c++)
        for(int c_in = 0; c_in < 128; c_in++)
            for(int h = 0; h < 3; h++)
                for(int w = 0; w < 3; w++)
                {
                    fgets(s, 100, fp2);
                    Weight[1][c*3*3*128 + c_in*3*3 + h*3 + w] = atoi(s);
                }

    for(int c = 0; c < 128; c++)
    {
        fgets(s, 100, fp10);
        Bias[0][c] = atof(s);

        fgets(s, 100, fp11);
        Bias[1][c] = atof(s);

        fgets(s, 100, fp19);
        BN_G[0][c] = atof(s);
        fgets(s, 100, fp20);
        BN_B[0][c] = atof(s);
        fgets(s, 100, fp21);
        BN_RM[0][c] = atof(s);
        fgets(s, 100, fp22);
        BN_RV[0][c] = atof(s);

        fgets(s, 100, fp23);
        BN_G[1][c] = atof(s);
        fgets(s, 100, fp24);
        BN_B[1][c] = atof(s);
        fgets(s, 100, fp25);
        BN_RM[1][c] = atof(s);
        fgets(s, 100, fp26);
        BN_RV[1][c] = atof(s);
    }

    for(int c = 0; c < 256; c++)
        for(int c_in = 0; c_in < 128; c_in++)
            for(int h = 0; h < 3; h++)
                for(int w = 0; w < 3; w++)
                {
                    fgets(s, 100, fp3);
                    Weight[2][c*3*3*128 + c_in*3*3 + h*3 + w] = atoi(s);
                }
    
    for(int c = 0; c < 256; c++)
        for(int c_in = 0; c_in < 256; c_in++)
            for(int h = 0; h < 3; h++)
                for(int w = 0; w < 3; w++)
                {
                    fgets(s, 100, fp4);
                    Weight[3][c*3*3*256 + c_in*3*3 + h*3 + w] = atoi(s);
                }

    for(int c = 0; c < 256; c++)
    {
        fgets(s, 100, fp12);
        Bias[2][c] = atof(s);

        fgets(s, 100, fp13);
        Bias[3][c] = atof(s);

        fgets(s, 100, fp27);
        BN_G[2][c] = atof(s);
        fgets(s, 100, fp28);
        BN_B[2][c] = atof(s);
        fgets(s, 100, fp29);
        BN_RM[2][c] = atof(s);
        fgets(s, 100, fp30);
        BN_RV[2][c] = atof(s);

        fgets(s, 100, fp31);
        BN_G[3][c] = atof(s);
        fgets(s, 100, fp32);
        BN_B[3][c] = atof(s);
        fgets(s, 100, fp33);
        BN_RM[3][c] = atof(s);
        fgets(s, 100, fp34);
        BN_RV[3][c] = atof(s);
    }

    for(int c = 0; c < 512; c++)
        for(int c_in = 0; c_in < 256; c_in++)
            for(int h = 0; h < 3; h++)
                for(int w = 0; w < 3; w++)
                {
                    fgets(s, 100, fp5);
                    Weight[4][c*3*3*256 + c_in*3*3 + h*3 + w] = atoi(s);
                }
    
    for(int c = 0; c < 512; c++)
        for(int c_in = 0; c_in < 512; c_in++)
            for(int h = 0; h < 3; h++)
                for(int w = 0; w < 3; w++)
                {
                    fgets(s, 100, fp6);
                    Weight[5][c*3*3*512 + c_in*3*3 + h*3 + w] = atoi(s);
                }

    for(int c = 0; c < 512; c++)
    {
        fgets(s, 100, fp14);
        Bias[4][c] = atof(s);

        fgets(s, 100, fp15);
        Bias[5][c] = atof(s);

        fgets(s, 100, fp35);
        BN_G[4][c] = atof(s);
        fgets(s, 100, fp36);
        BN_B[4][c] = atof(s);
        fgets(s, 100, fp37);
        BN_RM[4][c] = atof(s);
        fgets(s, 100, fp38);
        BN_RV[4][c] = atof(s);

        fgets(s, 100, fp39);
        BN_G[5][c] = atof(s);
        fgets(s, 100, fp40);
        BN_B[5][c] = atof(s);
        fgets(s, 100, fp41);
        BN_RM[5][c] = atof(s);
        fgets(s, 100, fp42);
        BN_RV[5][c] = atof(s);
    }

    for(int c = 0; c < 1024; c++)
        for(int h = 0; h < 8192; h++)
        {
            fgets(s, 100, fp7);
            Weight[6][c*8192 + h] = atoi(s);
        }

    for(int c = 0; c < 1024; c++)
        for(int h = 0; h < 1024; h++)
        {
            fgets(s, 100, fp8);
            Weight[7][c*1024 + h] = atoi(s);
        }

    for(int c = 0; c < 10; c++)
        for(int h = 0; h < 1024; h++)
        {
            fgets(s, 100, fp9);
            Weight[8][c*1024 + h] = atoi(s);
        }

    for(int c = 0; c < 1024; c++)
    {
        fgets(s, 100, fp16);
        Bias[6][c] = atof(s);

        fgets(s, 100, fp17);
        Bias[7][c] = atof(s);

        fgets(s, 100, fp43);
        BN_G[6][c] = atof(s);
        fgets(s, 100, fp44);
        BN_B[6][c] = atof(s);
        fgets(s, 100, fp45);
        BN_RM[6][c] = atof(s);
        fgets(s, 100, fp46);
        BN_RV[6][c] = atof(s);
        
        fgets(s, 100, fp47);
        BN_G[7][c] = atof(s);
        fgets(s, 100, fp48);
        BN_B[7][c] = atof(s);
        fgets(s, 100, fp49);
        BN_RM[7][c] = atof(s);
        fgets(s, 100, fp50);
        BN_RV[7][c] = atof(s);
    }

    for(int c = 0; c < 10; c++)
    {
        fgets(s, 100, fp18);
        Bias[8][c] = atof(s);

        fgets(s, 100, fp51);
        BN_G[8][c] = atof(s);
        fgets(s, 100, fp52);
        BN_B[8][c] = atof(s);
        fgets(s, 100, fp53);
        BN_RM[8][c] = atof(s);
        fgets(s, 100, fp54);
        BN_RV[8][c] = atof(s);
    }

    fclose(fp1); fclose(fp2); fclose(fp3); fclose(fp4); fclose(fp5); fclose(fp6);
    fclose(fp7); fclose(fp8); fclose(fp9); fclose(fp10); fclose(fp11); fclose(fp12);
    fclose(fp13); fclose(fp14); fclose(fp15); fclose(fp16); fclose(fp17); fclose(fp18);
    fclose(fp19); fclose(fp20); fclose(fp21); fclose(fp22); fclose(fp23); fclose(fp24);
    fclose(fp25); fclose(fp26); fclose(fp27); fclose(fp28); fclose(fp29); fclose(fp30);
    fclose(fp31); fclose(fp32); fclose(fp33); fclose(fp34); fclose(fp35); fclose(fp36);
    fclose(fp37); fclose(fp38); fclose(fp39); fclose(fp40); fclose(fp41); fclose(fp42);
    fclose(fp43); fclose(fp44); fclose(fp45); fclose(fp46); fclose(fp47); fclose(fp48);
    fclose(fp49); fclose(fp50); fclose(fp51); fclose(fp52); fclose(fp53); fclose(fp54);
}
