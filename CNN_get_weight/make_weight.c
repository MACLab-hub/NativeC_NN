#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

void read_weight(float *W1, float *W2, float *W3, float *W4, float *W5, float *W6, float *W7, float *W8, float *W9){
    FILE * fp1 = fopen("./conv2d_3x3x1x128.txt","r");
    FILE * fp2 = fopen("./conv2d_3x3x128x128.txt","r");
    FILE * fp3 = fopen("./conv2d_3x3x128x256.txt","r");
    FILE * fp4 = fopen("./conv2d_3x3x256x256.txt","r");
    FILE * fp5 = fopen("./conv2d_3x3x256x512.txt","r");
    FILE * fp6 = fopen("./conv2d_3x3x512x512.txt","r");
    FILE * fp7 = fopen("./fc_8192x1024.txt","r");
    FILE * fp8 = fopen("./fc_1024x1024.txt","r");
    FILE * fp9 = fopen("./fc_1024x10.txt","r");

    char s[100] = {0};
    for(int h=0;h<3;h++){
        for(int w=0;w<3;w++){
            for(int c=0;c<128;c++){
                fgets(s,100,fp1);
                W1[c*9 + h * 3 + w] = atof(s);
            }
        }
    }
    for(int h=0;h<3;h++){
        for(int w=0;w<3;w++){
            for(int c_in=0;c_in<128;c_in++){
                for(int c=0;c<128;c++){
                    fgets(s,100,fp2);
                    W2[c*128*9 + c_in*9 + h * 3 + w] = atof(s);
                }
            }
        }
    }
    for(int h=0;h<3;h++){
        for(int w=0;w<3;w++){
            for(int c_in=0;c_in<128;c_in++){
                for(int c=0;c<256;c++){
                    fgets(s,100,fp3);
                    W3[c*128*9 + c_in*9 + h * 3 + w] = atof(s);
                }
            }
        }
    }
    for(int h=0;h<3;h++){
        for(int w=0;w<3;w++){
            for(int c_in=0;c_in<256;c_in++){
                for(int c=0;c<256;c++){
                    fgets(s,100,fp4);
                    W4[c*256*9 + c_in*9 + h * 3 + w] = atof(s);
                }
            }
        }
    }
    for(int h=0;h<3;h++){
        for(int w=0;w<3;w++){
            for(int c_in=0;c_in<256;c_in++){
                for(int c=0;c<512;c++){
                    fgets(s,100,fp5);
                    W5[c*256*9 + c_in*9 + h * 3 + w] = atof(s);
                }
            }
        }
    }
    for(int h=0;h<3;h++){
        for(int w=0;w<3;w++){
            for(int c_in=0;c_in<512;c_in++){
                for(int c=0;c<512;c++){
                    fgets(s,100,fp6);
                    W6[c*512*9 + c_in*9 + h * 3 + w] = atof(s);
                }
            }
        }
    }
    for(int h=0;h<8192;h++){
        for(int r=0;r<1024;r++){
                fgets(s,100,fp7);
                W7[r*8192 + h] = atof(s);
        }
    }
    for(int h=0;h<1024;h++){
        for(int r=0;r<1024;r++){
                fgets(s,100,fp8);
                W8[r*1024 + h] = atof(s);

        }
    }
    for(int h=0;h<1024;h++){
        for(int r=0;r<10;r++){
                fgets(s,100,fp9);
                W9[r*1024 + h] = atof(s);
        }
    }
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);
    fclose(fp5);
    fclose(fp6);
    fclose(fp7);
    fclose(fp8);
    fclose(fp9);
}

void save_weight(float *W1, float *W2, float *W3, float *W4, float *W5, float *W6, float *W7, float *W8, float *W9){
    char s[100] = {0};
    FILE * fp1 = fopen("../Weight_float/conv2d_3x3x1x128.txt","w");
    FILE * fp2 = fopen("../Weight_float/conv2d_3x3x128x128.txt","w");
    FILE * fp3 = fopen("../Weight_float/conv2d_3x3x128x256.txt","w");
    FILE * fp4 = fopen("../Weight_float/conv2d_3x3x256x256.txt","w");
    FILE * fp5 = fopen("../Weight_float/conv2d_3x3x256x512.txt","w");
    FILE * fp6 = fopen("../Weight_float/conv2d_3x3x512x512.txt","w");
    FILE * fp7 = fopen("../Weight_float/fc_8192x1024.txt","w");
    FILE * fp8 = fopen("../Weight_float/fc_1024x1024.txt","w");
    FILE * fp9 = fopen("../Weight_float/fc_1024x10.txt","w");

    for(int c=0;c<128;c++){
        for(int h=0;h<3;h++){
            for(int w=0;w<3;w++){
                sprintf(s,"%f\n",W1[c*9+h*3+w]);
                fputs(s,fp1);
            }
        }
    }
    
    for(int c=0;c<128;c++){
        for(int c_in=0;c_in<128;c_in++){
            for(int h=0;h<3;h++){
                for(int w=0;w<3;w++){
                    sprintf(s,"%f\n",W2[c*128*9+c_in*9+h*3+w]);
                    fputs(s,fp2);
                }   
            }
        }
    }

    for(int c=0;c<256;c++){
        for(int c_in=0;c_in<128;c_in++){
            for(int h=0;h<3;h++){
                for(int w=0;w<3;w++){
                    sprintf(s,"%f\n",W3[c*128*9+c_in*9+h*3+w]);
                    fputs(s,fp3);
                }   
            }
        }
    }
    
    for(int c=0;c<256;c++){
        for(int c_in=0;c_in<256;c_in++){
            for(int h=0;h<3;h++){
                for(int w=0;w<3;w++){
                    sprintf(s,"%f\n",W4[c*256*9+c_in*9+h*3+w]);
                    fputs(s,fp4);
                }   
            }
        }
    }

    for(int c=0;c<512;c++){
        for(int c_in=0;c_in<256;c_in++){
            for(int h=0;h<3;h++){
                for(int w=0;w<3;w++){
                    sprintf(s,"%f\n",W5[c*256*9+c_in*9+h*3+w]);
                    fputs(s,fp5);
                }   
            }
        }
    }
    
    for(int c=0;c<512;c++){
        for(int c_in=0;c_in<512;c_in++){
            for(int h=0;h<3;h++){
                for(int w=0;w<3;w++){
                    sprintf(s,"%f\n",W6[c*512*9+c_in*9+h*3+w]);
                    fputs(s,fp6);
                }   
            }
        }
    }
    




    for(int r=0;r<1024;r++){
        for(int h=0;h<8192;h++){
                sprintf(s,"%f\n",W7[r*8192+h]);
                fputs(s,fp7);
        }
    }
    
    for(int r=0;r<1024;r++){
        for(int h=0;h<1024;h++){
                sprintf(s,"%f\n",W8[r*1024+h]);
                fputs(s,fp8);
        }
    }
    
    for(int r=0;r<10;r++){
        for(int h=0;h<1024;h++){
                sprintf(s,"%f\n",W9[r*1024+h]);
                fputs(s,fp9);
        }
    }
    
    fclose(fp1);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);
    fclose(fp5);
    fclose(fp6);
    fclose(fp7);
    fclose(fp8);
    fclose(fp9);
}

int main()
{
    float *W1=NULL;
    float *W2=NULL;
    float *W3=NULL;
    float *W4=NULL;
    float *W5=NULL;
    float *W6=NULL;
    float *W7=NULL;
    float *W8=NULL;
    float *W9=NULL;

    W1 = (float*)malloc(sizeof(float)*3*3*1*128);
    W2 = (float*)malloc(sizeof(float)*3*3*128*128);
    W3 = (float*)malloc(sizeof(float)*3*3*128*256);
    W4 = (float*)malloc(sizeof(float)*3*3*256*256);
    W5 = (float*)malloc(sizeof(float)*3*3*256*512);
    W6 = (float*)malloc(sizeof(float)*3*3*512*512);
    W7 = (float*)malloc(sizeof(float)*8192*1024);
    W8 = (float*)malloc(sizeof(float)*1024*1024);
    W9 = (float*)malloc(sizeof(float)*1024*10);

    read_weight(W1,W2,W3,W4,W5,W6,W7,W8,W9);

    save_weight(W1,W2,W3,W4,W5,W6,W7,W8,W9);

    free(W1);
    free(W2);
    free(W3);
    free(W4);
    free(W5);
    free(W6);
    free(W7);
    free(W8);
    free(W9);
    return 0;
}
