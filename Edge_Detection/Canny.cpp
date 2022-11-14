#include "Func.h"
/////////////////////////////////////
//
//  GPU Computing Project - Canny Edge Detection
//  GPU_Func.cu외 모든 Code는 수정하지 말 것
//  불가피하게 수정해야 할 경우 조교에게 문의
//
/////////////////////////////////////

int main()
{
	FILE* fp = fopen("test_file.bmp", "rb");
	
    uint8_t test[200] = {0};
	fread(test, 200, 1, fp);
	
	fseek(fp, 0, SEEK_END);
	int len = 4096*4096;//ftell(fp);
	int width = 4096;
	int height = 4096;
	fseek(fp, 0, SEEK_SET); //go to beg.

    uint8_t min = 0;
    uint8_t max = 255;

	uint8_t* buf = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* gray = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* gaussian = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* sobel = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* sobel_angle = (uint8_t*)malloc((len) / 3); //malloc buffer
	uint8_t* suppression = (uint8_t*)malloc(len); //malloc buffer
	uint8_t* hysteresis = (uint8_t*)malloc(len); //malloc buffer

	memset(buf, 0, len);
	memset(gray, 0, len);
	memset(gaussian, 0, len);
	memset(sobel, 0, len);
	memset(sobel_angle, 0, len / 3);
	memset(suppression, 0, len);
	memset(hysteresis, 0, len);
///////////////////////////////Image Read//////////////////////////////
	fread(buf, len, 1, fp); //read into buffer
	len -= 2;
	for (int i = 0; i < test[10]; i++) {
		gray[i] = buf[i];
		gaussian[i] = buf[i];
		sobel[i] = buf[i];
		suppression[i] = buf[i];
		hysteresis[i] = buf[i];
	}
	for (int i = 18; i < 22; i++)
		width += test[i] * pow(256, i-18);
	for (int i = 22; i < 26; i++)
	height += test[i] * pow(256, i-22);
////////////////////////////GrayScale//////////////////////////////////
	Grayscale(buf, gray, 0, len);
////////////////////////////Noise_Reduction///////////////////////////////////////
    Noise_Reduction(width,height,gray, gaussian);
//////////////////////////Intensity_Gradient////////////////////////////////////
	Intensity_Gradient(width, height, gaussian, sobel, sobel_angle);
//////////////////////////Non-maximum_Suppression//////////////////////////////////////
	Non_maximum_Suppression(width, height, sobel_angle, sobel, suppression,min,max);
/////////////////////////Hysteresis Thresholding/////////////////////////////
	Hysteresis_Thresholding(width, height, suppression, hysteresis, min, max);
//////////////////////////////////////////////////////////////////
	
    fclose(fp);
	free(buf);
	free(gray);
	free(gaussian);
	free(sobel);
	free(sobel_angle);
	free(suppression);
	free(hysteresis);
	return 0;
}
