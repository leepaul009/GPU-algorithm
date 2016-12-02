// parallel-knn, 2016/11/28C including Serial,GPU-B-kNN
#pragma once

#define _MOBILE 1
//#define _FPRINTF 1
//#define _PRINT2 1
//#define S_KNN 1
#define B_KNN 1
//#define R_KNN 1
#include "../common.h"

#ifdef _MOBILE
#include <CL/cl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/opencl.h>
#endif


/****************************************************/
using namespace std;

#define MAX_BUFFER 2048
#define NUM_KNN_K 16 //大于4的整数,4的倍数
#define WRAP 64
#define SPLIT 64
#define _BITS 5
#define _RADIX (1 << _BITS)
#define _PASS 7
#define _CLASS 9
#define _MULTIPLY 1024 //综合考虑WRAP和SPLIT
#define _KSPLIT 8
#define MAX_SOURCE_SIZE (0x100000)

#define B_KEY 2048


#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
template <class T> static inline void swapVal(T& x, T& y) { T t=x; x=y; y=t; }


typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned short htype;

class CLKnn
{
public:
	CLKnn(cl_context Context, cl_device_id Device, cl_command_queue CommandQueue);
	~CLKnn();

	void init_variable_from_prob(const char *filename, uint *input_size, int lmt);
	//lmt can effect input_size
	void read_problem(const char *filename, float *h_input, uint input_size, int cha);
	void PreProcess(void);
	void CLDistance(uint idx);
#ifdef R_KNN
	void Sort();
		void Histogram(uint pass);
		void ScanHistogram();
		void Reorder(uint pass);
#endif
	void Predict(uint idx);
	void calcuPrediction(uint idx, float threshold);
	void PrintPrediction(const char *filename);

	void Merge(uint length);
	void bSort(uint length);
	void bSort2(uint length);
	void BitonicSelect(uint length);
	void BitonicSort2(void);
	void BitonicSort(void);
#ifdef S_KNN
	void serialKnn(uint idx);
		float *dist;
		uchar *label;
#endif
	cl_context Context;
	cl_device_id Device;
	cl_command_queue CommandQueue;
	cl_program Program;

	int nFeats;
	int split; // the number of splits in one training line
	int radix;
	int radixbits;
	int kSplit;
	int kValue;
	int nClass;


	uint nTests;
	uint nTrains;
	uint nTests_rounded;
	uint nTests_inWork;
	uint nTrains_rounded;

	float *h_test;
	float *h_train;


	htype *h_ClassHist;

	cl_mem d_test;
	cl_mem d_train;
	cl_mem d_inDist; //float
	cl_mem d_inLabel; //uchar
#ifdef R_KNN
	cl_mem d_outDist; //float
		cl_mem d_outLabel; //uchar
#endif
	cl_mem d_Histograms; //uint
	cl_mem d_globsum;
	cl_mem d_gsum;
	cl_mem d_ClassHist; //histogram of class



	cl_ulong localMem;// for checking of local memory size

	cl_kernel ckDist;
	cl_kernel ckHistogram;  // compute histograms
	cl_kernel ckScanHistogram; // scan local histogram
	cl_kernel ckScanHistogram2;
	cl_kernel ckPasteHistogram; // paste local histograms
	cl_kernel ckReorder; // final reordering
	cl_kernel ckClassifyHist;
	cl_kernel ckClassifyTable;

	cl_kernel bitonicSort;
	cl_kernel bitonicSort2;
	cl_kernel mergeSort;
	cl_kernel bitonicSelect;
	//cl_mem d_bSelectDist;
	//cl_mem d_bSelectLabel;

	int true_predict_num;
	int num_after_reject;
	float predictive_rate;

	int TP[_CLASS]; // true positive
	int FP[_CLASS]; // false positive
	int FN[_CLASS]; // false negative
	int TN[_CLASS]; // true negative
	int P[_CLASS];
	int R[_CLASS];


	long exetime;
};


