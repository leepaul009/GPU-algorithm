// parallel-svm-knn, 2016/11/28
#pragma once

#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define _MOBILE 1
//#define PREDICT_P 1 //附带概率的svm
//#define P_KNNSVM 1 //apply GPU computing to SVM


#include "../common.h"
#include "svm.h"

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
#define SVM_KEY 16


#define WRAP 64
#define SPLIT 64
#define _BITS 5
#define _RADIX (1 << _BITS)
#define _PASS 7
#define _CLASS 9
#define _MULTIPLY 64
#define _KSPLIT 8



#define MAX_SOURCE_SIZE (0x100000)
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned short htype;

#include <iterator>
__inline__ std::string loadProgram2(std::string input)
{
	std::ifstream stream(input.c_str());
	if (!stream.is_open()) {
		printf("Cannot open input file\n");
		exit(1);
	}
	return std::string( std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
}

__inline__ long getTimeNsec()
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (long) now.tv_sec*1000000000LL + now.tv_nsec;
}


class CLKnn
{
public:
	CLKnn(cl_context Context, cl_device_id Device, cl_command_queue CommandQueue);
	~CLKnn();

	void init_variable_from_prob(const char *filename, uint *input_size, int lmt);
	void read_problem(const char *filename, float *h_input, uint input_size, int cha);

	void PreProcess(void);

	void CLDistance(uint idx);
	void Sort();
	void Histogram(uint pass);
	void ScanHistogram();
	void Reorder(uint pass);

	void Predict(uint idx);
	void calcuPrediction(uint idx, uint idy, float threshold);

	void initKnnSvm(void);
	void knnSvm(uint idx, uint idy);
	void parse_command_line(int argc, char **argv);
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_node *x_space;
	struct svm_node_Q *kv_space;
	struct svm_node *test_nodes;

	cl_context Context;
	cl_device_id Device;
	cl_command_queue CommandQueue;
	cl_program Program;

	int nFeats;
	int nPureFeats;
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
	htype *predict_list;

	htype *h_ClassHist;

	float *h_A;
	float *h_B;
	SNQfloat *h_C;

	cl_mem d_test;
	cl_mem d_train;


	cl_mem d_inDist; //float
	cl_mem d_inLabel; //uchar
	cl_mem d_inIdx;
	cl_mem d_outDist; //float
	cl_mem d_outLabel; //uchar
	cl_mem d_outIdx;
	cl_mem d_Histograms; //uint
	cl_mem d_globsum;
	cl_mem d_gsum;
	cl_mem d_ClassHist; //histogram of class

	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

	cl_ulong localMem;// for checking of local memory size

	cl_kernel ckDist;
	cl_kernel ckHistogram;  // compute histograms
	cl_kernel ckScanHistogram; // scan local histogram
	cl_kernel ckScanHistogram2;
	cl_kernel ckPasteHistogram; // paste local histograms
	cl_kernel ckReorder; // final reordering
	cl_kernel ckClassifyHist;
	cl_kernel ckClassifyTable;
	cl_kernel ckKernelComputation;

	int use_svm;
	int svm_true_predict_num;

	int SVM_TP[_CLASS]; // true positive
	int SVM_FP[_CLASS]; // false positive
	int SVM_FN[_CLASS]; // false negative
	int SVM_TN[_CLASS]; // true negative
	int SVM_P[_CLASS];
	int SVM_R[_CLASS];

	FILE *fp_comp;


	long exetime;
	uint64_t uT;
	//float dist_time, histo_time, scan_time, reorder_time, sort_time, predict_time;
};


