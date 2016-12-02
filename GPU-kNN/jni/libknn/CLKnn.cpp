// parallel-knn, 2016/11/28D including Serial,GPU-B-kNN
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <string>
#include <fstream>
#include <iostream>
#include <iterator>

#include "CLKnn.h"

__inline__ std::string load_clknn_Prog(std::string input)
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


using namespace std;

static char *line = NULL;
static int max_line_len;
static char* readline(FILE *input)
{
	int len;
	if(fgets(line,max_line_len,input) == NULL) return NULL;
	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


CLKnn::CLKnn(cl_context GPUContext, cl_device_id dev, cl_command_queue CommandQue):
		Context(GPUContext), Device(dev), CommandQueue(CommandQue)
{
/*** get input data ***/
#ifdef _MOBILE
	char input_train_file_name[1024] = "/storage/sdcard0/libsvm/b9f3t.scale";
	char input_test_file_name[1024] = "/storage/sdcard0/libsvm/b9f3p.scale";
	//char predict_result_file_name[1024] = "/storage/sdcard0/libsvm/predict_result.csv";
#else
	char input_train_file_name[1024] = "../0.data/data/b9f3t.s";
	char input_test_file_name[1024] = "../0.data/data/b9f3p.s";
#ifdef _FPRINTF
	char predict_result_file_name[1024] = "result/predict_result.csv";
#endif
#endif
	//同时,为h_train初始化,为nFeats,h_train和nTests赋值
	init_variable_from_prob(input_train_file_name, &nTrains, 0); //0:lmt
	init_variable_from_prob(input_test_file_name, &nTests, 6848);

	cl_int err;
	d_train	= clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*nTrains*nFeats, NULL, &err);
	if(err != CL_SUCCESS) debug("d_train failed to create!\n");
	h_train = (float*)clEnqueueMapBuffer(CommandQueue, d_train, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float)*nTrains*nFeats, 0, NULL, NULL, &err);
	if(err != CL_SUCCESS) debug("Cannot map from d_train to h_train");

	d_test	= clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*nTests*nFeats, NULL, &err);
	if(err != CL_SUCCESS) debug("d_test failed to create!\n");
	h_test = (float*)clEnqueueMapBuffer(CommandQueue, d_test, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float)*nTests*nFeats, 0, NULL, NULL, &err);
	if(err != CL_SUCCESS) debug("Cannot map from d_test to h_test");

	read_problem(input_train_file_name, h_train, nTrains, 1); //1:reverse storage
	read_problem(input_test_file_name, h_test, nTests, 0);

	err = clEnqueueUnmapMemObject(CommandQueue, d_train, h_train, 0, NULL, NULL);
	if(err != CL_SUCCESS) debug("d_train failed to unmap!\n");
	err = clEnqueueUnmapMemObject(CommandQueue, d_test, h_test, 0, NULL, NULL);
	if(err != CL_SUCCESS) debug("d_test failed to unmap!\n");

	/*** init host data ***/
	split = SPLIT;
	radix = _RADIX;
	radixbits = _BITS;
	kValue = NUM_KNN_K;
	kSplit = NUM_KNN_K/4; //表示 kernel class_hist 的线程数目
	nClass = _CLASS;
	kValue = (int)ceil((float)kValue / (float)kSplit)*kSplit;
	nTrains_rounded = ((uint)ceil((double)nTrains / (double)_MULTIPLY))*_MULTIPLY;
#ifdef B_KNN
	nTrains_rounded = 57344;
#endif
	nTests_inWork = 4; //满足 nTests_inWork*split 是WRAP的复数
	h_ClassHist = new htype[nTests_inWork*kSplit*nClass];

	//保证histogram function的线程数 大于 可操作的数据块
	clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
	debug("Testing data has %d rows, and training data after width-fixing has %d rows\n", nTests, nTrains_rounded);

	size_t max_group_size;
	clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_group_size, NULL);
	debug("max_group_size is %d...",max_group_size);
	/*** create kernel and MEM object ***/
	PreProcess();


	/*** initiate evaluation factor ***/
	true_predict_num = 0;
	num_after_reject = 0;
/*	for(int i=0; i<nClass; i++)
	{
		TP[i] = 0; FP[i] = 0; FN[i] = 0; TN[i] = 0; P[i] = 0; R[i] = 0;
	}*/
#ifdef S_KNN
	h_train = (float*)clEnqueueMapBuffer(CommandQueue, d_train, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*nTrains*nFeats, 0, NULL, NULL, &err);
	if(err != CL_SUCCESS) debug("Cannot map from d_train to h_train");
	h_test = (float*)clEnqueueMapBuffer(CommandQueue, d_test, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*nTests*nFeats, 0, NULL, NULL, &err);
	if(err != CL_SUCCESS) debug("Cannot map from d_test to h_test");

	dist = new float[nTrains];
	label = new uchar[nTrains];

	debug("%d loop...",nTests/nTests_inWork);
	for(uint i=0; i<nTests/nTests_inWork; i++){
		debug("%d",i);
		serialKnn(i);
	}

	err = clEnqueueUnmapMemObject(CommandQueue, d_train, h_train, 0, NULL, NULL);
	if(err != CL_SUCCESS) debug("d_train failed to unmap!\n");
	err = clEnqueueUnmapMemObject(CommandQueue, d_test, h_test, 0, NULL, NULL);
	if(err != CL_SUCCESS) debug("d_test failed to unmap!\n");
#else
#ifdef _PRINT2
	FILE *fp2 = fopen("result/res03.csv","wt");
#endif
	/*** get executing time ***/
	//exetime = getTimeNsec();
	debug("%d...",nTests/nTests_inWork);
	for(uint i=0; i<nTests/nTests_inWork; i++)
	{
		debug("%d",i);
		CLDistance(i);
#ifdef B_KNN
		//BitonicSort();
		BitonicSort2();
#endif
#ifdef R_KNN
		/* radix sort */
		Sort();
#endif
#ifdef _PRINT2
		float *h_inDist = new float[nTests_inWork*nTrains_rounded];

		err = clEnqueueReadBuffer(CommandQueue, d_inDist, CL_TRUE, 0, sizeof(float)*nTests_inWork*nTrains_rounded, h_inDist, 0, NULL, NULL);
		if(err != CL_SUCCESS) debug("Can not read buffer of d_ClassHist\n");

		uchar *h_inLabel = new uchar[nTests_inWork*nTrains_rounded];
		err = clEnqueueReadBuffer(CommandQueue, d_inLabel, CL_TRUE, 0, sizeof(uchar)*nTests_inWork*nTrains_rounded, h_inLabel, 0, NULL, NULL);
		if(err != CL_SUCCESS) debug("Can not read buffer of d_ClassHist\n");

		//for(int ix=0; ix<16; ix++) fprintf(fp2, "%f-%d,", h_inDist[ix], h_inLabel[ix]);
		for(uint ix=0; ix<nTrains_rounded; ix++) fprintf(fp2, "%f\n", h_inDist[ix]);
		//fprintf(fp2, "\n");
		free(h_inDist);
		free(h_inLabel);
#endif
		Predict(i);
		err = clEnqueueReadBuffer(CommandQueue, d_ClassHist, CL_TRUE, 0, sizeof(htype)*nTests_inWork*kSplit*nClass, h_ClassHist, 0, NULL, NULL);
		if(err != CL_SUCCESS) debug("Can not read buffer of d_ClassHist\n");
		calcuPrediction(i, 0.0);
#ifdef _PRINT2
		//for(int ix=0; ix<kSplit*nClass; ix++) fprintf(fp2, "%d,", h_ClassHist[ix]);
		//fprintf(fp2, "\n");
#endif
	}
	//exetime = getTimeNsec() - exetime;
#ifdef _PRINT2
	fclose(fp2);
#endif
#endif
	//PrintPrediction(predict_result_file_name);
	float rate;
	rate = (float)true_predict_num/(float)num_after_reject;
	debug("well predicted number: %d, overall accuracy: %f\n", true_predict_num, rate);
	//debug("final computing time is %f\n", (float)exetime/1000000000.0);
}

CLKnn::~CLKnn()
{
#ifdef S_KNN
	free(dist);
	free(label);
#endif
	delete[] h_ClassHist;

	clReleaseMemObject(d_test);
	clReleaseMemObject(d_train);

	clReleaseMemObject(d_inDist);
	clReleaseMemObject(d_inLabel);
#ifdef R_KNN
	clReleaseMemObject(d_outDist);
	clReleaseMemObject(d_outLabel);

	clReleaseMemObject(d_Histograms);
	clReleaseMemObject(d_globsum);
	clReleaseMemObject(d_gsum);
#endif
	clReleaseMemObject(d_ClassHist);

	clReleaseKernel(ckDist);
#ifdef R_KNN
	clReleaseKernel(ckHistogram);
	clReleaseKernel(ckScanHistogram);
	clReleaseKernel(ckScanHistogram2);
	clReleaseKernel(ckPasteHistogram);
	clReleaseKernel(ckReorder);
#endif
	clReleaseKernel(ckClassifyHist);
#ifdef B_KNN
	clReleaseKernel(bitonicSort);
	clReleaseKernel(bitonicSort2);
	clReleaseKernel(mergeSort);
	clReleaseKernel(bitonicSelect);
#endif
	clReleaseProgram(Program);
	clReleaseCommandQueue(CommandQueue);
	clReleaseContext(Context);
}

void CLKnn::PreProcess(void)
{
	cl_int err;
	/*** create program with source & build program ***/
	string fileDir;
#ifdef _MOBILE
	fileDir.append("/storage/sdcard0/libsvm/kernel_pknn.cl");
#else
	fileDir.append("kernel_pknn.cl");
#endif
	string kernelSource = load_clknn_Prog(fileDir);
	const char* kernelSourceChar = kernelSource.c_str();
	debug("create program with source!");
	Program = clCreateProgramWithSource(Context, 1, &kernelSourceChar, NULL, &err);
	if(err != CL_SUCCESS)debug("clCreateProgramWithSource failed\n");
	err = clBuildProgram(Program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS)debug("create program failed\n");

	/*** create kernel ***/
	debug("create kernel...");
	ckDist 				= clCreateKernel(Program, "Dist", &err);
#ifdef R_KNN
	ckHistogram 		= clCreateKernel(Program, "histogram", &err);
	ckScanHistogram 	= clCreateKernel(Program, "scanhistograms", &err);
	ckScanHistogram2 	= clCreateKernel(Program, "scanhistograms2", &err);
	ckPasteHistogram 	= clCreateKernel(Program, "pastehistograms", &err);
	ckReorder 			= clCreateKernel(Program, "reorder", &err);
#endif
	ckClassifyHist 		= clCreateKernel(Program, "class_histogram", &err);
#ifdef B_KNN
	bitonicSort 		= clCreateKernel(Program, "bitonicSort", &err);
	bitonicSort2		= clCreateKernel(Program, "bitonicSort2", &err);
	mergeSort 			= clCreateKernel(Program, "bitonicMerge", &err);
	bitonicSelect 		= clCreateKernel(Program, "bitonicSelect", &err);
	//d_bSelectDist = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float)*nTests_inWork*kValue*numSelectPortion, NULL, &err);
	//d_bSelectLabel = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uchar)*nTests_inWork*kValue*numSelectPortion, NULL, &err);
#endif
	/*** create device buffer ***/
	debug("create buffer...");

	d_inDist 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float)*nTests_inWork*nTrains_rounded, NULL, &err);
	d_inLabel 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uchar)*nTests_inWork*nTrains_rounded, NULL, &err);
#ifdef R_KNN
	d_outDist 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float)*nTests_inWork*nTrains_rounded, NULL, &err);
	d_outLabel 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uchar)*nTests_inWork*nTrains_rounded, NULL, &err);

	d_Histograms = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork*split*radix, NULL, &err);
	d_globsum 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork*radix, NULL, &err);
	d_gsum 		= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork, NULL, &err);
#endif
	d_ClassHist = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork*kSplit*nClass, NULL, &err);
}

void CLKnn::CLDistance(uint idx)
{
	cl_int err;
	cl_event eve;
	size_t globalSize = nTests_inWork*nTrains_rounded;
	size_t localSize = WRAP;

	err = clSetKernelArg(ckDist, 0, sizeof(cl_mem), &d_test);
	err &= clSetKernelArg(ckDist, 1, sizeof(cl_mem), &d_train);
	err &= clSetKernelArg(ckDist, 2, sizeof(cl_mem), &d_inDist);
	err &= clSetKernelArg(ckDist, 3, sizeof(cl_mem), &d_inLabel);
	err &= clSetKernelArg(ckDist, 4, sizeof(uint), &nTests_inWork); //参加此次运算的test数据
	err &= clSetKernelArg(ckDist, 5, sizeof(uint), &nTrains_rounded);
	err &= clSetKernelArg(ckDist, 6, sizeof(uint), &nTrains);
	err &= clSetKernelArg(ckDist, 7, sizeof(uint), &nFeats);
	err &= clSetKernelArg(ckDist, 8, sizeof(uint), &idx);
	if(err != CL_SUCCESS) debug("Cannot set kernel arguments of CLDistance");

	err = clEnqueueNDRangeKernel(CommandQueue, ckDist, 1, NULL, &globalSize, NULL, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckDist!");
	clFinish(CommandQueue);
/*
	cl_ulong debut, fin;
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	dist_time += (float)(fin - debut) / 1e9;
 */
}

#ifdef R_KNN
void CLKnn::Sort()
{
	for(uint pass=0; pass<_PASS; pass++){
		Histogram(pass);
		ScanHistogram();
		Reorder(pass);
	}
}

void CLKnn::Histogram(uint pass)
{
	cl_int err;
	cl_event eve;
	size_t globalSize = nTests_inWork*split;
	size_t localSize = WRAP;

	err = clSetKernelArg(ckHistogram, 0, sizeof(cl_mem), &d_inDist);
	err = clSetKernelArg(ckHistogram, 1, sizeof(htype)*localSize*radix, NULL); //4096byte
	err = clSetKernelArg(ckHistogram, 2, sizeof(uint), &pass);
	err = clSetKernelArg(ckHistogram, 3, sizeof(cl_mem), &d_Histograms);
	err = clSetKernelArg(ckHistogram, 4, sizeof(uint), &radixbits);
	err = clSetKernelArg(ckHistogram, 5, sizeof(uint), &nTrains_rounded);
	err = clSetKernelArg(ckHistogram, 6, sizeof(uint), &split);
	err = clSetKernelArg(ckHistogram, 7, sizeof(uint), &radix);
	if(err != CL_SUCCESS) debug("cannot set kernel arg of Histogram\n");
	if(sizeof(htype)*localSize*radix >= localMem) debug("local memory is not enough for Histogram\n");

	err = clEnqueueNDRangeKernel(CommandQueue, ckHistogram, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckHistogram\n");
	clFinish(CommandQueue);

/*	cl_ulong debut, fin;
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	histo_time += (float)(fin - debut) / 1e9;*/
}

void CLKnn::ScanHistogram(void)
{
	/**********************  1th scan histogram  **********************/
	cl_int err;
	cl_event eve;
	size_t globalSize = nTests_inWork*radix;
	size_t localSize = radix;

	err = clSetKernelArg(ckScanHistogram, 0, sizeof(cl_mem), &d_Histograms);
	err = clSetKernelArg(ckScanHistogram, 1, sizeof(htype)*radix*split, NULL); //4096byte
	err = clSetKernelArg(ckScanHistogram, 2, sizeof(cl_mem), &d_globsum);
	err = clSetKernelArg(ckScanHistogram, 3, sizeof(uint), &split);
	if(err != CL_SUCCESS) debug("cannot set kernel arg of ScanHistogram1\n");
	if(sizeof(htype)*radix*split >= localMem) debug("local memory is not enough for ScanHistogram1\n");

	err = clEnqueueNDRangeKernel(CommandQueue, ckScanHistogram, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckScanHistogram1\n");
	clFinish(CommandQueue);
/*
	cl_ulong debut, fin;
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	scan_time += (float)(fin - debut) / 1e9;
*/
	/**********************  2th scan histogram  *********************/
	localSize = radix/2; //16
	globalSize = nTests_inWork*localSize;

	err = clSetKernelArg(ckScanHistogram2, 0, sizeof(cl_mem), &d_globsum);
	err = clSetKernelArg(ckScanHistogram2, 1, sizeof(htype)*radix, NULL);
	err = clSetKernelArg(ckScanHistogram2, 2, sizeof(cl_mem), &d_gsum);
	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckScanHistogram2\n");
	if(sizeof(htype)*radix >= localMem) debug("local memory is not enough for ScanHistogram2\n");

	err = clEnqueueNDRangeKernel(CommandQueue, ckScanHistogram2, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckScanHistogram2\n");
	clFinish(CommandQueue);
/*
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	scan_time += (float)(fin - debut) / 1e9;
*/
	/**********************  paste histogram  **********************/
	globalSize = nTests_inWork*radix;
	localSize = radix;

	err = clSetKernelArg(ckPasteHistogram, 0, sizeof(cl_mem), &d_Histograms);
	err = clSetKernelArg(ckPasteHistogram, 1, sizeof(cl_mem), &d_globsum);
	err = clSetKernelArg(ckPasteHistogram, 2, sizeof(uint), &split);
	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckPasteHistogram\n");

	err = clEnqueueNDRangeKernel(CommandQueue, ckPasteHistogram, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckPasteHistogram\n");
	clFinish(CommandQueue);
/*
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	scan_time += (float)(fin - debut) / 1e9;*/
}

void CLKnn::Reorder(uint pass)
{
	cl_int err;
	cl_event eve;
	size_t globalSize = nTests_inWork*split;
	size_t localSize = WRAP;

	err = clSetKernelArg(ckReorder, 0, sizeof(cl_mem), &d_inDist);
	err = clSetKernelArg(ckReorder, 1, sizeof(cl_mem), &d_inLabel);
	err = clSetKernelArg(ckReorder, 2, sizeof(cl_mem), &d_outDist);
	err = clSetKernelArg(ckReorder, 3, sizeof(cl_mem), &d_outLabel);
	err = clSetKernelArg(ckReorder, 4, sizeof(cl_mem), &d_Histograms);
	err = clSetKernelArg(ckReorder, 5, sizeof(htype)*localSize *radix, NULL);
	err = clSetKernelArg(ckReorder, 6, sizeof(uint), &pass);
	err = clSetKernelArg(ckReorder, 7, sizeof(uint), &split);
	err = clSetKernelArg(ckReorder, 8, sizeof(uint), &nTrains_rounded);
	err = clSetKernelArg(ckReorder, 9, sizeof(uint), &radixbits);
	err = clSetKernelArg(ckReorder, 10, sizeof(uint), &radix);

	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckReorder\n");
	assert(sizeof(htype)*localSize*radix < localMem);

	err = clEnqueueNDRangeKernel(CommandQueue, ckReorder, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckReorder\n");
	clFinish(CommandQueue);
/*
	cl_ulong debut, fin;
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	reorder_time += (float)(fin - debut) / 1e9;
*/
	cl_mem d_temp;
	d_temp = d_inDist;
	d_inDist = d_outDist;
	d_outDist = d_temp;

	d_temp = d_inLabel;
	d_inLabel = d_outLabel;
	d_outLabel = d_temp;
}
#endif

void CLKnn::Predict(uint idx)
{
	cl_int err;
	cl_event eve;

	err = clSetKernelArg(ckClassifyHist, 0, sizeof(cl_mem), &d_inLabel);
	err = clSetKernelArg(ckClassifyHist, 1, sizeof(cl_mem), &d_ClassHist);
	err = clSetKernelArg(ckClassifyHist, 2, sizeof(uint), &kValue);
	err = clSetKernelArg(ckClassifyHist, 3, sizeof(uint), &nClass);
	err = clSetKernelArg(ckClassifyHist, 4, sizeof(uint), &kSplit);
	err = clSetKernelArg(ckClassifyHist, 5, sizeof(uint), &nTrains_rounded);
	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckClassifyHist\n");

	size_t globalSize = nTests_inWork * kSplit;
	size_t localSize = kSplit;

	err = clEnqueueNDRangeKernel(CommandQueue, ckClassifyHist, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckClassifyHist\n");
	clFinish(CommandQueue);
/*
	cl_ulong debut, fin;
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	predict_time += (float)(fin - debut) / 1e9;
*/
}

void CLKnn::calcuPrediction(uint idx, float threshold)
{
	cl_int err;
	h_test = (float*)clEnqueueMapBuffer(CommandQueue, d_test, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float)*nTests*nFeats, 0, NULL, NULL, &err);
	if(err != CL_SUCCESS) debug("Cannot map from d_test to h_test");
	//float predictive_rate;
	for(uint i=0; i<nTests_inWork; i++)
	{
		int h_class[nClass];
		for(int j=0; j<nClass; j++) h_class[j] = 0;
		for(int j=0; j<kSplit; j++)
			for(int ir=0; ir<nClass; ir++)
				h_class[ir] += h_ClassHist[i*kSplit*nClass + j*nClass + ir];

		int true_label = 0;
		int predict_label = 0;
		int big_val = h_class[predict_label];
		for(int j=1; j<nClass; j++)
			if(big_val < h_class[j])
			{
				predict_label = j;
				big_val = h_class[j];
			}

		true_label = (int)h_test[ idx*nTests_inWork*nFeats + i*nFeats + nFeats-1 ] - 1;
		//predictive_rate = (float)big_val/(float)kValue;
		//if(predictive_rate > threshold)
		//{
		/*
			for(int idx=0; idx<nClass; idx++)
			{
				if(true_label == idx) 		R[idx]++; //FN+TP
				if(predict_label == idx) 	P[idx]++; //TP+FP
				if((predict_label == idx) && (true_label == predict_label)) TP[idx]++;
			}
		 */
			if(true_label == predict_label) true_predict_num++;
			num_after_reject++;
		//}
	}//end for loop
	err = clEnqueueUnmapMemObject(CommandQueue, d_test, h_test, 0, NULL, NULL);
	if(err != CL_SUCCESS) debug("d_test failed to unmap!\n");

}

void CLKnn::PrintPrediction(const char *filename)
{

#ifdef _FPRINTF
	FILE *fp;
	fp = fopen(filename,"wt");
	for(int l=0; l<nClass; l++)
	{
		FN[l] = R[l] - TP[l];
		FP[l] = P[l] - TP[l];
		TN[l] = num_after_reject - FN[l] - TP[l] - FP[l];
	}

	fprintf(fp,"class idx, accuracy, specificity, precision, recall\n");
	for(int l=0; l<nClass; l++)
	{
		fprintf(fp,"%d,",l+1);
		fprintf(fp,"%f,", (float)(TP[l] + TN[l])/(float)(TP[l] + TN[l] + FP[l] + FN[l]) );//accuracy
		fprintf(fp,"%f,", (float)TN[l]/(float)(TN[l] + FP[l]) );//specificity
		fprintf(fp,"%f,", (float)TP[l]/(float)(TP[l] + FP[l]) );//precision
		fprintf(fp,"%f\n", (float)TP[l]/(float)(TP[l] + FN[l]) );//recall
	}

	fprintf(fp,"well predicted number, number after rejection, number before rejection, overall accuracy\n");
	fprintf(fp,"%d,",true_predict_num);
	fprintf(fp,"%d,",num_after_reject);
	fprintf(fp,"%d,",nTests);
	fprintf(fp,"%f,",rate);
	fclose(fp);
#endif


}

void CLKnn::init_variable_from_prob(const char *filename, uint *input_size, int lmt)
{
	FILE *fp;
	fp = fopen(filename,"r");
	max_line_len = MAX_BUFFER;
	line = Malloc(char,max_line_len);
	ushort tmp_width;
	ushort width =0;
	int prob_len = 0;

	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t");
		tmp_width = 0;
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n')break;
			++tmp_width;
		}
		if(width < tmp_width) width = tmp_width;
		++prob_len;
	}
	if(lmt!=0)
	{
		if(prob_len < lmt){
			debug("length erro\n");
			return;
		}else{
			prob_len = lmt;
		}
	}
	*input_size = prob_len;
	nFeats = width;

	fclose(fp);
	free(line);
	debug("problem length: %d, feature number: %d", prob_len, nFeats);
}

void CLKnn::read_problem(const char *filename, float *h_input, uint prob_len, int cha)
{
	FILE *fp;
	fp = fopen(filename,"r");
	max_line_len = MAX_BUFFER;
	line = Malloc(char,max_line_len);

	char *endptr,*idx,*val, *label;
	int row = 0;
	int col = 0;

	for(uint i=0;i<prob_len;i++)
	{
		readline(fp);
		col = 0;
		label = strtok(line," \t\n");

		if(cha) h_train[ (nFeats - 1)*prob_len + row] = (float)strtod(label,&endptr);
		else h_test[row*nFeats + nFeats - 1] = (float)strtod(label,&endptr);

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");
			if(val == NULL)	break;

			int tmpIndex = (int) strtol(idx,&endptr,10);

			if((col+1) == tmpIndex)
			{
				if(cha) h_train[col*prob_len + row] = (float)strtod(val,&endptr);
				else h_test[row*nFeats + col] = (float)strtod(val,&endptr);
			}
			else if((col+1) < tmpIndex)
			{
				int tmpii = tmpIndex - (col + 1);
				for(int irr=0; irr<tmpii; irr++)
				{
					if(cha) h_train[col*prob_len + row] = 0.0;
					else h_test[row*nFeats + col] = 0.0;
					col++;
				}
				if(cha) h_train[col*prob_len + row] = (float)strtod(val,&endptr);
				else h_test[row*nFeats + col] = (float)strtod(val,&endptr);
			}
			col++;
		}
		row++;
	}
	fclose(fp);
	free(line);
	debug("problem length: %d, row/col: %d/%d, feature number: %d", prob_len, row, col, nFeats);
}

void CLKnn::BitonicSort(void){

	bSort(256);	Merge(512);	bSort(256);

	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);

	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);

	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);

	Merge(8192);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);

	Merge(8192*2);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);
	Merge(8192);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);


	Merge(8192*4);
	bSort(256);	Merge(512);	bSort(256);

	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);

	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);

	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);

	Merge(8192);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);

	Merge(8192*2);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);
	Merge(8192);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(4096);
	bSort(256);	Merge(512);	bSort(256);
	Merge(1024);
	bSort(256);	Merge(512);	bSort(256);
	Merge(2048);
	bSort(256); Merge(512); bSort(256);
	Merge(1024);
	bSort(256); Merge(512); bSort(256);
}

void CLKnn::BitonicSort2(void)
{
	bSort(B_KEY/2); Merge(B_KEY); bSort(B_KEY/2);
	BitonicSelect(B_KEY);

	uint numSelectPortion = nTrains_rounded/B_KEY; //32768/2048=16
	bSort2( numSelectPortion*kValue );

}

void CLKnn::Merge(uint length)
{
	cl_int err;
	cl_event eve;

	uint order = 4; // 64 threads -> 256 values
	uint localMemSize = WRAP * 2 * order; //64 * 2 * 4 = 512, local memory size for a group of work items.
	uint numGC = length/localMemSize; //512/512 = 1, number of group cluster
	uint groupNum = nTrains_rounded/localMemSize;

	size_t globalSize = nTests_inWork * groupNum * WRAP;
	size_t localSize = WRAP;

	err  = clSetKernelArg(mergeSort, 0, sizeof(cl_mem), &d_inDist);
	err  = clSetKernelArg(mergeSort, 1, sizeof(cl_mem), &d_inLabel);
	err  = clSetKernelArg(mergeSort, 2, sizeof(float)*localMemSize, NULL);
	err  = clSetKernelArg(mergeSort, 3, sizeof(uchar)*localMemSize, NULL);
	err  = clSetKernelArg(mergeSort, 4, sizeof(uint), &numGC);
	err  = clSetKernelArg(mergeSort, 5, sizeof(uint), &order);
	if(err != CL_SUCCESS)debug("mergeSort arguments failed to set!");

	err = clEnqueueNDRangeKernel(CommandQueue, mergeSort, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS)debug("mergeSort failed to run!");
	clFinish(CommandQueue);

/*
	cl_ulong debut,fin;
	err=clGetEventProfilingInfo (eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*) &debut, NULL);
	assert(err== CL_SUCCESS);
	err=clGetEventProfilingInfo (eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*) &fin, NULL);
	assert(err== CL_SUCCESS);
*/
}

void CLKnn::bSort(uint length)
{
	cl_int err;
	int locData_length = length; //1024
	int split = locData_length/WRAP;//16
	err  = clSetKernelArg(bitonicSort, 0, sizeof(cl_mem), &d_inDist);
	err  = clSetKernelArg(bitonicSort, 1, sizeof(cl_mem), &d_inLabel);
	//err  = clSetKernelArg(bitonicSort, 4, sizeof(float)*locData_length, NULL); //1024 ... *4byte
	//err  = clSetKernelArg(bitonicSort, 5, sizeof(uchar)*locData_length, NULL); //1024 ... *2byte
	err  = clSetKernelArg(bitonicSort, 2, sizeof(uint), &split); //16
	if(err != CL_SUCCESS)debug("bitonicSort arguments failed to set!");

	size_t group_num = nTests_inWork*nTrains_rounded/locData_length; //=32k/1024=32
	size_t globalSize = group_num*WRAP; //32*64
	size_t localSize = WRAP; //64

	cl_event eve;

	err = clEnqueueNDRangeKernel(CommandQueue, bitonicSort, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS)debug("bitonicSort failed to run!");
	clFinish(CommandQueue);

/*
	cl_ulong debut,fin;
	err=clGetEventProfilingInfo (eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*) &debut, NULL);
	assert(err== CL_SUCCESS);
	err=clGetEventProfilingInfo (eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*) &fin, NULL);
	assert(err== CL_SUCCESS);
*/
}

void CLKnn::BitonicSelect(uint length)
{
	uint numSelectPortion = nTrains_rounded/length; //32768/2048
	cl_int err;
	err  = clSetKernelArg(bitonicSelect, 0, sizeof(cl_mem), &d_inDist);
	err  = clSetKernelArg(bitonicSelect, 1, sizeof(cl_mem), &d_inLabel);
	err  = clSetKernelArg(bitonicSelect, 2, sizeof(uint), &kValue);
	err  = clSetKernelArg(bitonicSelect, 3, sizeof(uint), &length);//2048
	err  = clSetKernelArg(bitonicSelect, 4, sizeof(uint), &numSelectPortion);//16
	if(err != CL_SUCCESS)debug("bitonicSort arguments failed to set!");

	size_t globalSize = nTests_inWork*numSelectPortion; //n*16
	err = clEnqueueNDRangeKernel(CommandQueue, bitonicSelect, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
	if(err != CL_SUCCESS)debug("bitonicSelect failed to run!");
	clFinish(CommandQueue);

}

void CLKnn::bSort2(uint length)
{
	cl_int err;
	int split = length/WRAP;//256/64
	err  = clSetKernelArg(bitonicSort2, 0, sizeof(cl_mem), &d_inDist);
	err  = clSetKernelArg(bitonicSort2, 1, sizeof(cl_mem), &d_inLabel);
	err  = clSetKernelArg(bitonicSort2, 2, sizeof(float)*length, NULL); //256 ... *4byte
	err  = clSetKernelArg(bitonicSort2, 3, sizeof(uchar)*length, NULL); //256 ... *2byte
	err  = clSetKernelArg(bitonicSort2, 4, sizeof(uint), &split); //4
	err  = clSetKernelArg(bitonicSort2, 5, sizeof(uint), &nTrains_rounded);
	if(err != CL_SUCCESS)debug("bitonicSort arguments failed to set!");

	size_t globalSize = nTests_inWork*WRAP; //n*4*64
	size_t localSize = WRAP; //64

	err = clEnqueueNDRangeKernel(CommandQueue, bitonicSort2, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if(err != CL_SUCCESS)debug("bitonicSort failed to run!");
	clFinish(CommandQueue);

}

#ifdef S_KNN
void CLKnn::serialKnn(uint idx)
{

	for(uint ir=0; ir<nTests_inWork; ++ir)
	{
		int test_sid = idx*nTests_inWork*nFeats + ir*nFeats;
		int true_label = (int)h_test[test_sid + nFeats-1];
		/* Distance */
		for(uint ix=0; ix<nTrains; ++ix)
		{
			float sum = 0;
			for(int jx=0; jx<(nFeats-1); ++jx)
			{
				int train_id = jx*nTrains + ix;
				float temp_sum = h_test[test_sid + jx] - h_train[train_id];
				sum += temp_sum*temp_sum;
			}
			dist[ix] = sum;
			label[ix] = (uchar)h_train[(nFeats-1)*nTrains + ix];
		}
		/* Sort */
		for(int ix=0; ix<kValue; ++ix)
		{
			for(uint jx=ix; jx<nTrains; ++jx)
			{
				if(dist[ix] > dist[jx]){
					swapVal(dist[ix], dist[jx]);
					swapVal(label[ix], label[jx]);
				}
			}
		}
		int class_hist[nClass];
		for(int ix=0; ix<nClass; ++ix) class_hist[ix] = 0;

		for(int ix=0; ix<kValue; ++ix)
		{
			int temp_id = label[ix]-1;
			++class_hist[temp_id];
		}
		int big_id = 0;
		int big_val = class_hist[big_id];
		for(int ix=1; ix<nClass; ++ix)
			if(class_hist[ix] > big_val)
			{
				big_id = ix;
				big_val = class_hist[ix];
			}
		int predict_label =  big_id+1;

		for(int idx=0; idx<nClass; idx++)
		{
			if(true_label == idx) 		R[idx]++; //FN+TP
			if(predict_label == idx) 	P[idx]++; //TP+FP
			if((predict_label == idx) && (true_label == predict_label)) TP[idx]++;
		}
		if(true_label == predict_label) ++true_predict_num;
		++num_after_reject;

	}

}
#endif



