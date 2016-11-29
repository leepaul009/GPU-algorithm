// parallel-svm-knn, 2016/11/28
#include "CLKnn.h"
#include "svm.h"
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
	char input_train_file_name[1024] = "/storage/sdcard0/libsvm/b9f3t_36k.scale";
	char input_test_file_name[1024] = "/storage/sdcard0/libsvm/b9f3p.scale";
#else
	char input_train_file_name[1024] = "../0.data/data/b9f3t.s";
	char input_test_file_name[1024] = "../0.data/data/b9f3p.s";
#endif
	//同时,为h_train初始化,为nFeats,h_train和nTests赋值
	init_variable_from_prob(input_train_file_name, &nTrains, 0); //0:lmt
	init_variable_from_prob(input_test_file_name, &nTests, 0);

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

	split = SPLIT;
	radix = _RADIX;
	radixbits = _BITS;
	kValue = NUM_KNN_K;
	kSplit = NUM_KNN_K/4; //表示 kernel class_hist 的线程数目
	nClass = _CLASS;
	kValue = (int)ceil((float)kValue / (float)kSplit)*kSplit;
	nTrains_rounded = ((uint)ceil((double)nTrains / (double)_MULTIPLY))*_MULTIPLY;
	nTests_inWork = 2; //满足 nTests_inWork*split 是WRAP的复数
	h_ClassHist = new htype[nTests_inWork*kSplit*nClass];

	//保证histogram function的线程数 大于 可操作的数据块
	clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
	debug("Testing data has %d rows, and training data after width-fixing has %d rows", nTests, nTrains_rounded);
	/*** create kernel and MEM object ***/
	PreProcess();
	/*** initiate evaluation factor ***/
	for(int i=0; i<nClass; i++)
	{
		SVM_TP[i] = 0; SVM_FP[i] = 0; SVM_FN[i] = 0; SVM_TN[i] = 0; SVM_P[i] = 0; SVM_R[i] = 0;
	}
	/*** initiate SVM parameter ***/
	initKnnSvm();

	debug("%d...",nTests/nTests_inWork);
	for(uint i=0; i<nTests/nTests_inWork; i++)
	{
		debug("%d", i);
		CLDistance(i);
		Sort();
		Predict(i);

		err = clEnqueueReadBuffer(CommandQueue, d_ClassHist, CL_TRUE, 0, sizeof(htype)*nTests_inWork*kSplit*nClass, h_ClassHist, 0, NULL, NULL);

		h_test = (float*)clEnqueueMapBuffer(CommandQueue, d_test, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*nTests*nFeats, 0, NULL, NULL, &err);
		if(err != CL_SUCCESS) debug("Cannot map from d_test to h_test");
		h_train = (float*)clEnqueueMapBuffer(CommandQueue, d_train, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*nTrains*nFeats, 0, NULL, NULL, &err);
		if(err != CL_SUCCESS) debug("Cannot map from d_train to h_train");
		predict_list = (htype*)clEnqueueMapBuffer(CommandQueue, d_inIdx, CL_TRUE, CL_MAP_READ, 0, sizeof(htype)*nTests_inWork*nTrains_rounded, 0, NULL, NULL, &err);
		if(err != CL_SUCCESS) debug("Cannot map from d_inIdx to predict_list");

		for(uint ir=0; ir<nTests_inWork; ir++){
			calcuPrediction(i, ir, 0.0);
			if(use_svm == 1) knnSvm(i, ir);
		}

		err = clEnqueueUnmapMemObject(CommandQueue, d_test, h_test, 0, NULL, NULL);
		if(err != CL_SUCCESS) debug("d_test failed to unmap!\n");
		err = clEnqueueUnmapMemObject(CommandQueue, d_train, h_train, 0, NULL, NULL);
		if(err != CL_SUCCESS) debug("d_train failed to unmap!\n");
		err = clEnqueueUnmapMemObject(CommandQueue, d_inIdx, predict_list, 0, NULL, NULL);
		if(err != CL_SUCCESS) debug("d_inIdx failed to unmap!\n");
	}


	float rate = ( (float)svm_true_predict_num )/(float)nTests;
	debug("svm predict number: %d, overall accuracy: %f", svm_true_predict_num, rate);
}


CLKnn::~CLKnn()
{
	free(prob.y);
	free(prob.x);
	free(prob.kv);
	free(x_space);

	free(test_nodes);
#ifdef P_KNNSVM
	//free( h_A );
	free( h_C );
	free(kv_space);
#endif

	delete[] h_ClassHist;

	clReleaseMemObject(d_test);
	clReleaseMemObject(d_train);
	clReleaseMemObject(d_inDist);
	clReleaseMemObject(d_inLabel);

	clReleaseMemObject(d_inIdx);
	clReleaseMemObject(d_outIdx);

	clReleaseMemObject(d_outDist);
	clReleaseMemObject(d_outLabel);
	clReleaseMemObject(d_Histograms);
	clReleaseMemObject(d_globsum);
	clReleaseMemObject(d_gsum);
	clReleaseMemObject(d_ClassHist);
#ifdef P_KNNSVM
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_C);
#endif
	clReleaseKernel(ckDist);
	clReleaseKernel(ckHistogram);
	clReleaseKernel(ckScanHistogram);
	clReleaseKernel(ckScanHistogram2);
	clReleaseKernel(ckPasteHistogram);
	clReleaseKernel(ckReorder);
	clReleaseKernel(ckClassifyHist);
	clReleaseKernel(ckKernelComputation);

	clReleaseProgram(Program);
	clReleaseCommandQueue(CommandQueue);
	clReleaseContext(Context);
}

void CLKnn::PreProcess(void)
{
	cl_int err;
	/*** create program with source & build program ***/
	debug("create program with source");
	string fileDir;
#ifdef _MOBILE
	fileDir.append("/storage/sdcard0/libsvm/kernel_psvmknn.cl");
#else
	fileDir.append("kernel_psvmknn.cl");
#endif
	string kernelSource = loadProgram2(fileDir);
	const char* kernelSourceChar = kernelSource.c_str();
	Program = clCreateProgramWithSource(Context, 1, &kernelSourceChar, NULL, &err);
	err = clBuildProgram(Program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS)debug("create program failed");

	/*** create kernel ***/
	debug("create kernel");
	ckDist 				= clCreateKernel(Program, "Dist", &err);
	ckHistogram 		= clCreateKernel(Program, "histogram", &err);
	ckScanHistogram 	= clCreateKernel(Program, "scanhistograms", &err);
	ckScanHistogram2 	= clCreateKernel(Program, "scanhistograms2", &err);
	ckPasteHistogram 	= clCreateKernel(Program, "pastehistograms", &err);
	ckReorder 			= clCreateKernel(Program, "reorder", &err);
	ckClassifyHist 		= clCreateKernel(Program, "class_histogram", &err);
	ckKernelComputation = clCreateKernel(Program, "myKernelComputation", &err);

	/*** create device buffer ***/
	debug("create buffer");

	d_inIdx		= clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(htype)*nTests_inWork*nTrains_rounded, NULL, &err);
	d_outIdx	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork*nTrains_rounded, NULL, &err);

	d_inDist 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float)*nTests_inWork*nTrains_rounded, NULL, &err);
	d_inLabel 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uchar)*nTests_inWork*nTrains_rounded, NULL, &err);
	d_outDist 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float)*nTests_inWork*nTrains_rounded, NULL, &err);
	d_outLabel 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(uchar)*nTests_inWork*nTrains_rounded, NULL, &err);

	d_Histograms = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork*split*radix, NULL, &err);
	d_globsum 	= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork*radix, NULL, &err);
	d_gsum 		= clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork, NULL, &err);
	d_ClassHist = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(htype)*nTests_inWork*kSplit*nClass, NULL, &err);

	debug("PreProcess finished");
}

void CLKnn::CLDistance(uint idx)
{
	cl_int err;
	cl_event eve;
	size_t globalSize = nTests_inWork*nTrains_rounded;
	size_t localSize = WRAP;

	err = clSetKernelArg(ckDist, 0, sizeof(cl_mem), &d_test);
	err = clSetKernelArg(ckDist, 1, sizeof(cl_mem), &d_train);
	err = clSetKernelArg(ckDist, 2, sizeof(cl_mem), &d_inDist);
	err = clSetKernelArg(ckDist, 3, sizeof(cl_mem), &d_inLabel);
	err = clSetKernelArg(ckDist, 4, sizeof(uint), &nTests_inWork); //参加此次运算的test数据
	err = clSetKernelArg(ckDist, 5, sizeof(uint), &nTrains_rounded);
	err = clSetKernelArg(ckDist, 6, sizeof(uint), &nTrains);
	err = clSetKernelArg(ckDist, 7, sizeof(uint), &nFeats);
	err = clSetKernelArg(ckDist, 8, sizeof(cl_mem), &d_inIdx);
	err = clSetKernelArg(ckDist, 9, sizeof(uint), &idx);
	if(err != CL_SUCCESS) debug("Cannot set kernel arg of CLDistance");

	err = clEnqueueNDRangeKernel(CommandQueue, ckDist, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckDist");
	clFinish(CommandQueue);
/*
	cl_ulong debut, fin;
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	dist_time += (float)(fin - debut) / 1e9;*/
}

void CLKnn::Sort()
{
	for(uint pass=0; pass<_PASS; pass++)
	{
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
	if(err != CL_SUCCESS) debug("cannot set kernel arg of Histogram");
	if(sizeof(htype)*localSize*radix >= localMem) debug("local memory is not enough for Histogram");

	err = clEnqueueNDRangeKernel(CommandQueue, ckHistogram, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckHistogram");
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
	if(err != CL_SUCCESS) debug("cannot set kernel arg of ScanHistogram1");
	if(sizeof(htype)*radix*split >= localMem) debug("local memory is not enough for ScanHistogram1");

	err = clEnqueueNDRangeKernel(CommandQueue, ckScanHistogram, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckScanHistogram1");
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
	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckScanHistogram2");
	if(sizeof(htype)*radix >= localMem) debug("local memory is not enough for ScanHistogram2");

	err = clEnqueueNDRangeKernel(CommandQueue, ckScanHistogram2, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckScanHistogram2");
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
	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckPasteHistogram");

	err = clEnqueueNDRangeKernel(CommandQueue, ckPasteHistogram, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckPasteHistogram");
	clFinish(CommandQueue);
/*
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	scan_time += (float)(fin - debut) / 1e9;*/
}

void CLKnn::Reorder(uint pass){

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
	err = clSetKernelArg(ckReorder, 11, sizeof(cl_mem), &d_inIdx);
	err = clSetKernelArg(ckReorder, 12, sizeof(cl_mem), &d_outIdx);

	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckReorder");
	assert(sizeof(htype)*localSize*radix < localMem);

	err = clEnqueueNDRangeKernel(CommandQueue, ckReorder, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckReorder");
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

	d_temp = d_inIdx;
	d_inIdx = d_outIdx;
	d_outIdx = d_temp;
}

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
	if(err != CL_SUCCESS) debug("Cannot set kernel arg of ckClassifyHist");

	size_t globalSize = nTests_inWork * kSplit;
	size_t localSize = kSplit;

	err = clEnqueueNDRangeKernel(CommandQueue, ckClassifyHist, 1, NULL, &globalSize, &localSize, 0, NULL, &eve);
	if(err != CL_SUCCESS) debug("Cannot run kernel ckClassifyHist");
	clFinish(CommandQueue);
/*
	cl_ulong debut, fin;
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&debut, NULL);
	err = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&fin, NULL);
	assert(err == CL_SUCCESS);
	predict_time += (float)(fin - debut) / 1e9;
*/
}

void CLKnn::calcuPrediction(uint idx, uint i, float threshold)
{
	int h_class[nClass];
	int sort_class[nClass];

	for(int j=0; j<nClass; j++) h_class[j] = 0;
	for(int j=0; j<kSplit; j++)
		for(int ir=0; ir<nClass; ir++)
			h_class[ir] += h_ClassHist[i*kSplit*nClass + j*nClass + ir];

/* 判断此条knn预测是否可信 */
	use_svm = 0;
	for(int ir=0; ir<nClass; ir++)
		sort_class[ir] = h_class[ir];

	for(int ir=0; ir<2; ir++)
		for(int jr=ir+1; jr<nClass; jr++)
			if(sort_class[ir] < sort_class[jr])
				swap(sort_class[ir] , sort_class[jr]);

	int gap = sort_class[0] - sort_class[1];
	if(  (sort_class[1]>0) && ((float)gap/(float)sort_class[0] < 0.7) ) use_svm = 1;

	if(use_svm == 0)
	{
		int true_label = (int)h_test[ idx*nTests_inWork*nFeats + i*nFeats + nFeats-1 ] - 1;
		int predict_label = 0;
		int big_val = h_class[predict_label];

		for(int j=1; j<nClass; j++)
			if(big_val < h_class[j])
			{
				predict_label = j;
				big_val = h_class[j];
			}
		for(int idx=0; idx<nClass; idx++)
		{
			if(true_label == idx) SVM_R[idx]++; //FN+TP
			if(predict_label == idx) SVM_P[idx]++; //TP+FP
			if((predict_label == idx) && (true_label == predict_label)) SVM_TP[idx]++;
		}
		if(true_label == predict_label) svm_true_predict_num++;
	}
}


void CLKnn::initKnnSvm(void)
{
	debug("initTnnSvm...");
	string cmdIn;
	string cmdString;
	cmdIn = "-t 2 -s 0 -c 100 -g 0.1 -e 0.1 -b 1";
	vector<char*> v;
	cmdString = string("dummy ")+cmdIn;
	cmdToArgv(cmdString, v);

	parse_command_line((int)v.size(), &v[0]);
	for(int ir=0;ir<(int)v.size();ir++) free(v[ir]);

	prob.l 	= SVM_KEY;
	prob.y 	= Malloc(double,prob.l);
	prob.x 	= Malloc(struct svm_node *, prob.l);
	prob.kv = Malloc(struct svm_node_Q *, prob.l);

	test_nodes 	= Malloc(struct svm_node, nFeats); //*199 ..-1
	x_space 	= Malloc(struct svm_node, prob.l*(nFeats)); //*199 ..-1

#ifdef P_KNNSVM
	kv_space 	= Malloc(struct svm_node_Q, prob.l*(prob.l+1));

	cl_int status;
	nPureFeats = nFeats - 1; //198

	d_A = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, prob.l*nPureFeats*sizeof(float), NULL, &status);
	if(status != CL_SUCCESS) debug("d_A failed to create!\n");
	d_C = clCreateBuffer(Context, CL_MEM_READ_WRITE, prob.l*sizeof(SNQfloat), NULL, &status);
	if(status != CL_SUCCESS) debug("d_C failed to create!\n");

	//h_A = Malloc(float, prob.l*nPureFeats);
	h_C = Malloc(SNQfloat, prob.l);
#endif
}

void CLKnn::knnSvm(uint idx, uint i)
{
#ifdef P_KNNSVM
	cl_int err;
	int pro_len = prob.l;
	h_A = (float*)clEnqueueMapBuffer(CommandQueue, d_A, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float)*pro_len*nPureFeats, 0, NULL, NULL, &err);
	if(err != CL_SUCCESS) debug("Cannot map from d_A to h_A");
#endif
	struct svm_model *model;
	int key = 0;
	double predict_label;
	int true_label;
	for(int ir=0; ir<prob.l; ir++)
	{
		htype idx = predict_list[i*nTrains_rounded + ir];
		prob.y[ir] = (double)h_train[ (nFeats-1)*nTrains + idx ];
		prob.x[ir] = &x_space[key];

		for(int jr=0; jr<(nFeats-1); jr++)
		{
			x_space[key].index = jr+1;
			float tmp = h_train[jr*nTrains + idx];
			x_space[key].value = tmp;
#ifdef P_KNNSVM
			h_A[jr*pro_len + ir] = tmp;
#endif
			++key;
		}
		x_space[key++].index = -1;
	}
#ifdef P_KNNSVM
	err = clEnqueueUnmapMemObject(CommandQueue, d_A, h_A, 0, NULL, NULL);
		if(err != CL_SUCCESS) debug("d_A failed to unmap!\n");


		/* Parallel Computation of Kernel */
		int index = 0;
		for(int ir=0; ir<pro_len; ir++)
		{
			prob.kv[ir] = &kv_space[index];

			float gamma = (float)param.gamma;
			err = clSetKernelArg(ckKernelComputation, 0, sizeof(cl_mem), (void *)&d_A);
			err = clSetKernelArg(ckKernelComputation, 1, sizeof(int), (void *)&ir);
			err = clSetKernelArg(ckKernelComputation, 2, sizeof(cl_mem), (void *)&d_C);
			err = clSetKernelArg(ckKernelComputation, 3, sizeof(int), (void *)&nPureFeats);//width
			err = clSetKernelArg(ckKernelComputation, 4, sizeof(int), (void *)&pro_len);//height
			err = clSetKernelArg(ckKernelComputation, 5, sizeof(float), (void *)&gamma);//param

			size_t global_work_size = pro_len;
			err = clEnqueueNDRangeKernel(CommandQueue, ckKernelComputation, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
			if(err != CL_SUCCESS) debug("run ckMatrix failed!\n");
			err = clFinish(CommandQueue);
			if(err != CL_SUCCESS) debug("clFinish failed!\n");

			err = clEnqueueReadBuffer(CommandQueue, d_C, CL_TRUE, 0, pro_len*sizeof(SNQfloat), h_C, 0, NULL, NULL);
			if(err != CL_SUCCESS) debug("read failed!\n");

			SNQindex kernelIdx = 0;
			kv_space[index].index = kernelIdx;
			kv_space[index].value = (SNQfloat)(ir+1);
			++index; ++kernelIdx;
			for(int jr=0; jr<pro_len; jr++)
			{
				kv_space[index].index = kernelIdx;
				kv_space[index].value = (SNQfloat)exp((double)h_C[jr]);
				//debug("kv......%f",kv_space[index].value);
				++index; ++kernelIdx;
			}
		}
		//debug("kernel computation finished!");
		param.kernel_type = PRECOMPUTED;
#endif

	model = svm_train(&prob,&param);

	for(int ir=0;ir<(nFeats-1);ir++)
	{
		test_nodes[ir].index = ir+1;
		test_nodes[ir].value = h_test[idx*nTests_inWork*nFeats + i*nFeats + ir];
	}
	test_nodes[nFeats-1].index = -1;

#ifdef PREDICT_P
	int nr_class = svm_get_nr_class(model);
		double *prob_estimates = (double *) malloc(nr_class*sizeof(double));
		for(int ir=0; ir<nr_class; ir++) prob_estimates[ir] = 0;

		predict_label = svm_predict_probability(model, test_nodes, prob_estimates);
#else
	predict_label = svm_predict(model, test_nodes);
#endif

	true_label = (int)h_test[idx*nTests_inWork*nFeats + i*nFeats + nFeats-1];

	//compute prediction information of GPU-SVM-KNN
	for(int idx=0; idx<nClass; idx++)
	{
		if(true_label == idx) SVM_R[idx]++; //FN+TP
		if((int)predict_label == idx) SVM_P[idx]++; //TP+FP
		if(((int)predict_label == idx) && (true_label == (int)predict_label)) SVM_TP[idx]++;
	}
	if((int)predict_label == true_label) svm_true_predict_num++;

#ifdef PRINT
	#ifdef PREDICT_P
		fprintf(fp_comp, "%d,", (int)predict_label);

		int *labels=(int *) malloc(nr_class*sizeof(int));
		svm_get_labels(model,labels);
		fprintf(fp_comp, "P:,");
		for(int ir=0; ir<nr_class; ir++)
			fprintf(fp_comp,"%d,%g,", labels[ir], prob_estimates[ir]);

		fprintf(fp_comp, "\n");
		free(labels);
		free(prob_estimates);
#else
		fprintf(fp_comp, "%d\n", (int)predict_label);
#endif
#endif

	svm_free_and_destroy_model(&model);
}

void CLKnn::parse_command_line(int argc, char **argv)
{
	int i;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;

	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.nu = 0.5;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	//cross_validation = 0;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc) {
			debug("error...");
			break;
		}
		//exit_with_help();
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				printf("Unknown option: -%c\n", argv[i-1][1]);
				//exit_with_help();
		}
	}

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
