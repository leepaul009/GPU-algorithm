//version.2016.11.04 22:20

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "../common.h"
#include "svm-train.h"
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

namespace svmtrain {
	void print_null(const char *s) {}

	void exit_with_help()
	{
		debug(
		"Usage: svm-train [options] training_set_file [model_file]\n"
		"options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC		(multi-class classification)\n"
		"	1 -- nu-SVC		(multi-class classification)\n"
		"	2 -- one-class SVM\n"
		"	3 -- epsilon-SVR	(regression)\n"
		"	4 -- nu-SVR		(regression)\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-d degree : set degree in kernel function (default 3)\n"
		"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		"-v n: n-fold cross validation mode\n"
		"-q : quiet mode (no outputs)\n"
		);
		exit(1);
	}

	void exit_input_error(int line_num)
	{
		debug("Wrong input format at line %d\n", line_num);
		exit(1);
	}

	void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
	void read_problem(const char *filename);
	void do_cross_validation();
#ifdef USE_P_KERNEL
	void init_opencl();
#endif
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob;		// set by read_problem
	struct svm_model *model;
	struct svm_node *x_space;
	int cross_validation;
	int nr_fold;
	int nfeats;

	static char *line = NULL;
	static int max_line_len;

	static char* readline(FILE *input)
	{
		int len;
		
		if(fgets(line,max_line_len,input) == NULL)
			return NULL;

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

	int main(int argc, char **argv)
	{
		char input_file_name[1024];
		char model_file_name[1024];
		const char *error_msg;

		parse_command_line(argc, argv, input_file_name, model_file_name);
		read_problem(input_file_name);
		error_msg = svm_check_parameter(&prob,&param);
#ifdef USE_P_KERNEL
		init_opencl();
#endif
		if(error_msg)
		{
			debug("ERROR: %s\n",error_msg);
			exit(1);
		}

		if(cross_validation)
		{
			do_cross_validation();
		}
		else
		{
			model = svm_train(&prob,&param);
			if(svm_save_model(model_file_name,model))
			{
				debug( "can't save model to file %s\n", model_file_name);
				exit(1);
			}
			svm_free_and_destroy_model(&model);
		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		free(line);
#ifdef USE_P_KERNEL
		clReleaseKernel(prob.ckMatrix);
		clReleaseCommandQueue(prob.CommandQueue);
		clReleaseContext(prob.Context);
		clReleaseMemObject(prob.d_A);
		clReleaseMemObject(prob.d_B);
		clReleaseMemObject(prob.d_C);
		free(prob.h_C);
#endif
		return 0;
	}

#ifdef USE_P_KERNEL
	void init_opencl()
	{
		cl_device_id* Devices;
		cl_platform_id* Platforms;
		cl_uint NbPlatforms, NbDevices, DeviceKey, PlatformKey;
		cl_context Context;
		cl_command_queue CommandQueue;
		cl_kernel ckMatrix;
		cl_int status;
		cl_program Program;

		PlatformKey = 0; DeviceKey = 0;
		/*** get platforms ***/
		status = clGetPlatformIDs(0, NULL, &NbPlatforms);
		Platforms = (cl_platform_id*) malloc( NbPlatforms*sizeof(cl_platform_id) );
		status = clGetPlatformIDs(NbPlatforms, Platforms, NULL);

		char clinfo[1000];
		status = clGetPlatformInfo(Platforms[PlatformKey], CL_PLATFORM_NAME, sizeof(clinfo), clinfo, NULL);
		debug("%s", clinfo);
		/*** get devices ***/
		status = clGetDeviceIDs(Platforms[PlatformKey], CL_DEVICE_TYPE_GPU, 0, NULL, &NbDevices);
		Devices = (cl_device_id*) malloc( NbDevices*sizeof(cl_device_id) );
		status = clGetDeviceIDs(Platforms[PlatformKey], CL_DEVICE_TYPE_GPU, NbDevices, Devices, NULL);

		status = clGetDeviceInfo(Devices[DeviceKey], CL_DEVICE_NAME, sizeof(clinfo), clinfo, NULL);
		debug("%s", clinfo);

		Context = clCreateContext(0, 1, &Devices[DeviceKey], NULL, NULL, &status);
		CommandQueue = clCreateCommandQueue(Context, Devices[DeviceKey], CL_QUEUE_PROFILING_ENABLE, &status);
		debug("OpenCL initialization OK!\n");

		std::string fileDir;
#ifdef _MOBILE
		fileDir.append("/storage/sdcard0/libsvm/inbuildMatrix.cl");
#else
		fileDir.append("inbuildMatrix.cl");
#endif
		std::string kernelSource = loadProgram(fileDir);
		const char* kernelSourceChar = kernelSource.c_str();
		debug("write matrix.cl OK !");

		/*** create Program ***/
		Program = clCreateProgramWithSource(Context, 1, &kernelSourceChar, NULL, &status);
		if(status != CL_SUCCESS) debug("clCreateProgramWithSource failed!\n");
		debug("clCreateProgramWithSource OK !");

		status = clBuildProgram(Program, 0, NULL, NULL, NULL, NULL);
		if(status != CL_SUCCESS) debug("clBuildProgram failed!\n");
		debug("clBuildProgram OK !");

		/*** create Kernel ***/
		ckMatrix = clCreateKernel(Program, "myKernelComputation", &status);
		if(status != CL_SUCCESS) debug("clCreateKernel failed!\n");
		debug("clCreateKernel OK!");

		int len = SUB_PROB_LEN;

		prob.d_A = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, len*nfeats*sizeof(float), NULL, &status);
		if(status != CL_SUCCESS) debug("d_A failed to create!\n");
		prob.d_B = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, nfeats*sizeof(float), NULL, &status);
		if(status != CL_SUCCESS) debug("d_B failed to create!\n");
		prob.d_C = clCreateBuffer(Context, CL_MEM_READ_WRITE, len*sizeof(float), NULL, &status);
		if(status != CL_SUCCESS) debug("d_C failed to create!\n");

		prob.CommandQueue = CommandQueue;
		prob.Context = Context;
		prob.ckMatrix = ckMatrix;
		prob.h_C = Malloc(float, len);
		prob.nFeats = nfeats;
	}
#endif

	void do_cross_validation()
	{
		int i;
		int total_correct = 0;
		double total_error = 0;
		double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
		double *target = Malloc(double,prob.l);

		svm_cross_validation(&prob,&param,nr_fold,target);
		if(param.svm_type == EPSILON_SVR ||
		   param.svm_type == NU_SVR)
		{
			for(i=0;i<prob.l;i++)
			{
				double y = prob.y[i];
				double v = target[i];
				total_error += (v-y)*(v-y);
				sumv += v;
				sumy += y;
				sumvv += v*v;
				sumyy += y*y;
				sumvy += v*y;
			}
			debug("Cross Validation Mean squared error = %g\n",total_error/prob.l);
			debug("Cross Validation Squared correlation coefficient = %g\n",
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
				);
		}
		else
		{
			for(i=0;i<prob.l;i++)
				if(target[i] == prob.y[i])
					++total_correct;
			debug("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
		}
		free(target);
	}

	void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
	{
		int i;
		void (*print_func)(const char*) = NULL;	// default printing to stdout

		// default values
		param.svm_type = C_SVC;
		param.kernel_type = RBF;
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		cross_validation = 0;

		// parse options
		for(i=1;i<argc;i++)
		{
			if(argv[i][0] != '-') break;
			if(++i>=argc)
				exit_with_help();
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
				case 'q':
					print_func = &print_null;
					i--;
					break;
				case 'v':
					cross_validation = 1;
					nr_fold = atoi(argv[i]);
					if(nr_fold < 2)
					{
						debug("n-fold cross validation: n must >= 2\n");
						exit_with_help();
					}
					break;
				case 'w':
					++param.nr_weight;
					param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
					param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
					param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
					param.weight[param.nr_weight-1] = atof(argv[i]);
					break;
				default:
					debug("Unknown option: -%c\n", argv[i-1][1]);
					exit_with_help();
			}
		}

		svm_set_print_string_function(print_func);

		// determine filenames

		if(i>=argc)
			exit_with_help();

		strcpy(input_file_name, argv[i]);

		if(i<argc-1)
			strcpy(model_file_name,argv[i+1]);
		else
		{
			char *p = strrchr(argv[i],'/');
			if(p==NULL)
				p = argv[i];
			else
				++p;
			sprintf(model_file_name,"%s.model",p);
		}
	}

	void read_problem(const char *filename)
	{

		int tmp_nfeats;

		int max_index, inst_max_index, i;
		size_t j;
		FILE *fp = fopen(filename,"r");
		char *endptr;
		char *idx, *val, *label;

		if(fp == NULL)
		{
			debug("can't open input file %s\n",filename);
			exit(1);
		}

		prob.l = 0;
		nfeats = 0;

		max_line_len = 1024;
		line = Malloc(char,max_line_len);

		while(readline(fp)!=NULL)
		{
			char *p = strtok(line," \t"); // label

			tmp_nfeats = 0;
			while(1)
			{
				p = strtok(NULL," \t");
				if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
					break;
				++tmp_nfeats;
			}
			if(tmp_nfeats > nfeats) nfeats = tmp_nfeats;
			++prob.l;
		}
		rewind(fp);
		debug("The column of input is:%d", nfeats);
		prob.y = Malloc(double,prob.l);
		prob.x = Malloc(struct svm_node *, prob.l);
		x_space = Malloc(struct svm_node, prob.l*nfeats);

		max_index = 0; j=0;
		for(i=0;i<prob.l;i++)
		{
			inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
			readline(fp);
			prob.x[i] = &x_space[j];
			label = strtok(line," \t\n");
			if(label == NULL) // empty line
				exit_input_error(i+1);

			prob.y[i] = strtod(label,&endptr);
			if(endptr == label || *endptr != '\0')
				exit_input_error(i+1);

			int col = 0;
			while(1)
			{
				idx = strtok(NULL,":");
				val = strtok(NULL," \t");

				if(val == NULL) break;

				errno = 0;
				int tmpIndex = (int) strtol(idx,&endptr,10);
				double tmpValue = strtod(val,&endptr);

				int gap = 0;
				if( (col+1) == tmpIndex )
				{
					x_space[j].index = tmpIndex;
					x_space[j].value = tmpValue;
				}
				else if( (col+1) < tmpIndex )
				{
					gap = tmpIndex - (col+1); // the gap
					for(int irr=0; irr<gap; irr++)
					{
						x_space[j].index = col+1;
						x_space[j].value = 0.0;
						++j;
						++col;
					}
					x_space[j].index = tmpIndex;
					x_space[j].value = tmpValue;
				}

				if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
					exit_input_error(i+1);
				else
					inst_max_index = x_space[j].index;
				if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
					exit_input_error(i+1);

				++col;
				++j;
			}
			x_space[j++].index = -1;

			if(inst_max_index > max_index) max_index = inst_max_index;
		}

		if(param.gamma == 0 && max_index > 0) param.gamma = 1.0/max_index;

		if(param.kernel_type == PRECOMPUTED)
			for(i=0;i<prob.l;i++)
			{
				if (prob.x[i][0].index != 0)
				{
					debug("Wrong input format: first column must be 0:sample_serial_number\n");
					exit(1);
				}
				if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
				{
					debug("Wrong input format: sample_serial_number out of range\n");
					exit(1);
				}
			}

		fclose(fp);
	}
}
