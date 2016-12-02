//
// Created by root on 16-9-4.
//
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <malloc.h>

/*
#include <android/log.h>
#include <string.h>
#include <string>
#include <vector>
#include <math.h>
#include <time.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
*/

#include "test.h"
#include "common.h"

#define LEN(arr) sizeof(arr) / sizeof(arr[0])
#define N 1024
#define NUM_THREAD 128

namespace test
{
    cl_uint num_platform;
    cl_uint num_device;
    cl_platform_id *platforms;
    cl_device_id *devices;
    cl_int err;
    cl_context context;
    cl_command_queue cmdQueue;

    void Init_OpenCL();
    void Context_cmd();

    //int main(int argc, char **argv){
    int main(void)
    {
        Init_OpenCL();
        Context_cmd();
        CLKnn knn(context, devices[0], cmdQueue);
        return 0;
    }
    void Init_OpenCL()
    {
        err = clGetPlatformIDs(0, 0, &num_platform);
        platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platform);
        err = clGetPlatformIDs(num_platform, platforms, NULL);

        err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id)*num_device);
        err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_device, devices, NULL);
    }
    void Context_cmd()
    {
        context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
        cmdQueue = clCreateCommandQueue(context, devices[0], 0, &err);
    }

}
