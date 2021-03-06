#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdarg.h>
#include "common.h"
#include "algorithm.h"
#include "./libknn/CLKnn.h"

namespace algorithm
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
    int main(void)
    {
        Init_OpenCL();
        Context_cmd();
        static CLKnn knn(context, devices[0], cmdQueue);
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
