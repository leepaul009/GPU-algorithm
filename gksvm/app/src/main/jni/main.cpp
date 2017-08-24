#include <jni.h>
#include <string.h>
#include <android/log.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "common.h"
#include "algorithm.h"
#include <CL/cl.h>
#include <malloc.h>

#include "com_example_gksvm_MainActivity.h"

JNIEXPORT jstring JNICALL Java_com_example_gksvm_MainActivity_jnifunc(JNIEnv *env, jobject obj){
    char buffer[1024] = "Run GPU-SVM-kNN!";
    return env->NewStringUTF(buffer);
}

JNIEXPORT jstring JNICALL Java_com_example_gksvm_MainActivity_jniSvmRes
  (JNIEnv *env, jobject obj, jstring cmdIn){
    algorithm::main();
    char buffer[1024];
    return env->NewStringUTF(buffer);
}
