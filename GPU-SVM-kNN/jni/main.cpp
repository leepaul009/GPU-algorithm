//
// Created by namh on 2015-11-26.
//

#include <jni.h>
#include <string.h>
#include <android/log.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "com_example_ndksvm_MainActivity.h"
#include "common.h"
#include "test.h"

#include <CL/cl.h>
#include <malloc.h>



JNIEXPORT jstring JNICALL Java_com_example_ndksvm_MainActivity_jniSvmRes
  (JNIEnv *env, jobject obj, jstring cmdIn){

    test::main();

    char buffer[1024];

    return env->NewStringUTF(buffer);
}


JNIEXPORT void JNICALL Java_com_example_ndksvm_MainActivity_jniSvmTrain
(JNIEnv *env, jobject obj, jstring cmdIn){


}



JNIEXPORT void JNICALL Java_com_example_ndksvm_MainActivity_jniSvmPredict
(JNIEnv *env, jobject obj, jstring cmdIn){


}