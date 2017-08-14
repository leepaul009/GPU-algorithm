//
// Created by lee on 2016-09-02.
//

#include <jni.h>
#include <string.h>
#include <android/log.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "common.h"
#include "test.h"
#include <CL/cl.h>
#include <malloc.h>

#include "com_example_gksvm_MainActivity.h"



JNIEXPORT jstring JNICALL Java_com_example_gksvm_MainActivity_jnifunc(JNIEnv *env, jobject obj){

    char buffer[1024] = "hello,jni!";
    //sprintf(buffer, "%d\n");
    return env->NewStringUTF(buffer);
    //return (*env)->NewStringUTF(env, "Hello World! I am Native interface");
}

JNIEXPORT jstring JNICALL Java_com_example_gksvm_MainActivity_jniSvmRes
  (JNIEnv *env, jobject obj, jstring cmdIn){
    test::main();
    char buffer[1024];
    return env->NewStringUTF(buffer);
}