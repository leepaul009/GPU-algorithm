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
#include "./libsvm/svm-train.h"
#include "./libsvm/svm-predict.h"
#include "common.h"

#include <CL/cl.h>
#include <malloc.h>



JNIEXPORT jstring JNICALL Java_com_example_ndksvm_MainActivity_jniSvmRes
  (JNIEnv *env, jobject obj, jstring cmdIn){

   // const char *cmd = env->GetStringUTFChars(cmdIn, 0);
   // debug("Java_com_example_ndksvm_MainActivity_jniSvmRes cmd = %s", cmd);
   // std::vector<char*> v;
    //std::string cmdString = std::string("dummy ") + std::string(cmd);
    //std::string cmdString = std::string(cmd);
    //cmdToArgv(cmdString, v);

    //test::main(v.size(),&v[0]);
    //test::main();

    //for(int i=0;i<v.size();i++){
        //debug("Java_com_example_ndksvm_MainActivity_jniSvmRes_vector = %s", v[i]);
    	//free(v[i]);
    //}
   // env->ReleaseStringUTFChars(cmdIn, cmd);
    //int res = test();
    char buffer[1024];
    return env->NewStringUTF(buffer);
}


JNIEXPORT void JNICALL Java_com_example_ndksvm_MainActivity_jniSvmTrain
(JNIEnv *env, jobject obj, jstring cmdIn){

	const char *cmd = env->GetStringUTFChars(cmdIn, 0);
	debug("jniSvmTrain cmd = %s", cmd);

	std::vector<char*> v;
	// add dummy head to meet argv/command format
	std::string cmdString = std::string("dummy ") + std::string(cmd);

	cmdToArgv(cmdString, v);

	// make svm train by libsvm
	svmtrain::main(v.size(),&v[0]);

	// free vector memory
	for(int i=0;i<v.size();i++){
		free(v[i]);
	}

	// free java object memory
	env->ReleaseStringUTFChars(cmdIn, cmd);
}



JNIEXPORT void JNICALL Java_com_example_ndksvm_MainActivity_jniSvmPredict
(JNIEnv *env, jobject obj, jstring cmdIn){

	const char *cmd = env->GetStringUTFChars(cmdIn, 0);
	debug("jniSvmPredict cmd = %s", cmd);

	std::vector<char*> v;

	// add dummy head to meet argv/command format
	std::string cmdString = std::string("dummy ")+std::string(cmd);

	cmdToArgv(cmdString, v);

	// make svm train by libsvm
	svmpredict::main(v.size(),&v[0]);


	// free vector memory
	for(int i=0;i<v.size();i++){
		free(v[i]);
	}

	// free java object memory
	env->ReleaseStringUTFChars(cmdIn, cmd);
}