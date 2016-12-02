/*** version 0925 ***/

#include <jni.h>
#include <string.h>
#include <android/log.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "com_example_svm_MainActivity.h"
#include "./libsvm/svm-train.h"
#include "./libsvm/svm-predict.h"
#include "common.h"


JNIEXPORT void JNICALL Java_com_example_svm_MainActivity_jniSvmTrain(JNIEnv *env, jobject obj, jstring cmdIn){

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



JNIEXPORT void JNICALL Java_com_example_svm_MainActivity_jniSvmPredict(JNIEnv *env, jobject obj, jstring cmdIn){

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