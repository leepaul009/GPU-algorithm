LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := NDKLib
#LOCAL_SRC_FILES := main.cpp
LOCAL_SRC_FILES := \
	common.cpp main.cpp \
	libsvm/svm-train.cpp \
	libsvm/svm-predict.cpp \
	libsvm/svm.cpp

LOCAL_LDLIBS := -llog -ldl

include $(BUILD_SHARED_LIBRARY)