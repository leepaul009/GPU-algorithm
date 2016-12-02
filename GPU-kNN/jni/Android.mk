LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)


LOCAL_MODULE    := jnilibsvm
LOCAL_SRC_FILES := \
	common.cpp test.cpp main.cpp \
	libknn/CLKnn.cpp


LOCAL_CPPFLAGS	:= -DARM -DOS_LNX -DARCH_32
LOCAL_C_INCLUDES  := $(LOCAL_PATH)/include
#LOCAL_LDLIBS      := -ljnigraphics -llog -ldl $(LOCAL_PATH)/libs/libOpenCL.so

#LOCAL_LDLIBS      := -llog -ldl $(LOCAL_PATH)/libs/libGLES_mali.so
LOCAL_LDLIBS      := -llog -ldl $(LOCAL_PATH)/libs/libOpenCL.so
#LOCAL_ARM_MODE    := arm

include $(BUILD_SHARED_LIBRARY)
