# GPU-algorithm
The pattern recognition algorithms optimized with GPU and OpenCL. The algorithms are implemented with Android NDK and will run with Android application. Fhe following algorithms are included:
```
GPU-SVM, parallel version of kNN with a GPU-based run-time acceleration. 
SVM, serial version of kNN, reference:http://www.csie.ntu.edu.tw/~cjlin/libsvm/.
GPU-SVM-kNN, using a SVM-based decision-maker to refine the outcome of kNN.
GPU-kNN, including GPU-Radix-kNN and GPU-Bitonic-kNN.
gksvm, optimized GPU-SVM-kNN
```


## Prerequisites：
A GPU is needed in the smartphone, otherwise the algorithm could not be built successfully. Besides, the GPU should be available in OpenCL 1.2 or OpenCL 2.0. The vendor of smartphone always provide a OpenCL Dynamic Link Library, named XXXOpenCL.so. This DLL library is needed in this implementation. 


# How to install:
## Install Android studio
```
Install java-sdk
Install Android studio
Download Android-Native-Develop-Kit, android-ndk-r15c-windows-x86_64.zip
Create android project, named 'gksvm'
```
## The setting of Android studio 
For Android studio, in menu Setting->Tools->External Tools, add a new item:
```
Name: javah
Program: $YourJavaPath\bin\javah.exe
Parameters: -classpath $Classpath$ -v -jni $FileClass$
Working directory: $YourAndroidProjectPath$\app\src\main\jni
```
And add another new item for External Tools:
```
Name: ndk-build
Program: $YourNdkPath\ndk-build.cmd
Working directory: $YourAndroidProjectPath$\app\src\main\jni
```

## Create Java head file
Edit $YourAndroidProjectPath\local.properties, in the end of file, add following:
```
ndk.dir=$YourNdkPath\\android-ndk-r15c
sdk.dir=$YourSdkPath\\Sdk 
```
Edit $YourAndroidProjectPath\gradle.properties, in the end of file, add following:
```
android.useDeprecatedNdk=true
```
Firstly, we will use original MainActivity.java to build a simple object. And add native function in MainActivity.java, as following:
```
private native String jniSvmRes(String cmd);
```
Then build your project, and the object file will be created for uses of following step. In the terminal window of Android studio, run following commands(In Windows OS):
```
cd app/src/main
mkdir jni
javah -d jni -classpath $YourSdkPath\platforms\android-24\android.jar;$YourSdkPath\extras\android\support\v7\appcompat\libs\android-support-v7-appcompat.jar;$YourSdkPath\extras\android\support\v4\android-support-v4.jar;..\..\build\intermediates\classes\debug com.example.gksvm.MainActivity
```
By doing that, we will get a java head file under $YourAndroidProjectPath\app\src\main\jni
```
com_example_gksvm_MainActivity.h
```

## Build native code
In the $YourAndroidProjectPath\app\src\main\AndroidManifest.xml, add following code:
```
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```
We have provided OpenCL library and algorithm code under Github, GPU-algorithm/gksvm/app/src/main/jni/. Please copy the following folders into jni:
```
include/CL //This is openCL head file
libknn //This is the code of GPU-SVM-kNN algorithm
libs //This is Dynamic Link Library of OpenCL, that is provided by the vendor of phone device 
libsvm //SVM library that is used in GPU-SVM-kNN algorithm
```
Furthermore, add following files into jni:
```
Android.mk //Configuration for Android NDK
Application.mk //Configuration for Android NDK
common.cpp
common.h
main.cpp
test.cpp
test.h
```
Use given MainActivity.java to replace the previous one in your project.
Define your app-folder in MainActivity.java (Because app-folder is different from phone devices)
In GPU-algorithm/gksvm/app/src/main/assert, there is OpenCL kernel. Copy assert fold into $YourAndroidProjectPath\app\src\main
```
kernel_psvmknn.cl //OpenCL kernel file
train.scale //training data
test.scale //testing data
```
GPU-algorithm/gksvm/app/build.gradle help us to build NDK project, please copy it into $YourAndroidProjectPath\app

## Build your project and run on the phone device




