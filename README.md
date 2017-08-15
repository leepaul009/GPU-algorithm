# GPU-algorithm
The pattern recognition algorithms optimized with GPU and OpenCL. The algorithms are implemented with Android NDK and will run with Android application. To run the programs, the Android OpenCL library file is needed. In this case, the lib file is under "\jni\libs\libOpenCL.so"(Android Project) and "/system/lib/libOpenCL.so"(Phone Device), which is lib file from smart phone, XiaoMi 2.


The algoriths include:
1)GPU-SVM, parallel version of kNN with a GPU-based run-time acceleration. 
2)SVM, serial version of kNN, reference:http://www.csie.ntu.edu.tw/~cjlin/libsvm/.
3)GPU-SVM-kNN, using a SVM-based decision-maker to refine the outcome of kNN.
4)GPU-kNN, including GPU-Radix-kNN and GPU-Bitonic-kNN.


How to install:

Install java-sdk
Install android studio
Download android native develop kit, android-ndk-r15c-windows-x86_64.zip
Create android project, named 'gksvm'

A. Firstly, we will set ndk for android project 

A1) The setting of android studio 
For android studio, in menu Setting->Tools->External Tools, add a new item:
	Name: javah
	Program: $YourJavaPath\bin\javah.exe
	Parameters: -classpath $Classpath$ -v -jni $FileClass$
	Working directory: $YourAndroidProjectPath$\app\src\main\jni
And add another new item for External Tools:
	Name: ndk-build
	Program: $YourNdkPath\ndk-build.cmd
	Working directory: $YourAndroidProjectPath$\app\src\main\jni

A2) Create Java head file
Edit $YourProjectFolder\local.properties, in the end of file, add following:
	ndk.dir=$YourNdkPath\\android-ndk-r15c
	sdk.dir=$YourSdkPath\\Sdk 
Edit $YourProjectFolder\gradle.properties, in the end of file, add following:
	android.useDeprecatedNdk=true
Firstly, we will use original MainActivity.java. Add native function in MainActivity.java, as following:
	private native String jnifunc();
	private native String jniSvmRes(String cmd);
Then build your project, and the object file will be created for following step
In the terminal window of android studio, run following command(In the case of Windows OS):
	cd app/src/main
	mkdir jni
	javah -d jni -classpath $YourSdkPath\platforms\android-24\android.jar;$YourSdkPath\extras\android\support\v7\appcompat\libs\android-support-v7-appcompat.jar;$YourSdkPath\extras\android\support\v4\android-support-v4.jar;..\..\build\intermediates\classes\debug com.example.gksvm.MainActivity
After that, you will get a java head file under $YourAndroidProjectPath\app\src\main\jni
	com_example_gksvm_MainActivity.h

B Build native code
In the $YourAndroidProjectPath\app\src\main\AndroidManifest.xml, add following code:
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
We provide library and implementation code under Github, app\src\main\jni
Copy the following folders into jni:
	include/CL //openCL head file
	libknn //library of knn
	libs //drynamic lib of opencl that is provided by smartphone vendor
	libsvm //library of svm
Add following files into jni:
	Android.mk
	Application.mk
	common.cpp
	common.h
	main.cpp
	test.cpp
	test.h
Use given MainActivity.java to replace the one in your project
Define your app folder in MainActivity.java (Because app folder is up to phone device)
Copy assert fold into $YourAndroidProjectPath\app\src\main
	kernel_psvmknn.cl //OpenCL kernel file
Copy the file build.gradle into $YourAndroidProjectPath\app
Build your project
