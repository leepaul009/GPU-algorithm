# GPU-algorithm
The pattern recognition algorithms optimized with GPU and OpenCL. The algorithms are implemented with Android NDK and will run with Android application. To run the programs, the Android OpenCL library file is needed. In this case, the lib file is under "\jni\libs\libOpenCL.so"(Android Project) and "/system/lib/libOpenCL.so"(Phone Device), which is lib file from smart phone, XiaoMi 2.


The algoriths include:
1)GPU-SVM, parallel version of kNN with a GPU-based run-time acceleration. 
2)SVM, serial version of kNN, reference:http://www.csie.ntu.edu.tw/~cjlin/libsvm/.
3)GPU-SVM-kNN, using a SVM-based decision-maker to refine the outcome of kNN.
4)GPU-kNN, including GPU-Radix-kNN and GPU-Bitonic-kNN.
