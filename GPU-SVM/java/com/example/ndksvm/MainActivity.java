//version.2016.11.04 22:20
package com.example.ndksvm;


import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


public class MainActivity extends Activity {
    public static final String LOG_TAG = "IBK-Svm";//Inside building Kernel

    String appFolderPath;
    String systemPath;

    // link jni library
    static {
        System.load("/system/lib/libOpenCL.so");
        //System.load("/vendor/lib/egl/libGLES_mali.so");
        System.loadLibrary("jnilibsvm");
    }

    // connect the native functions
    private native String jniSvmRes(String cmd);
    private native void jniSvmTrain(String cmd);
    private native void jniSvmPredict(String cmd);


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        systemPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/";
        appFolderPath = systemPath+"libsvm/";
        long train_time, predict_time, start, end;

        // 1. create necessary folder to save model files
        CreateAppFolderIfNeed();
        copyAssetsDataIfNeed();

        // 2. assign model/output paths
        String dataTrainPath    = appFolderPath+"b9f3t_54k.scale ";
        String dataPredictPath  = appFolderPath+ "b9f3p.scale ";
        String modelPath        = appFolderPath+"model ";
        String outputPath       = appFolderPath+"predict ";
        //clProgPath       = appFolderPath+ "inbuildMatrix.cl ";

        // 3. make SVM train
        start = System.currentTimeMillis();
        String svmTrainOptions = "-t 2 -s 0 -c 100 -g 0.1 -e 0.1 -b 1 ";
        jniSvmTrain(svmTrainOptions + dataTrainPath + modelPath);
        end = System.currentTimeMillis();

        train_time = end - start;

        // 4. make SVM predict
        svmTrainOptions = " -b 1 ";
        start = System.currentTimeMillis();
        jniSvmPredict(svmTrainOptions + dataPredictPath + modelPath + outputPath);
        end = System.currentTimeMillis();

        predict_time = end - start;

        String s1 = Long.toString(train_time);
        String s2 = Long.toString(predict_time);

        TextView view = (TextView)findViewById(R.id.tv1);
        view.setText("Train cost: " + s1 + ", Test cost: " + s2);
    }

    /*
    * Some utility functions
    * */
    private void CreateAppFolderIfNeed(){
        // 1. create app folder if necessary
        File folder = new File(appFolderPath);

        if (!folder.exists()) {
            folder.mkdir();
            Log.d(LOG_TAG,"Appfolder is not existed, create one");
        } else {
            Log.w(LOG_TAG,"WARN: Appfolder has not been deleted");
        }
    }

    private void copyAssetsDataIfNeed(){
        //String targetPath[] = {C.systemPath+C.INPUT_FOLDER+C.INPUT_PREFIX+AudioConfigManager.inputConfigTrain+".wav", C.systemPath+C.INPUT_FOLDER+C.INPUT_PREFIX+AudioConfigManager.inputConfigPredict+".wav",C.systemPath+C.INPUT_FOLDER+"SomeoneLikeYouShort.mp3"};
        String assetsToCopy[] = {"inbuildMatrix.cl", "b9f3t.scale","b9f3p.scale"};

        for(int i=0; i<assetsToCopy.length; i++){
            String from = assetsToCopy[i];
            String to = appFolderPath+from;

            // 1. check if file exist
            File file = new File(to);
            if(file.exists()){
                Log.d(LOG_TAG, "copyAssetsDataIfNeed: file exist, no need to copy:"+from);
            } else {
                // do copy
                boolean copyResult = copyAsset(getAssets(), from, to);
                Log.d(LOG_TAG, "copyAssetsDataIfNeed: copy result = "+copyResult+" of file = "+from);
            }
        }
    }

    private boolean copyAsset(AssetManager assetManager, String fromAssetPath, String toPath) {
        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open(fromAssetPath);
            new File(toPath).createNewFile();
            out = new FileOutputStream(toPath);
            copyFile(in, out);
            in.close();
            in = null;
            out.flush();
            out.close();
            out = null;
            return true;
        } catch(Exception e) {
            e.printStackTrace();
            Log.e(LOG_TAG, "[ERROR]: copyAsset: unable to copy file = "+fromAssetPath);
            return false;
        }
    }

    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }

}
