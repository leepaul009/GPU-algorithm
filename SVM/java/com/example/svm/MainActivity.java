/*** version 0925 ***/
package com.example.svm;

import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    public static final String LOG_TAG = "ndk_svm";

    String appFolderPath;
    String systemPath;
    String testStr;

    static {
        System.loadLibrary("NDKLib");
    }
    //public native String getStringNative();
    private native void jniSvmTrain(String cmd);
    private native void jniSvmPredict(String cmd);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //TextView view = (TextView) findViewById(R.id.tv);
        //view.setText(getStringNative());

        systemPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/";
        appFolderPath = systemPath+"libsvm/";

        CreateAppFolderIfNeed();
        copyAssetsDataIfNeed();

        testStr = systemPath + " and " + appFolderPath;


        String dataTrainPath = appFolderPath+"b9f3t_54k.scale ";
        String dataPredictPath = appFolderPath+"b9f3p.scale ";
        String modelPath = appFolderPath+"model ";
        String outputPath = appFolderPath+"predict ";

        String svmTrainOptions = "-t 2 -s 0 -c 100 -g 0.1 -e 0.1 -b 1 ";

        long time1, time2, start, end;
        start= System.currentTimeMillis();
        jniSvmTrain(svmTrainOptions + dataTrainPath + modelPath);
        end= System.currentTimeMillis();
        time1 = end - start;

        svmTrainOptions = " -b 1 ";
        start= System.currentTimeMillis();
        jniSvmPredict(svmTrainOptions + dataPredictPath + modelPath + outputPath);
        end= System.currentTimeMillis();
        time2 = end - start;

        String strTime1 = Long.toString(time1);
        String strTime2 = Long.toString(time2);

        TextView testView1 = (TextView) findViewById(R.id.tv1);
        testView1.setText(testStr + svmTrainOptions + dataTrainPath + modelPath);

        TextView testView2 = (TextView)findViewById(R.id.tv2);
        testView2.setText("svm train cost: " + strTime1 + ",svm predict cost:"+ strTime2);
    }

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
        String assetsToCopy[] = {"b9f3t.scale","b9f3p.scale"};

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


