package com.example.gksvm;

import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    public static final String LOG_TAG = "AndroidLibSvm";
    String appFolderPath;
    String systemPath;

    static {
        System.load("/system/vendor/lib/libOpenCL.so");
        System.loadLibrary("jnilibsvm");
    }
    private native String jnifunc();
    private native String jniLocalization(String cmd);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        long time_beg, elapsed_time;
        systemPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/";
        appFolderPath = systemPath+"gksvm/";

        CreateAppFolderIfNeed();
        copyAssetsDataIfNeed();

        String clProgPath = appFolderPath+"kernel_psvmknn.cl ";
        time_beg = System.currentTimeMillis();
        String ett = jniSvmRes(clProgPath);
        elapsed_time = System.currentTimeMillis() - time_beg;

        TextView view = (TextView)findViewById(R.id.mainview);
        view.setText(jnifunc() + ", elapsed time = " + elapsed_time);
    }

    /*
     * Some utility functions
     * */
    private void CreateAppFolderIfNeed(){
        File folder = new File(appFolderPath);
        if (!folder.exists()) {
            folder.mkdir();
            Log.d(LOG_TAG,"Appfolder is not existed, create one");
        } else {
            Log.w(LOG_TAG,"WARN: Appfolder has not been deleted");
        }
    }

    private void copyAssetsDataIfNeed(){
        String assetsToCopy[] =  {"kernel_psvmknn.cl","b9f3p.scale","b9f3t.scale"};
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
