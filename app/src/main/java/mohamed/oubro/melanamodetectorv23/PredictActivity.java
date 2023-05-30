package mohamed.oubro.melanamodetectorv23;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class PredictActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_IMAGE_PICK = 2;
    private static final int IMAGE_MEAN = 0;
    private static final float IMAGE_STD = 255.0f;

    private ImageView mImageView;
    private TextView mResultLabel;
    private Bitmap mBitmap;
    private Interpreter mInterpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_predict);

        // Initialize the views
        mImageView = findViewById(R.id.image_view);
        mResultLabel = findViewById(R.id.result_label);

        // Load the TFLite model
        try {
            mInterpreter = new Interpreter(loadModelFile(), null);
        } catch (IOException e) {
            Log.e("SkinCancerDetection", "Failed to load model: " + e.getMessage());
            finish();
        }
    }

    // Load the TFLite model from assets folder
    private ByteBuffer loadModelFile() throws IOException {
        // old modedl model_cnn_01 name
        AssetFileDescriptor fileDescriptor = getAssets().openFd("model_cnn_92.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    // Take photo using camera
    public void takePhoto(View view) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, REQUEST_IMAGE_CAPTURE);
        } else {
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
        }
    }

    // Choose image from gallery
    public void chooseImage(View view) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_IMAGE_PICK);
        } else {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, REQUEST_IMAGE_PICK);
        }
    }

    // Handle the result of the photo taking or image choosing activity
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            switch (requestCode) {
                case REQUEST_IMAGE_CAPTURE:
                    // Get the captured photo
                    Bundle extras = data.getExtras();
                    if (extras != null) {
                        mBitmap = (Bitmap) extras.get("data");
                        // Set the photo to the image view
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case REQUEST_IMAGE_PICK:
                    // Get the chosen image
                    Uri uri = data.getData();
                    try {
                        mBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                        // Set the image to the image view
                        mImageView.setImageBitmap(mBitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    break;
            }
        }
    }

    // Detect the type of skin cancer
    public void predictImage(View view) {
        if (mBitmap == null) {
            Toast.makeText(this, "Please select an image", Toast.LENGTH_SHORT).show();
            return;
        }

        // Preprocess the image
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(mBitmap);

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                        .build();
        tensorImage = imageProcessor.process(tensorImage);
        ByteBuffer byteBuffer = tensorImage.getBuffer();

        // Run the inference
        float[][] output = new float[1][3]; // Update the output array size for 3 classes
        mInterpreter.run(byteBuffer, output);

        // Calculate the softmax probabilities
        float[] probabilities = softmax(output[0]);

        // Find the class with the highest probability
        int predictedClassIndex = argmax(probabilities);

        // Show the result
        if (predictedClassIndex == 0) {
            String classPercentages = String.format("Benign: %.2f%%",probabilities[predictedClassIndex] * 100);
            mResultLabel.setText(classPercentages);
            Toast.makeText(this, "Prediction: Benign", Toast.LENGTH_SHORT).show();
        } else if (predictedClassIndex == 1) {
            String classPercentages = String.format("Malignant:  %.2f%%",probabilities[predictedClassIndex] * 100);
            mResultLabel.setText(classPercentages);
            Toast.makeText(this, "Prediction: Malignant", Toast.LENGTH_SHORT).show();
        } else if (predictedClassIndex == 2) {
            String classPercentages = String.format("Carcinoma: %.2f%%",probabilities[predictedClassIndex] * 100);
            mResultLabel.setText(classPercentages);
            Toast.makeText(this, "Prediction: Carcinoma", Toast.LENGTH_SHORT).show();
        }



    }
    private float[] softmax(float[] input) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : input) {
            if (value > max) {
                max = value;
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.exp(input[i] - max);
            sum += input[i];
        }
        for (int i = 0; i < input.length; i++) {
            input[i] /= sum;
        }
        return input;
    }

    private int argmax(float[] input) {
        int maxIndex = 0;
        float max = input[0];
        for (int i = 1; i < input.length; i++) {
            if (input[i] > max) {
                max = input[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}







