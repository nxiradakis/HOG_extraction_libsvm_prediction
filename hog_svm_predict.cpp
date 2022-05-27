#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <opencv2/core/core.hpp>           
#include <algorithm>
#include <opencv2/videoio.hpp>  // Video write
#include "opencv2/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include <omp.h>
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc.hpp"



#include "svm.h" //LIBSVM Library header file
const char* MODEL_FILE = "newsvm3.model"; //saved SVM model


using namespace cv;
using namespace std;

// Global variables
struct svm_model* SVMModel;

/*   //in calling function
if ((SVMModel = svm_load_model(MODEL_FILE)) == 0) {
    fprintf(stderr, "Can't load SVM model %s\n", MODEL_FILE);
    return -2;
}
*/

double computer(const cv::Mat& image, struct svm_model* SVMModel)
{
    Mat gray;
    Mat resized;
    HOGDescriptor hog;
    struct svm_node* svmVec;

    //HOG Descriptor variables
    hog.winSize = Size(128, 128);
    hog.blockSize = Size(16, 16);
    hog.blockStride = Size(8, 8);
    hog.cellSize = Size(8, 8);
    hog.nbins = 9;
    hog.derivAperture = 1;
    hog.winSigma = 4;//#-1
    hog.histogramNormType = HOGDescriptor::L2Hys;
    hog.L2HysThreshold = 2.0000000000000001e-01;
    hog.gammaCorrection = 0;
    hog.nlevels = 64;

    //printf("%d\n",hog.getDescriptorSize());
    cvtColor(image, gray, COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(128, 128));
   
    vector < float > descriptors;

    hog.compute(resized, descriptors, Size(128, 128), Size(8, 8));

    int n = descriptors.size();
    //std::cout << "size: " << descriptors.size() << '\n';

    //Allocating memory for SVM node
    svmVec = (struct svm_node*)malloc((n + 1) * sizeof(struct svm_node));

    for (int j = 0; j < n; j++) {


        svmVec[j].index = j;  // Index starts from 1; Pre-computed kernel starts from 0
        svmVec[j].value = descriptors[j];
    }
    svmVec[n].index = -1;

    double prob_est[10];  // Probability estimates array for 10 classes


    double predictions = svm_predict_probability(SVMModel, svmVec, prob_est); //prediction of label
    free(svmVec); //deallocating dynamic memory 
    //cout << "Prediction: " << endl;
    return predictions;

}