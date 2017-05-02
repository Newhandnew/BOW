#include <stdio.h>
#include <iostream>
#include <time.h>
#include <sstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int numBOF = 20;
string imgDir = "imgEnv/";
string imgType = ".jpg";
char numCamera = 1;

/* @function main */
int main( int argc, char** argv )
{
	VideoCapture cap(numCamera);
	if(cap.isOpened() != 1)
	{
		printf("error in capturincd g");
		return -1;
	}
	
	namedWindow("video", 1);
	Mat cameraFrame, imgGray;
	//-- Step 1: Detect the keypoints using SURF Detector
  	int minHessian = 400;
  	Mat imgKeypoints;

  	Ptr<SURF> detector = SURF::create( minHessian );
  	vector<KeyPoint> keypoints;
  	//To store the SURF descriptor of current image
  	Mat descriptor;
	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;

	cout << "press 't' to train" << endl;
	while(1) 
	{
		char key = waitKey(100);
		if(key == 27)
		{
			cout << "exit" << endl;
			return 0;
		}
		else if (key == 116)   // "t"
		{
			cout << "start to capture..." << endl;
			break;
		}
	}
	
	for (int i = 0; i < numBOF; i++)
	{
		clock_t start = clock();
		cap >> cameraFrame;
		cvtColor(cameraFrame, imgGray, COLOR_BGR2GRAY);
		detector->detectAndCompute( imgGray, Mat(), keypoints, descriptor );
		featuresUnclustered.push_back(descriptor);
		clock_t end = clock();
		float spendSeconds = (float)(end - start) / CLOCKS_PER_SEC;

		drawKeypoints( cameraFrame, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
		imshow("video", imgKeypoints);

		stringstream imgName;
		imgName << imgDir << i << imgType;
		imwrite(imgName.str(), cameraFrame);

		cout << (i + 1) * 100 / numBOF << " percent done, time: " << spendSeconds << ", file name: " << imgName.str() << endl;
		if(waitKey(100) == 27)
		{
			cout << "exit" << endl;
			break;
		}
	}

	int dictionarySize = 200;
	TermCriteria tc(CV_TERMCRIT_ITER, 100.0, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
	FileStorage fTrainFeature("trainImgFeature.yml", FileStorage::WRITE);
	fTrainFeature << "train image feature" << featuresUnclustered;
	fTrainFeature.release();

	return 0;
}