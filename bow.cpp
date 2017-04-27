#include <stdio.h>
#include <iostream>
#include <time.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
int numBOF = 20;

/* @function main */
int main( int argc, char** argv )
{
	VideoCapture cap(0);
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
  	std::vector<KeyPoint> keypoints;
  	//To store the SURF descriptor of current image
  	Mat descriptor;
	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;

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

		printf("%i percent done, time: %f \n", (i + 1) * 100 / numBOF, spendSeconds);
		if(waitKey(30) == 27)
		{
			printf("exit");
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

	return 0;
}