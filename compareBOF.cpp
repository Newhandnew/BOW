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
using namespace std;

int numBOF = 20;
string imgDir = "imgEnv/";
string imgType = ".jpg";
char numCamera = 1;

int main( int argc, char** argv )
{
	VideoCapture cap(numCamera);
	if(cap.isOpened() != 1)
	{
		printf("error in capturincd g");
		return -1;
	}
	namedWindow("camera", 1);
	Mat dictionary; 
    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();    

    //create a nearest neighbor matcher

	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create SURF feature point extracter

    int minHessian = 400;
  	Ptr<SURF> detector = SURF::create( minHessian );
    //create Sift descriptor extractor
  	Ptr<SURF> extractor = SURF::create( minHessian );
    //create BoF (or BoW) descriptor extractor

    BOWImgDescriptorExtractor bowDE(extractor, matcher);
    //Set the dictionary with the vocabulary we created in the first step

    bowDE.setVocabulary(dictionary);
 
 	vector<Mat> bowKeyFrame;
	Mat imgCurrent;
	vector<KeyPoint> keypoints; 
    Mat bowDescriptor;     
    Mat test;   
    //extract BoW (or BoF) descriptor from given image
    // vector< vector<int> > pointIdxOfClusters;

    for (int i = 0; i < numBOF; i++)
    {
    	stringstream imgName;
		imgName << imgDir << i << imgType;
		imgCurrent = imread(imgName.str(), CV_LOAD_IMAGE_GRAYSCALE);
    	detector->detect(imgCurrent, keypoints);
    	bowDE.compute(imgCurrent, keypoints, bowDescriptor);
    	bowKeyFrame.push_back(bowDescriptor);
    }

    //open the file to write the resultant descriptor
    FileStorage fs1("descriptor.yml", FileStorage::WRITE);    
    fs1 << "test" << bowDescriptor;        
    fs1.release();

	Mat cameraFrame, imgGray;
	double limitMinDist = 0.15;		// need to be discussed

	cout << "press 't' to test image" << endl;
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
			cout << "start to test..." << endl;
			clock_t start = clock();
			cap >> cameraFrame;
			cvtColor(cameraFrame, imgGray, COLOR_BGR2GRAY);
		    detector->detect(cameraFrame, keypoints);
		    clock_t tSURFEnd = clock();
		    float spendSeconds = (float)(tSURFEnd - start) / CLOCKS_PER_SEC;
		    cout << "SURF time: " << spendSeconds << endl;
		    //To store the BoW (or BoF) representation of the image
		    // Mat bowDescriptor;        
		    //extract BoW (or BoF) descriptor from given image
		    // vector< vector<int> > pointIdxOfClusters;

		    bowDE.compute(cameraFrame, keypoints, bowDescriptor);
		    clock_t tBOWEnd = clock();
		    spendSeconds = (float)(tBOWEnd - tSURFEnd) / CLOCKS_PER_SEC;
		    cout << "BOW compute time: " << spendSeconds << endl;
	    	int best=0;
			double minDist = 999999999;
			for (size_t i=0; i<bowKeyFrame.size(); i++)
			{
			     double dist = norm(bowKeyFrame[i], bowDescriptor); //calc L2 distance
			     if (dist < minDist) // keep the one with smallest distance
			     {
			          minDist = dist;
			          best = i;
			     }
			}

			if(minDist < limitMinDist)
			{
				clock_t end = clock();
				spendSeconds = (float)(end - start) / CLOCKS_PER_SEC;
				cout << "success: "<< best << ", minimum distance: " << minDist << ", spend time: " << spendSeconds << endl;
				imshow("camera", cameraFrame);
				stringstream imgName;
				imgName << imgDir << best << imgType;
				Mat imgMatch;
				imgMatch = imread(imgName.str(), CV_LOAD_IMAGE_COLOR);
				imshow("match", imgMatch);
			}
			else
			{
				cout << "fail" << " minimum distance: " << minDist << endl;
			}
		}
	}

}