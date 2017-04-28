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

int main( int argc, char** argv )
{
	VideoCapture cap(0);
	if(cap.isOpened() != 1)
	{
		printf("error in capturincd g");
		return -1;
	}

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
    	// cout << bowDescriptor.size() << endl;
    	bowKeyFrame.push_back(bowDescriptor);
    	// cout << bowDescriptor << endl;
    	// test = bowDescriptor;
    }
    // cout << bowKeyFrame[0] << endl;

    //open the file to write the resultant descriptor
    FileStorage fs1("descriptor.yml", FileStorage::WRITE);    

    // vector<KeyPoint> keypoints;        
    //Detect SIFT keypoints (or feature points)
	Mat cameraFrame, imgGray;
	cap >> cameraFrame;
	cvtColor(cameraFrame, imgGray, COLOR_BGR2GRAY);
    detector->detect(cameraFrame, keypoints);
    //To store the BoW (or BoF) representation of the image
    // Mat bowDescriptor;        
    //extract BoW (or BoF) descriptor from given image
    // vector< vector<int> > pointIdxOfClusters;

    bowDE.compute(cameraFrame, keypoints, bowDescriptor);
    //write the new BoF descriptor to the file
    fs1 << "test" << bowDescriptor;        
    fs1.release();

    // vector<DMatch> matches;
    // matcher -> match(bowDescriptor, matches);
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

	cout << best << endl << minDist << endl;

    // std::vector< std::vector<DMatch> > nn_matches;
    // matcher.knnMatch(bowDescriptor, nn_matches, 1);
    // cout << matches.size() << endl << matches[0].imgIdx << endl;
    // cout << matches[0].trainIdx << endl << matches[1].trainIdx << endl << matches[2].trainIdx << endl << matches[2].trainIdx << endl;
    // std::cout << nn_matches.size() << std::endl;
   //  for (int i=0; i<pointIdxOfClusters.size();i++){
   //  	cout << pointIdxOfClusters[1][i] << endl;
 	 // }


}