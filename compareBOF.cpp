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
 
    //open the file to write the resultant descriptor

    FileStorage fs1("descriptor.yml", FileStorage::WRITE);    

    std::vector<KeyPoint> keypoints;        
    //Detect SIFT keypoints (or feature points)
	Mat cameraFrame, imgGray;
	cap >> cameraFrame;
	cvtColor(cameraFrame, imgGray, COLOR_BGR2GRAY);

    detector->detect(cameraFrame, keypoints);
    //To store the BoW (or BoF) representation of the image

    Mat bowDescriptor;        
    //extract BoW (or BoF) descriptor from given image

    bowDE.compute(cameraFrame, keypoints, bowDescriptor);
      
    //write the new BoF descriptor to the file

    fs1 << "test" << bowDescriptor;        

    fs1.release();
}