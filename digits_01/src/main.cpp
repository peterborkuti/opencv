/*
 * main.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "segment.hpp"
#include "digit.hpp"

#include <iostream>
#include <fstream>
#include <ctime>
#include <stdio.h>

int main(int argc, const char** argv) {
	cv::Mat blackImg = cv::Mat::zeros( 100, 100, CV_8UC3 );
	cv::Mat whiteImg = cv::Mat( 100, 100, CV_8UC3, cv::Scalar(255,255,255) );
	cv::Mat img1, img2;
	cv::cvtColor(blackImg, img1, CV_BGR2GRAY);
	cv::cvtColor(whiteImg, img2, CV_BGR2GRAY);

	Segment s(cv::Rect(10,10, 10, 2));
	std::cout << s.read(img1) << std::endl;
	std::cout << s.read(img2) << std::endl;

	Digit d(cv::Rect(5, 5, 80, 80));
	std::cout << d.read(img2) << std::endl;
	std::cout << d.read(img1) << std::endl;
	//d.draw(blackImg);


	imshow("img", blackImg);

	while( cv::waitKey(20) == -1 ) {
	}

}


