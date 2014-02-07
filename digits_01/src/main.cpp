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

cv::Mat readImage(int argc, const char** argv) {
    cv::Mat image;
	if( argc != 2)
    {
		image = cv::Mat::zeros( 640, 480, CV_8UC3 ); // black image
    }
	else {
		image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

		if(! image.data )                              // Check for invalid input
		{
			std::cout <<  "Could not open or find the image" << std::endl ;
			image = cv::Mat( 640, 480, CV_8UC3, cv::Scalar(255,255,255) );
		}
    }

	return image;
}

int main(int argc, const char** argv) {
	cv::Mat image = readImage(argc, argv);
	cv::Mat channels[3];
	cv::split(image, channels);
	cv::Mat blue = channels[0];

	Segment hold(cv::Rect(170,108, 111, 51));
	Segment scan(cv::Rect(63,112, 107, 48));
	hold.read(blue);
	hold.draw(image);
	scan.read(blue);
	scan.draw(image);

	std::cout << "scan:" << ((scan.getDigit())?"YES":"NO");
	std::cout << ", hold:" << ((hold.getDigit())?"YES":"NO");
	std::cout << std::endl;


	Digit* d[3];

	d[0] = new Digit(cv::Rect(153, 182, 112, 211));
	d[1] = new Digit(cv::Rect(272, 182, 103, 208));
	d[2] = new Digit(cv::Rect(419, 179, 105, 210));

	for (int i = 0; i < 3; i++) {
		d[i]->read(blue);
		d[i]->draw(image);
	}

	std::cout << d[0]->decode() << d[1]->decode() << "." << d[2]->decode() << std::endl;

	imshow("img", image);

	cv::waitKey(0);
}
