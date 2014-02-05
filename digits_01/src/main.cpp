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

	Segment s(cv::Rect(10,10, 10, 2));
	std::cout << s.read(blue) << std::endl;

	Digit d(cv::Rect(153, 182, 112, 211));
	std::cout << d.read(blue) << d.decode() <<   std::endl;
	d.draw(image);

	Digit d2(cv::Rect(272, 182, 103, 208));
	std::cout << d2.read(blue) << d.decode() << std::endl;
	d2.draw(image);

	Digit d3(cv::Rect(419, 179, 105, 210));
	std::cout << d3.read(blue) << d.decode() <<std::endl;
	d3.draw(image);

	imshow("img", image);

	cv::waitKey(0);
}
