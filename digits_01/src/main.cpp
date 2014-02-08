/*
 * main.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#define UTIL_HPP_DEBUG true

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "segment.hpp"
#include "digit.hpp"

#include <iostream>

const int NUM_OF_DIGITS = 4;

cv::Mat readImage(int argc, const char** argv, int i) {
	cv::Mat image;
	if (argc < 2 || i >= argc) {
		image = cv::Mat::zeros(640, 480, CV_8UC3); // black image
	} else {
		image = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);   // Read the file

		if (!image.data)                              // Check for invalid input
		{
			std::cout << "Could not open or find the image" << std::endl;
			image = cv::Mat(640, 480, CV_8UC3, cv::Scalar(255, 255, 255));
		}
	}

	return image;
}

void readData(const cv::Mat image, Digit * d[], Segment& scan, Segment& hold) {
	cv::Mat channels[3];
	cv::split(image, channels);
	cv::Mat blue = channels[0];
	cv::Mat binary;
	//cv::threshold( blue, binary, threshold_value, max_BINARY_value,threshold_type );
	cv::threshold(blue, binary, 145, 255, cv::THRESH_BINARY);

	for (int i = 0; i < NUM_OF_DIGITS; i++) {
		d[i]->read(binary);
	}

	scan.read(binary);
	hold.read(binary);
	imshow("binary", binary);

}

void printData(Digit * d[], Segment scan, Segment hold) {
	std::cout << scan.getStringValue() << hold.getStringValue();
	std::cout << d[0]->decode() << d[1]->decode() << d[2]->decode();
	std::cout << "." << d[3]->decode() << std::endl;
}

void showImage(cv::Mat image, Digit * d[], Segment scan, Segment hold) {
	hold.draw(image);
	scan.draw(image);

	for (int i = 0; i < NUM_OF_DIGITS; i++) {
		d[i]->draw(image);
	}

	imshow("img", image);
}

const char* keys = { "{1|  | 0 | camera number}" };

int main(int argc, const char** argv) {

	Segment scan(cv::Rect(63, 112, 107, 48), "SCAN", "    ");
	Segment hold(cv::Rect(170, 108, 111, 51), "HOLD", "    ");
	scan.setThreshold(0.01);
	hold.setThreshold(0.01);

	Digit* d[NUM_OF_DIGITS];

	d[0] = new Digit(cv::Rect(40, 182, 112, 211));
	d[1] = new Digit(cv::Rect(153, 182, 112, 211));
	d[2] = new Digit(cv::Rect(272, 182, 103, 208));
	d[3] = new Digit(cv::Rect(419, 179, 105, 210));

	/* From pictures
	 for (int im = 1; im < argc; im++) {
	 cv::Mat image = readImage(argc, argv, im);

	 readData(image, d, scan, hold);
	 printData(d, scan, hold);
	 showImage(image, d, scan, hold);

	 cv::waitKey(0);
	 }
	 */
	cv::VideoCapture cap;

	cv::CommandLineParser parser(argc, argv, keys);
	int camNum = parser.get<int>("1");

	cap.open(camNum);

	if (!cap.isOpened()) {
		//help();
		std::cout << "***Could not initialize capturing...***\n";
		std::cout << "Current parameter's value: \n";
		return (-1);
	}
	bool paused = false;

	cv::Mat frame;
	cv::Mat image;
	for (;;) {
		if (!paused) {
			cap >> frame;
			if (frame.empty()) {
				break;
			}
		}

		frame.copyTo(image);
		readData(image, d, scan, hold);
		printData(d, scan, hold);
		showImage(image, d, scan, hold);

		char c = (char) cv::waitKey(10);
		if (c == 27)
			break;
		switch (c) {
		case 'p':
			paused = !paused;
			break;
		default:
			;
		}
	}

	return 0;

}
