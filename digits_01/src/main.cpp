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
//The original picture was 640x480, so the digit's places was adopt to
//this size
const cv::Size* ORIG_SIZE = new cv::Size(640,480);
const cv::Size* NEW_SIZE = new cv::Size(160, 120);

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
	cv::resize(image, image, *NEW_SIZE);

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

void openCamera(cv::VideoCapture& cap, int argc, const char** argv) {
	cv::CommandLineParser parser(argc, argv, keys);
	int camNum = parser.get<int>("1");

	cap.open(camNum);

	if (!cap.isOpened()) {
		//help();
		std::cout << "***Could not initialize capturing...***\n";
		std::cout << "Current parameter's value: \n";
		exit (-1);
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 160);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
}

class RectResizer {
private:
	float dx, dy;
public:
	RectResizer(cv::Size origSize, cv::Size newSize) {
		dx = (float)newSize.width / (float)origSize.width;
		dy = (float)newSize.height / (float)origSize.height;
	}
	cv::Rect resize(cv::Rect r) {
		cv::Rect s;
		s.x = ((float) r.x) * dx;
		s.y = ((float)r.y) * dy;
		s.height = ((float)r.height) * dy;
		s.width = ((float)r.width) * dx;
		return s;
	}
};


int main(int argc, const char** argv) {

	RectResizer resizer(*ORIG_SIZE, *NEW_SIZE);
	Segment scan(resizer.resize(cv::Rect(63, 112, 107, 48)), "SCAN", "    ");
	Segment hold(resizer.resize(cv::Rect(170, 108, 111, 51)), "HOLD", "    ");
	scan.setThreshold(0.01);
	hold.setThreshold(0.01);

	Digit* d[NUM_OF_DIGITS];

	d[0] = new Digit(resizer.resize(cv::Rect(45, 182, 112, 214)));
	d[1] = new Digit(resizer.resize(cv::Rect(156, 182, 112, 214)));
	d[2] = new Digit(resizer.resize(cv::Rect(272, 182, 103, 214)));
	d[3] = new Digit(resizer.resize(cv::Rect(419, 179, 105, 214)));

	cv::Mat image;

	/*
	 //From pictures
	 for (int im = 1; im < argc; im++) {
	 image = readImage(argc, argv, im);

	 readData(image, d, scan, hold);
	 printData(d, scan, hold);
	 showImage(image, d, scan, hold);

	 cv::waitKey(0);
	 }
	*/


	cv::VideoCapture cap;

	openCamera(cap, argc, argv);

	bool paused = false;

	cv::Mat frame;

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
