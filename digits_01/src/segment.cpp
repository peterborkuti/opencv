/*
 * segment.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */


#include "segment.hpp"
#include "util.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

Segment::Segment(cv::Rect r) {
	digit = 0;
	isDigitized = false;
	rect = r;
}



short Segment::read(const cv::Mat grayImg) {
	cv::Mat roi(grayImg, rect);

	cv::MatND hist = Util::twoBinGrayHistogram(grayImg);

	digit = (hist.at<float>(0, 0) > hist.at<float>(0, 1)) ? 1 : 0;
	return digit;
}

void Segment::print(cv::Mat image) {

}

void Segment::draw(cv::Mat image) {
	cv::rectangle(image, rect, cv::Scalar(255, 0, 0));
}

short Segment::getDigit() {
	return digit;
}

