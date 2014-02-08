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
	value = 0;
	isDigitized = false;
	rect = r;
	strValue = "";
	strValues[0] = "OFF";
	strValues[1] = "ON";
}

Segment::Segment(cv::Rect r, std::string stringValueOn, std::string stringValueOff) {
	value = 0;
	isDigitized = false;
	rect = r;
	strValue = "";
	strValues[0] = stringValueOff;
	strValues[1] = stringValueOn;
}

std::string Segment::read(const cv::Mat grayImg) {
	cv::Mat roi(grayImg, rect);

	cv::MatND hist = Util::twoBinGrayHistogram(roi);

	value = (hist.at<float>(0, 0) > 2 * hist.at<float>(0, 1)) ? 1 : 0;
	strValue = strValues[value];

	return strValue;
}

void Segment::print(cv::Mat image) {

}

void Segment::draw(cv::Mat image) {
	cv::rectangle(image, rect, cv::Scalar(0, 255, 0));
}

short Segment::getValue() {
	return value;
}

std::string Segment::getStringValue() {
	return strValue;
}
