/*
 * digit.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#include "digit.hpp"

Digit::Digit(cv::Rect r) {
	rect = r;
	value = 0;
	isDigitized = false;

	int h = r.height / 7;
	int w = r.width / 3;
	int h2 = 2*h;

	//horiz top
	segments.push_back(Segment(cv::Rect(r.x+w, r.y+0*h, w, h )) );
	//horiz middle
	segments.push_back(Segment(cv::Rect(r.x+w, r.y+3*h, w, h )) );
	//horiz bottom
	segments.push_back(Segment(cv::Rect(r.x+w, r.y+6*h, w, h )) );

	//top left
	segments.push_back(Segment(cv::Rect(r.x, r.y+1*h, w, h2 )) );
	//bottom left
	segments.push_back(Segment(cv::Rect(r.x, r.y+4*h, w, h2 )) );
	//top right
	segments.push_back(Segment(cv::Rect(r.x+2*w, r.y+1*h, w, h2 )) );
	//bottom right
	segments.push_back(Segment(cv::Rect(r.x+2*w, r.y+4*h, w, h2 )) );

}

int Digit::read(cv::Mat grayImage) {

	int sum = 0; int multiplier = 1;
	for (int i = 0; i < 7; i++) {
		sum += segments[i].read(grayImage) * multiplier;
		multiplier *= 2;
	}

	value = sum;
	isDigitized = true;

	return value;
}

void Digit::draw(cv::Mat &image) {

	for (int i = 0; i < 7; i++) {
		segments[i].draw(image);
	}

	cv::rectangle(image, rect, cv::Scalar(0, 0, 255));
}

short Digit::getValue() {
	return value;
}
