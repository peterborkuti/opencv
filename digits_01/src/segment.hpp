/*
 * segment.hpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#ifndef SEGMENT_HPP_
#define SEGMENT_HPP_

#include "opencv2/core/core.hpp"

class Segment {
public:
	Segment(cv::Rect r);

	short read(const cv::Mat grayImage);
	void print(cv::Mat image);
	void draw(cv::Mat img);
	short getDigit();

private:
	cv::Rect rect;
	short digit;
	bool isDigitized;
};


#endif /* SEGMENT_HPP_ */
