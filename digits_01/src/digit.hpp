/*
 * digit.hpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#ifndef DIGIT_HPP_
#define DIGIT_HPP_

#include "opencv2/core/core.hpp"
#include "segment.hpp"
#include <vector>

class Digit {
public:
	Digit(cv::Rect r);

	int read(const cv::Mat grayImage);
	void draw(cv::Mat &image);
	short getValue();

private:
	std::vector<Segment> segments;
	cv::Rect rect;
	int value;
	bool isDigitized;
};


#endif /* DIGIT_HPP_ */
