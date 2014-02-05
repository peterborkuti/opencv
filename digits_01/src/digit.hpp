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
#include <string>

typedef std::map<std::string, char> DecoderMap;
typedef std::pair<std::string, char> PairDecoderMap;
typedef DecoderMap::iterator DecoderMapIterator;

class Digit {
public:
	Digit(cv::Rect r);

	std::string read(const cv::Mat grayImage);
	void draw(cv::Mat &image);
	std::string getValue();
	char decode();
private:
	std::vector<Segment> segments;
	cv::Rect rect;
	std::string value;
	bool isDigitized;
	DecoderMap decoderMap;
};


#endif /* DIGIT_HPP_ */
