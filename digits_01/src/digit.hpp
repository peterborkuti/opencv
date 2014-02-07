/*
 * digit.hpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#ifndef DIGIT_HPP_
#define DIGIT_HPP_

#define DIGIT_HPP_DEBUG

#include "opencv2/core/core.hpp"
#include "segment.hpp"
#include <map>
#include <string>

typedef unsigned int ValueType;
typedef std::map<ValueType, char> DecoderMap;
typedef std::pair<ValueType, char> PairDecoderMap;
typedef DecoderMap::iterator DecoderMapIterator;


class Digit {
public:
	Digit(cv::Rect r);

	ValueType read(const cv::Mat grayImage);
	void draw(cv::Mat &image);
	ValueType getValue();
	char decode();
	void printMap();
private:
	std::vector<Segment> segments;
	cv::Rect rect;
	ValueType value;
	bool isDigitized;
	DecoderMap decoderMap;
};


#endif /* DIGIT_HPP_ */
