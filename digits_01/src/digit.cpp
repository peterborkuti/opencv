/*
 * digit.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#include "digit.hpp"


Digit::Digit(cv::Rect r) {
	rect = r;
	value = "00000000";
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

	//segments: top|middle|botton|lefttop|leftbottom|righttop|rightbottom
	decoderMap.insert(PairDecoderMap("1011111",'0'));
	decoderMap.insert(PairDecoderMap("0000011",'1'));
	decoderMap.insert(PairDecoderMap("1110110",'2'));
	decoderMap.insert(PairDecoderMap("1110011",'3'));
	decoderMap.insert(PairDecoderMap("0101001",'4'));
	decoderMap.insert(PairDecoderMap("1111001",'5'));
	decoderMap.insert(PairDecoderMap("1111101",'6'));
	decoderMap.insert(PairDecoderMap("1000011",'7'));
	decoderMap.insert(PairDecoderMap("1111111",'8'));
	decoderMap.insert(PairDecoderMap("1111011",'9'));
	decoderMap.insert(PairDecoderMap("0100000",'-'));

}

std::string Digit::read(cv::Mat grayImage) {

	char bits[7];
	for (int i = 0; i < 7; i++) {
		bits[i] = (segments[i].read(grayImage) == 0) ? '0' : '1';
	}

	std::string s(bits);
	value = s;
	isDigitized = true;

	return value;
}

void Digit::draw(cv::Mat &image) {

	for (int i = 0; i < 7; i++) {
		segments[i].draw(image);
	}

	cv::rectangle(image, rect, cv::Scalar(0, 0, 255));
}

std::string Digit::getValue() {
	return value;
}

char Digit::decode() {
	DecoderMapIterator it = decoderMap.find(value);
	if (it == decoderMap.end()) {
		return ' ';
	}
	else {
		return it->second;
	}
}
