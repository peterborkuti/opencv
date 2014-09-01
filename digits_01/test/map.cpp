/*
 * map.cpp
 *
 *  Created on: Feb 7, 2014
 *      Author: peter
 */

#include <iostream>
#include <map>

typedef unsigned char ValueType;
typedef std::map<ValueType, char> DecoderMap;
typedef std::pair<ValueType, char> PairDecoderMap;
typedef DecoderMap::iterator DecoderMapIterator;

int mapTest(int argc, const char** argv) {

	DecoderMap decoderMap;

	decoderMap[0b01011111u] = '0';
	decoderMap[0b00000011u] = '1';

	ValueType value;

	value = 0b00000011u;
	if (decoderMap.count(value) == 1) {
		std::cout << "OK" << std::endl;
	}

	value = 0b01111111u;
	if (decoderMap.count(value) == 1) {
		std::cout << "BAD" << std::endl;
	}

	value = 0u;
	value |= 1 << 0;
	value |= 1 << 1;
	if (decoderMap.count(value) == 0) {
		std::cout << "BAD" << std::endl;
	}



}


