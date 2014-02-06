/*
 * util.hpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#ifndef UTIL_HPP_
#define UTIL_HPP_


#include "opencv2/core/core.hpp"

class Util {
public:
	static cv::MatND twoBinGrayHistogram(const cv::Mat grayImage);

};




#endif /* UTIL_HPP_ */
