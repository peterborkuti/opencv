/*
 * util.cpp
 *
 *  Created on: Feb 4, 2014
 *      Author: peter
 */

#include "util.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

cv::MatND Util::twoBinGrayHistogram(const cv::Mat grayImage) {
	//two bins
    int histSize[] = {2};
    // gray varies from 0 to 255
    float range[] = { 0, 255 };

    const float* ranges[] = { range };
    cv::MatND hist;
    // we compute only the 0-th channel
    int channels[] = {0};

    cv::calcHist( &grayImage, 1, channels, cv::Mat(), // do not use mask
             hist, 1, // 1D histogram,
             histSize, ranges,
             true, // the histogram is uniform
             false ); // not preserve previous histogram

#ifdef UTIL_HPP_DEBUG
    std::cout << "histogram:" << hist.at<float>(0, 0) <<
    		"," << hist.at<float>(0, 1) << std::endl;
#endif

    return hist;
}



