#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <string>
#include <fstream>

using namespace cv;

int MIN_AREA = 200;
double B = 0.95;
double A = 0.3;

/*
	Hue values of basic colors
	Orange  0-22
	Yellow 22- 38
	Green 38-75
	Blue 75-130
	Violet 130-160
	Red 160-179
 */

int hueLow = 160, hueHigh = 179,
	satLow = 30, satHigh = 200,
	valLow = 30, valHigh = 200;

int thresh = 100;

string winIn = "video";
string winOut = "output";
string winSetup = "setup";

static void help()
{
	printf(
			"\n"
			"Color detection. Default is red. Use trackbars to set.\n"
			"Hue values of basic colors\n"
			"Orange  0-22\n"
			"Yellow 22- 38\n"
			"Green 38-75\n"
			"Blue 75-130\n"
			"Violet 130-160\n"
			"Red 160-179\n"
		);
}

void putTextInBox(Mat &img, std::string s, Point p) {
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	int baseline = 0;
	String text = s;

	Size textSize = getTextSize(text, fontFace, fontScale, thickness,
			&baseline);
	Point textOrg(p);
	rectangle(img, textOrg,
			textOrg + Point(textSize.width, textSize.height + 2 * baseline),
			Scalar(0, 0, 0), CV_FILLED);
	putText(img, text, textOrg + Point(0, textSize.height + baseline), fontFace,
			fontScale, Scalar::all(255), thickness, 8);
}

static void crossHair(Mat& img, Scalar color=Scalar(0, 255, 0)) {
	line(img, Point(0, 0), Point(img.cols - 1, img.rows - 1), color);
	line(img, Point(0, img.rows - 1), Point(img.cols - 1, 0), color);
}

static void setupGUI() {
	namedWindow(winIn, 1);
	namedWindow(winOut, 1);
	namedWindow(winSetup, 1);

	createTrackbar("hue low", winSetup, &hueLow, 180);
	createTrackbar("hue high", winSetup, &hueHigh, 180);
	createTrackbar("saturation low", winSetup, &satLow, 255);
	createTrackbar("saturation high", winSetup, &satHigh, 255);
	createTrackbar("value low", winSetup, &valLow, 255);
	createTrackbar("value high", winSetup, &valHigh, 255);

}

static void init(int argc, char** argv, VideoCapture& cap) {
	help();

	if (argc < 2)
	{
		cap.open(1);
	}
	else
	{
		cap.open(std::string(argv[1]));
	}
	if (!cap.isOpened())
	{
		printf("\nCan not open camera or video file\n");
		exit(-1);
	}

	Mat tmp_frame;

	cap >> tmp_frame;
	if (!tmp_frame.data)
	{
		printf("can not read data from the video source\n");
		exit(-1);
	}

	printf("image height:%d\n", tmp_frame.rows);

}

static void detectColor(const Mat img, Mat& outImg) {
	Scalar lowThresh(hueLow, satLow, valLow);
	Scalar highThresh(hueHigh, satHigh, valHigh);

	vector<Mat> channels;

	split(img,channels);

	for (uint i = 0; i < channels.size(); i++) {
		equalizeHist(channels[i], channels[i]);
	}

	merge(channels, outImg);

	GaussianBlur(outImg, outImg, Size(3,3), 0, 0);
	cvtColor(outImg, outImg, CV_BGR2HSV);
	inRange(outImg, lowThresh, highThresh, outImg);
	//erode(outImg, outImg, Mat());
	dilate(outImg, outImg, Mat(), Point(-1, -1), 1);
	GaussianBlur(outImg, outImg, Size(3,3), 0, 0);

}

static void findMassCenter(const Mat src_gray, Mat& canny_output) {
	  vector<vector<Point> > contours;
	  vector<Vec4i> hierarchy;

	  /// Detect edges using canny
	  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
	  /// Find contours
	  findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );


	  /// Get the biggest area
	  double maxArea = 0;
	  int maxIndex = -1;
	  for( uint i = 0; i < contours.size(); i++) {
		  double area = contourArea(contours[i]);
		  if (area > maxArea) {
			  maxArea = area;
			  maxIndex = i;
		  }
	  }

	  if (maxArea == 0) {
		  return;
	  }

	  Moments mu = moments(contours[maxIndex], false);
	  ///  Get the mass center
	  Point2f mc(mu.m10/mu.m00 , mu.m01/mu.m00 );

	  circle(canny_output, mc, 5, Scalar(255, 0, 0));

}

static void save(std::string fileName) {
	std::ofstream oFile;

	//saving data
	oFile.open(fileName.c_str(), std::ios_base::app);

	oFile << hueLow << " " << hueHigh << " ";
	oFile << satLow << " " << satHigh << " ";
	oFile << valLow << " " << valHigh << std::endl;

	oFile.flush();
	oFile.close();
}

static void loadLastLine(std::string fileName) {
	std::ifstream iFile;
	std::string line;

	iFile.open(fileName.c_str());
	while (iFile.good()) {
		std::string tmp;
		getline(iFile, tmp);
		if (!tmp.empty()) {
			line = tmp;
		}
	}
	iFile.close();
	std::stringstream s(line);

	printf("last line:%s", line.c_str());

	s >> hueLow;
	s >> hueHigh;
	s >> satLow;
	s >> satHigh;
	s >> valLow;
	s >> valHigh;

	printf("values:%d,%d,%d,%d,%d,%d\n", hueLow, hueHigh, satLow, satHigh,valLow, valHigh);

}

int main(int argc, char** argv)
{

	VideoCapture cap;

	init(argc, argv, cap);

	loadLastLine("thresholds.txt");

	setupGUI();

	Mat tmp_frame, out_frame;

	for (;;)
	{
		cap >> tmp_frame;
		if (!tmp_frame.data)
			break;
		detectColor(tmp_frame, out_frame);
		findMassCenter(out_frame, out_frame);
		imshow(winIn, tmp_frame);
		imshow(winOut, out_frame);
		char keycode = waitKey(30);
		if (keycode == 's') {
			save("thresholds.txt");
		}
		if (keycode == 27)
			break;

	}

	save("thresholds.txt");

	cap.release();
	return 0;
}
