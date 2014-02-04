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

int hueLow = 160, hueHigh = 179, satLow = 30, satHigh = 200, valLow = 30,
		valHigh = 200;

int thresh = 100;

string winIn = "video";
string winOut = "output";
string winSetup = "setup";

bool isEdgeDetect = true;

static void help() {
	printf("\n"
			"Color detection. Default is red. Use trackbars to set.\n"
			"Hue values of basic colors\n"
			"Orange  0-22\n"
			"Yellow 22- 38\n"
			"Green 38-75\n"
			"Blue 75-130\n"
			"Violet 130-160\n"
			"Red 160-179\n");
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

static void crossHair(Mat& img, Scalar color = Scalar(0, 255, 0)) {
	line(img, Point(0, 0), Point(img.cols - 1, img.rows - 1), color);
	line(img, Point(0, img.rows - 1), Point(img.cols - 1, 0), color);
}

static void setupGUI() {
	namedWindow(winIn, 1);
	namedWindow(winOut, 1);
	namedWindow(winSetup, 1);

	/*
	 createTrackbar("hue low", winSetup, &hueLow, 180);
	 createTrackbar("hue high", winSetup, &hueHigh, 180);
	 createTrackbar("saturation low", winSetup, &satLow, 255);
	 createTrackbar("saturation high", winSetup, &satHigh, 255);
	 createTrackbar("value low", winSetup, &valLow, 255);
	 createTrackbar("value high", winSetup, &valHigh, 255);
	 */

}

static void init(int argc, char** argv, VideoCapture& cap) {
	help();

	if (argc < 2) {
		cap.open(1);
	} else {
		cap.open(std::string(argv[1]));
	}
	if (!cap.isOpened()) {
		printf("\nCan not open camera or video file\n");
		exit(-1);
	}

	Mat tmp_frame;

	cap >> tmp_frame;
	if (!tmp_frame.data) {
		printf("can not read data from the video source\n");
		exit(-1);
	}

	printf("image height:%d\n", tmp_frame.rows);

}

static void binarizeImage(const Mat img, Mat& outImg) {
	cvtColor(img, outImg, CV_BGR2GRAY);
	adaptiveThreshold(outImg, outImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, //
			cv::THRESH_BINARY_INV, //
			7, //
			7 //
			);
	GaussianBlur(outImg, outImg, Size(3, 3), 0, 0);

}

static void fineImage(const Mat img, Mat& outImg) {
	vector<Mat> channels;

	split(img, channels);

	for (uint i = 0; i < channels.size(); i++) {
		equalizeHist(channels[i], channels[i]);
	}

	merge(channels, outImg);

	GaussianBlur(outImg, outImg, Size(3, 3), 0, 0);
}

static void detectColor(const Mat img, Mat& outImg) {
	Scalar lowThresh(hueLow, satLow, valLow);
	Scalar highThresh(hueHigh, satHigh, valHigh);

	cvtColor(img, outImg, CV_BGR2HSV);
	inRange(outImg, lowThresh, highThresh, outImg);
	//erode(outImg, outImg, Mat());
	dilate(outImg, outImg, Mat(), Point(-1, -1), 1);
	GaussianBlur(outImg, outImg, Size(3, 3), 0, 0);

}

static void findMassCenter(const Mat src_gray, Mat& canny_output) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Get the biggest area
	double maxArea = 0;
	int maxIndex = -1;
	for (uint i = 0; i < contours.size(); i++) {
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
	Point2f mc(mu.m10 / mu.m00, mu.m01 / mu.m00);

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

	printf("values:%d,%d,%d,%d,%d,%d\n", hueLow, hueHigh, satLow, satHigh,
			valLow, valHigh);

}

static bool findMaxContour(const Mat binaryImg, vector<Point> contour) {
	bool found = false;

	Mat tmp;
	binaryImg.copyTo(tmp);
	printf("fm1\n");
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(tmp, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_SIMPLE);
	if (!contours.empty()) {
		found = true;
		uint n = 0;
		uint idx = 0;
		for (uint i = 0; i < contours.size(); i++) {
			if (n < contours[i].size()) {
				n = contours.size();
				idx = i;
			}
		}
		contour = contours[idx];
		printf("found %d points in external contour\n", contour.size());
	}

	Mat out;
	cvtColor(binaryImg, out, CV_GRAY2BGR);

	drawContours(out, contours, -1, Scalar(255, 0, 0), 2);
	imshow(winSetup, out);
	/*
	 uint maxIdx = 0;
	 double maxArea = 0;

	 printf("fm2\n");
	 if (!contours.empty() && !hierarchy.empty()) {
	 // iterate through all the top-level contours,
	 // draw each connected component with its own random color
	 int idx = 0;
	 for (; idx >= 0; idx = hierarchy[idx][0]) {
	 printf("fm3\n");
	 double area = contourArea(contours[idx]);
	 if (area > maxArea) {
	 maxArea = area;
	 maxIdx = idx;
	 }
	 }
	 if (maxArea > 0) {
	 printf("fm4\n");
	 contour = contours[maxIdx];
	 found = true;
	 }
	 }
	 */

	return found;
}

static bool detectBiggestEllipse(Mat const binaryImg, RotatedRect& rrect) {
	vector<Point> contour;
	bool found = findMaxContour(binaryImg, contour);
	printf("dbe1\n");
	if (found) {
		if (contour.size() < 5) {
			printf("contour.size=%d\n", (int) contour.size());
			return false;
		}
		Mat pointsf;
		Mat(contour).convertTo(pointsf, CV_32F);
		printf("dbe2\n");
		rrect = fitEllipse(pointsf);
		printf("dbe3\n");

	}

	return found;
}

static bool detectHoughCircle(Mat const binaryImg, Vec3f &circle) {
	bool found = false;
	vector<cv::Vec3f> circles;
	printf("detectHoughCircle 1\n");
	HoughCircles(binaryImg, circles, CV_HOUGH_GRADIENT, 2, // accumulator resolution (size of the image / 2)
			50, // minimum distance between two circles
			200, // Canny high threshold
			100, // minimum number of votes
			20, 500); // min and max radius
	printf("detectHoughCircle 2\n");
	float maxr = 0;
	uint idx = 0;
	for (uint i = 0; i < circles.size(); i++) {
		if (circles[i][2] > maxr) {
			maxr = circles[i][2];
			idx = i;
		}
	}

	if (!circles.empty()) {
		printf("detectHoughCircle 3,%f,%d\n", maxr, idx);
		circle = circles[idx];
		printf("detectHoughCircle 4\n");
		found = true;
	}

	return found;
}

static bool detectBiggestCircle(Mat const binaryImg, Point2f& center,
		float& radius) {

	vector<Point> contour;
	bool found = findMaxContour(binaryImg, contour);

	if (found) {
		minEnclosingCircle(cv::Mat(contour), center, radius);
	}

	return found;
}

static void drawRotatedRect(Mat img, RotatedRect rrect,
		Scalar color = Scalar(0, 0, 255)) {
	Point2f vtx[4];
	rrect.points(vtx);

	for (int j = 0; j < 4; j++) {
		line(img, vtx[j], vtx[(j + 1) % 4], color, 1, CV_AA);
	}

}

int main(int argc, char** argv) {

	VideoCapture cap;

	init(argc, argv, cap);

	loadLastLine("thresholds.txt");

	setupGUI();

	Mat tmp_frame, out_frame;
	RotatedRect rrect;

	for (;;) {
		cap >> tmp_frame;
		if (!tmp_frame.data)
			break;
		//fineImage(tmp_frame, out_frame);
		binarizeImage(tmp_frame, out_frame);

		printf("1\n");
		float radius;
		cv::Point2f center;
		bool found = false;
		//bool found = detectBiggestCircle(out_frame, center, radius);
		Vec3f circle_;
		found = detectHoughCircle(out_frame, circle_);
		//found = detectBiggestEllipse(out_frame, rrect);
		printf("2\n");
		//detectColor(tmp_frame, out_frame);
		//if (isEdgeDetect) {
		//	findMassCenter(out_frame, out_frame);
		//}
		if (found) {
			//for (uint i = 0; i < circles.size(); i++) {
			Vec3f c = circle_;
			circle(tmp_frame, Point(c[0], c[1]), static_cast<int>(c[2]),
					cv::Scalar(0, 255, 0), 4);
			//drawRotatedRect(tmp_frame, rrect);
			//}
			printf("3\n");
		}

		imshow(winIn, tmp_frame);
		imshow(winOut, out_frame);
		char keycode = waitKey(30);
		if (keycode == 's') {
			save("thresholds.txt");
		}
		if (keycode == 27)
			break;
		if (keycode == 'e') {
			isEdgeDetect = !isEdgeDetect;
		}

	}

	save("thresholds.txt");

	cap.release();
	return 0;
}
