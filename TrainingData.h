#ifndef TRAININGDATA_H
#define TRAININGDATA_H


#include <fstream>
#include <vector>
#include <string>
#include<opencv2/opencv.hpp>

using cv::Mat;
using std::string;
using std::vector;
using std::ifstream;



class TrainingData {
public:
	TrainingData(const string& imagesPath, const string& labelsPath);

	vector<Mat> getImages() const;

	vector<uint8_t> getLabels() const;

	vector<Mat> getprocessedImages() const;

private:
	vector<Mat> images;
	vector<uint8_t> labels;
	vector<Mat> processedImages;

	uint32_t readFileData(ifstream& file);

	vector<Mat> readImages(const string& imagesPath);

	vector<uint8_t> readLabels(const string& labelsPath);

	vector<Mat> preprocessData(const vector<Mat>& inputImages);

};


#endif
