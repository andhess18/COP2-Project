#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <winsock2.h>
#include "TrainingData.h"

using cv::Mat;
using std::string;
using std::vector;
using std::ifstream;


TrainingData::TrainingData(const string& imagesPath, const string& labelsPath) {

	images = readImages(imagesPath);
	labels = readLabels(labelsPath);
	processedImages = preprocessData(images);
}
	

//const is used because its not supposed to change member variables in any way
vector<Mat> TrainingData::getImages() const {
		
	return images;
}

vector<uint8_t> TrainingData::getLabels() const {
		
	return labels;
}
	vector<Mat> TrainingData::getprocessedImages() const {
		
	return processedImages; 
}

/*private:
	vector<Mat> images;
	vector<uint8_t> labels;
	vector<Mat> processedImages;*/

	// Reads a 4-byte (32-bit) integer from the MNIST file and converts it from big-endian to the host os byte order.
uint32_t TrainingData::readFileData(ifstream& file) {
	uint32_t value;
	file.read(reinterpret_cast<char*>(&value), sizeof(value));//converts 32bit-int pointer to char pointer so read can read the file data (read() expects char)(data is unsigned 32-bit int and pointer is moved after each call)
	return ntohl(value);
}

	
	
vector<Mat> TrainingData::readImages(const string& imagesPath) {
	// opens "imagespath" file in binary mode(file contains binary data not text)
	ifstream file(imagesPath, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Could not open file: " << imagesPath << std::endl;
		exit(EXIT_FAILURE);
	}


	uint32_t magicNumber = readFileData(file);  //first int (file type and version)
	uint32_t numImages = readFileData(file);    //second int (number of images)
	uint32_t numRows = readFileData(file);	  //dimensions of images in dataset
	uint32_t numCols = readFileData(file);

	vector<Mat> trainingImages;
	trainingImages.reserve(numImages); //reserves the correct amount of memory for thevector(improves preformance)

	for (uint32_t i = 0; i < numImages; i++) { //loop to add all images to vector
		Mat image(numRows, numCols, CV_8UC1);
		file.read(reinterpret_cast<char*> (image.data), numRows * numCols);
		trainingImages.push_back(image);
	}

	return trainingImages;
}



vector<uint8_t> TrainingData::readLabels(const string& labelsPath) {
	ifstream file(labelsPath, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Could not open file: " << labelsPath << std::endl;
		exit(EXIT_FAILURE);
}


	uint32_t magicNumber = readFileData(file);
	uint32_t numLabels = readFileData(file);

	vector<uint8_t> labels(numLabels); //only needs to be 8 bits
	file.read(reinterpret_cast<char*>(labels.data()), numLabels); //labels.data()returns a pointer to first element in vector. Because labels is avector<uint8_t>, each byte read from the file is assigned to an element in thevector in order.

	return labels;

}

// Preprocesses the input images by converting their data type and normalizing pixel values.
vector<Mat> TrainingData::preprocessData(const vector<Mat>& inputImages) { 
	vector<Mat> processedImages;
	processedImages.reserve(inputImages.size());

	for (const auto& image : inputImages) {
		Mat processedImage;
		image.convertTo(processedImage, CV_64F, 1.0 / 255.0);
		processedImage = processedImage.reshape(1, 1);
		processedImages.push_back(processedImage);
	}

	return processedImages;
}

