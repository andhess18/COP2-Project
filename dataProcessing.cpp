#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <winsock2.h>

using cv::Mat;
using std::string;
using std::vector;
using std::ifstream;


class TrainingData {
public:
	TrainingData(const string &imagesPath, const string &labelsPath) {
		images = readImages(imagesPath);
		labels = readLabels(labelsPath);
		processedImages = preprocessData(images);
	}

	//const is used because its not supposed to change member variables in any way
	vector<Mat> getImages() const {
		return images;
	}

	vector<uint8_t> getLabels() const {
		return labels;
	}
	vector<Mat> getprocessedImages() const {
		return processedImages; 
	}

private:
	vector<Mat> images;
	vector<uint8_t> labels;
	vector<Mat> processedImages;

	// Reads a 4-byte (32-bit) integer from the MNIST file and converts it from big-endian to the host byte order.
	uint32_t readFileData(ifstream &file) {//& passes a refrence instead of entire data structure(faster)
		uint32_t value;
		file.read(reinterpret_cast<char*>(&value), sizeof(value));//converts 32bit-int pointer to char pointer so read can read the file data (read() expects char)(data is unsigned 32-bit int and pointer is moved after each call)
		return ntohl(value);
	}

	
	
	vector<Mat> readImages(const string &imagesPath) {
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
		trainingImages.reserve(numImages); //reserves the correct amount of memory for the vector(improves preformance)

		for (uint32_t i = 0; i < numImages; i++) { //loop to add all images to vector
			Mat image(numRows, numCols, CV_8UC1);
			file.read(reinterpret_cast<char*> (image.data), numRows * numCols);
			trainingImages.push_back(image);
		}

		return trainingImages;
	}



	vector<uint8_t> readLabels(const string &labelsPath) {
		ifstream file(labelsPath, std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Could not open file: " << labelsPath << std::endl;
			exit(EXIT_FAILURE);
		}


		uint32_t magicNumber = readFileData(file);
		uint32_t numLabels = readFileData(file);

		vector<uint8_t> labels(numLabels); //only needs to be 8 bits
		file.read(reinterpret_cast<char*>(labels.data()), numLabels); //labels.data() returns a pointer to first element in vector. Because labels is a vector<uint8_t>, each byte read from the file is assigned to an element in the vector in order.

		return labels;

	}
	
	
	// Preprocesses the input images by converting their data type and normalizing pixel values.
	vector<Mat> preprocessData(const vector<Mat> &inputImages) { //refrence to 'images' vector
		vector<Mat> processedImages;
		processedImages.reserve(inputImages.size());

		for (const auto& image : inputImages) {
			Mat processedImage;
			image.convertTo(processedImage, CV_32F, 1.0 / 255.0);
			processedImages.push_back(processedImage);
		}

		return processedImages;
	}


};



void dataTest() {
	const string imagesPath = "C:\\Users\\andhe\\source\\repos\\COP Project\\COP Project\\trainingdata\\train-images.idx3-ubyte";
	const string labelsPath = "C:\\Users\\andhe\\source\\repos\\COP Project\\COP Project\\trainingdata\\train-labels.idx1-ubyte";

	TrainingData data(imagesPath, labelsPath);
	vector<Mat> images = data.getImages();
	vector<uint8_t> labels = data.getLabels();
	vector<Mat> processedImages = data.getprocessedImages();


	// Check the number of images and labels
	std::cout << "Number of images: " << images.size() << std::endl;
	std::cout << "Number of labels: " << labels.size() << std::endl;

	// Display the first image and its label
	cv::imshow("First Image (Original)", images[1]);
	cv::imshow("First Image (Processed)", processedImages[1]);
	std::cout << "First image label: " << static_cast<int>(labels[1]) << std::endl;

	// Wait for a key press and close the image windows
	cv::waitKey(0);
	cv::destroyAllWindows();
}

int main() {
	dataTest();
	return 0;
}