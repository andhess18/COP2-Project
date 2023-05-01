#ifndef INEURALNETWORK_H
#define INEURALNETWORK_H

#include <vector>
#include <opencv2/opencv.hpp>

using std::vector;
using cv::Mat;

class INeuralNetwork {
public:
	virtual void train(const vector<Mat>& inputImages, const vector<uint8_t>& labels, int epochs, double learningRate) = 0;
	virtual ~INeuralNetwork() = default;
};

#endif
