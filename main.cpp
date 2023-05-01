#include "NeuralNetwork.h"
#include "TrainingData.h"
#include "userDrawing.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using std::vector;
using cv::Mat;

int main() {
    // Define the structure of your neural network
    int inputNodes = 784; 
    vector<int> hiddenNodes = { 128 }; 
    int outputNodes = 10; 
	
    NeuralNetwork nn(inputNodes, hiddenNodes, outputNodes);
	
	//Load training data
	const string imagesPath = "C:\\Users\\andhe\\source\\repos\\COP Project\\COP Project\\trainingdata\\train-images.idx3-ubyte";
	const string labelsPath = "C:\\Users\\andhe\\source\\repos\\COP Project\\COP Project\\trainingdata\\train-labels.idx1-ubyte";

	TrainingData trainData(imagesPath, labelsPath);
	vector<Mat> trainImages = trainData.getprocessedImages();

	vector<uint8_t> trainLabels = trainData.getLabels();

	//Load test data
	const string testImagesPath = "C:\\Users\\andhe\\source\\repos\\COP Project\\COP Project\\trainingdata\\t10k-images.idx3-ubyte";
	const string testLabelsPath = "C:\\Users\\andhe\\source\\repos\\COP Project\\COP Project\\trainingdata\\t10k-labels.idx1-ubyte";

	TrainingData testData(testImagesPath, testLabelsPath);
	vector<Mat> testImages = testData.getprocessedImages();

	vector<uint8_t> testLabels = testData.getLabels();

	int epochs = 10;
	double learningRate = 0.001;

	nn.train(trainImages, trainLabels, epochs, learningRate);
	int correctPredictions = 0;

	for (size_t i = 0; i < testImages.size(); ++i) {
		int prediction = nn.predict(testImages[i]);
		if (prediction == testLabels[i]) {
			correctPredictions++;
		}
	}

	double accuracy = static_cast<double>(correctPredictions) / testImages.size();
	std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl;

	getuserImage(nn);
}