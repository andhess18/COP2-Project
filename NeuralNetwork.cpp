#include "NeuralNetwork.h"
#include "AbstractNeuralNetwork.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <opencv2/opencv.hpp>
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

NeuralNetwork::NeuralNetwork(int inputNodes, const vector<int>& hiddenNodes, int outputNodes)
    : AbstractNeuralNetwork(inputNodes, hiddenNodes, outputNodes) {
    // Creates the input layer
    layers.push_back(std::make_shared<InputLayer>(inputNodes, hiddenNodes[0]));

    // Creates hidden layers
    for (size_t i = 1; i < hiddenNodes.size(); i++) {
        layers.push_back(std::make_shared<HiddenLayer>(hiddenNodes[i - 1], hiddenNodes[i]));
    }

    // Creates the output layer
    layers.push_back(std::make_shared<OutputLayer>(hiddenNodes.back(), outputNodes));

    numLayers = layers.size();
}


// Forward propagation: Calculate the output of each layer in the neural network
void NeuralNetwork::forward(const Mat& inputImage, vector<Mat>& layerOutputs) {
    layerOutputs.resize(layers.size());
    Mat currOutput = inputImage.clone();
    inputImage.convertTo(currOutput, CV_64F);

    if (!currOutput.isContinuous()) {
        currOutput = currOutput.clone();
    }

    currOutput = currOutput.reshape(1, 1);

    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->forward(currOutput);
        currOutput = layers[i]->getOutput();
        layerOutputs[i] = currOutput;
    }
}


// Backpropagation: Compute the gradients of the weights and biases with respect to the loss function
void NeuralNetwork::backPropagation(const vector<Mat>& layerOutputs, const Mat& inputImage, uint8_t label, vector<Mat>& weightGradients, vector<Mat>& biasGradients) {
    weightGradients.resize(numLayers);
    biasGradients.resize(numLayers);

    Mat outputLayerError = Mat::zeros(1, outputNodes, CV_64F);
    for (int i = 0; i < outputNodes; ++i) {
        double outputValue = layerOutputs.back().at<double>(0, i);
        double targetValue = (i == label) ? 1.0 : 0.0;
        double error = outputValue - targetValue;
        outputLayerError.at<double>(0, i) = error;
    }

    // Computes gradients for the output layer
    Mat outputWeightGradient = layerOutputs[layerOutputs.size() - 2].t() * outputLayerError;
    weightGradients.back() = outputWeightGradient;
    biasGradients.back() = outputLayerError.clone();

    // Computes gradients for the hidden layers
    Mat prevLayerError = outputLayerError;
    for (int i = numLayers - 2; i >= 0; --i) {
        Mat currLayerError = prevLayerError * layers[i + 1]->getWeights().t();
        Mat derivative = layers[i]->getActivationDerivative(layerOutputs[i]);
        
        currLayerError = currLayerError.mul(derivative);
        
        // Compute input layer
        Mat inputLayer = (i == 0) ? inputImage : layerOutputs[i - 1];
        Mat currWeightGradient = inputLayer.t() * currLayerError;

        weightGradients[i] = currWeightGradient;
        biasGradients[i] = currLayerError.clone();

        prevLayerError = currLayerError;
    }

    // Update the weight and bias gradients for each layer
    for (int i = 0; i < numLayers; ++i) {
        layers[i]->setWeightGradients(weightGradients[i]);
        layers[i]->setBiasGradients(biasGradients[i]);
    }

}


//Trains the neural network using gradient descent with backpropagation
void NeuralNetwork::train(const vector<Mat>& inputImages, const vector<uint8_t>& labels, int epochs, double learningRate) {
    int numSamples = inputImages.size();
    vector<Mat> layerOutputs(numLayers);
    vector<Mat> weightGradients(numLayers);
    vector<Mat> biasGradients(numLayers);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double totalLoss = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            // Forward propagation
            Mat currOutput = inputImages[i].clone();
            inputImages[i].convertTo(currOutput, CV_64F);

            if (!currOutput.isContinuous()) {
                currOutput = currOutput.clone();
            }

            currOutput = currOutput.reshape(1, 1);

            for (int j = 0; j < numLayers; ++j) {
                layers[j] -> forward(currOutput);
                currOutput = layers[j] -> getOutput();
                layerOutputs[j] = currOutput.clone();
            }

            // Compute loss
            double loss = crossEntropyLoss(layerOutputs.back(), labels[i]);
            totalLoss += loss;

            // Backpropagation
            backPropagation(layerOutputs, inputImages[i], labels[i], weightGradients, biasGradients);
            // Gradient descent update
            for (auto& layer : layers) {
                layer -> gradientDescent(learningRate);
            }
        }

        std::cout << "Epoch: " << epoch << " Loss: " << totalLoss / numSamples << std::endl;
    }
}


// Predict the class of the input image using the trained neural network
int NeuralNetwork::predict(const Mat& inputImage) {
    Mat currOutput = inputImage.clone();
    inputImage.convertTo(currOutput, CV_64F);

    if (!currOutput.isContinuous()) {
        currOutput = currOutput.clone();
    }

    currOutput = currOutput.reshape(1, 1);

    for (auto& layer : layers) {
        layer -> forward(currOutput);
        currOutput = layer -> getOutput();
    }

    // Find the index of the maximum value in the output layer
    double maxValue = currOutput.at<double>(0, 0);
    int maxIndex = 0;
    for (int i = 1; i < outputNodes; ++i) {
        double value = currOutput.at<double>(0, i);
        if (value > maxValue) {
            maxValue = value;
            maxIndex = i;
        }
    }

    return maxIndex;
}

// Calculate the cross-entropy loss between the predicted output and the true label
double NeuralNetwork::crossEntropyLoss(const Mat& output, uint8_t label) {
    const double epsilon = 1e-8; //to prevent log(0) in the loss calculation
    return -std::log(output.at<double>(0, label) + epsilon);
}

