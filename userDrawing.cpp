//Andrew Hess
//COP 3003
//start of handwriting recognition program. simply takes in an image, converts it to a vector, 
//and preprocesses the image to the proper size as well as removes artificats and unnessarry noise using the opencv2 library

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "NeuralNetwork.h"
#include "userDrawing.h"


using std::vector;
using cv::Mat;

Mat image;

Mat currentUserImage;

//function for drawing number
void drawAt(int x, int y) {
    //draws red circles and displays it in window
    cv::circle(image, cv::Point(x, y), 10, cv::Scalar(0, 0, 255), -1);
    cv::imshow("Window", image);
}

//checks for mouse input and stores completed image in userImages
void handleMouseEvents(int event, int x, int y, int flags, void* userdata) {
    static bool isDrawing = false;
    NeuralNetwork* neuralNet = reinterpret_cast<NeuralNetwork*>(userdata);

    if (event == cv::EVENT_LBUTTONDOWN) {//if left mousebutton is pressed
        isDrawing = true;
        image = Mat::zeros(500, 500, CV_8UC3); 
    }

    else if (event == cv::EVENT_MOUSEMOVE && isDrawing)
        drawAt(x, y);

    //handles processing and prediction of user number once user releases the left mouse button
    else if (event == cv::EVENT_LBUTTONUP && isDrawing) {
        isDrawing = false;
        currentUserImage = processUserImage(image);
        int predictedNumber = neuralNet->predict(currentUserImage);
        std::cout << "The predicted number is: " << predictedNumber << std::endl;
    }

}


void getuserImage(NeuralNetwork&  neuralNet) {
    image = Mat::zeros(500, 500, CV_8UC3);
    cv::namedWindow("Window");
    cv::setMouseCallback("Window",handleMouseEvents, reinterpret_cast<void*>(&neuralNet));
    cv::imshow("Window", image);
    cv::waitKey(0);
}


Mat processUserImage(Mat &curentuserImage) {
    //empty grayscale image(28 by 28 pixels)
    Mat processedUserImage(28, 28, CV_8UC1);

    //converts image to grayscale by exttracting the red channel
    Mat grayscaleImage;
    vector<Mat> channels(3);//stores seperate color chanels of user image
    cv::split(curentuserImage, channels);
    grayscaleImage = channels[2]; //red channel
    
    
    //resizes the grayscale image to 28x28 pixels
    cv::resize(grayscaleImage, processedUserImage, processedUserImage.size(), 0, 0, cv::INTER_AREA);

    //Normalizes the pixel values to a range between 0 and 1
    processedUserImage.convertTo(processedUserImage, CV_64F, 1.0 / 255.0);
    processedUserImage = processedUserImage.reshape(1, 1);

    return processedUserImage;
}

