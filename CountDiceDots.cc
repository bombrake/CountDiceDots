/*
C++ OpenCV program to count the dice dots for each die, and also the total number of dice dots
in the image.
Author: Hariharan Ramshankar
*/
//includes 
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
//Opencv includes
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    //Assuming that the images are stored in the input images folder in png format.
    vector<String> file_names; //vector to store the file names for the images
    string path="../input images/*.png";
    glob(path,file_names);
    //now read the images
    cout<<"Number of images to process:  "<<file_names.size()<<endl; //print the number of images in the folder
    vector<cv::Mat> images;
    //checking if they are read properly by displaying
    /*
    //start check
    cv::Mat image;
    for(int i=0;i<6;i++)
    {
        image=cv::imread(file_names[i]);
        cv::imshow("image",image);
        waitKey(1000);
    }
    //end check
    */
   //reading the images into a vector for easier processing
   for(int i=0;i<6;i++)
   {
       images.push_back(imread(file_names[i]));
       cout<<"Read image : "<<i+1<<endl;
   }
   cout<<"All images read!!";
    //now detect the dots 
    /*
    Approach- the dots are circles, so use hough circles function.
    cv2.HoughCircles(image, method, dp, minDist)
    image- single channel
    method- Hough_gradient
    minDist- key parameter, distance in pixels between center of detected dots
    */
   for(int i=0;i<6;i++)
   {
    //STEP1-convert image to grayscale
    cv::Mat output=images[i].clone();
    cv::Mat gray; //to hold grayscale image
    cv::cvtColor(output, gray, CV_BGR2GRAY);
    //apply gaussian blur
    cv::GaussianBlur(gray, gray, Size(3, 3), 2, 2);
    //apply threshold to reduce the effect of backgroud
    //cv::adaptiveThreshold(gray, gray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 75, 30);
    cv::threshold(gray,gray,180,255.0,CV_THRESH_BINARY_INV);
    //cv::imshow("thresholded",gray);
    //cv::waitKey(0);
    vector<Vec3f> dots; //to store the information on dots
    cv::HoughCircles(gray,dots,CV_HOUGH_GRADIENT,2,15,60.0,30.0,5,17);
    
    /*
    if(dots.size()==0)
    {
        cout<<"No dots found!! :("<<endl;
        //exit(0);
    }
    */
    //else
    //{

        cout << "Number of dots: " << dots.size() << endl;
        //display detected circles
        std::vector<cv::Vec3f>::const_iterator iterator_dots = dots.begin();
        while (iterator_dots != dots.end())
        {

            cv::circle(output,
                       cv::Point((*iterator_dots)[0], (*iterator_dots)[1]), // circle centre
                       (*iterator_dots)[2],                       // circle radius
                       cv::Scalar(0, 255, 0),           // color
                       2);                              // thickness

            ++iterator_dots;
        }
        //Add the text at top left of the image
        std::string label = format("Sum: %d", dots.size());
        putText(output, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        //end text overlay
        //now display the image
        cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
        cv::imshow("image", output);
        cv::waitKey(0);
        //write it to disk
        string output_path="../output images/output_"+to_string(i)+".png";
        cv::imwrite(output_path,output);
    //} //end else
   }
    return 0;
}