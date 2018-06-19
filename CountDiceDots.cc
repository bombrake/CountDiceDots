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
#include <math.h>
//Opencv includes
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//from https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp
static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

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
       //cout<<"Read image : "<<i+1<<endl;
   }
   cout<<"All images read!!"<<endl;
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
    //Display thresholded images
    //cv::imshow("thresholded",gray);
    //cv::waitKey(0);
    vector<Vec3f> dots; //to store the information on dots
    vector<vector<Point>> contours; //to store the information about dice
    vector<Point> approx; //approx points
    vector<vector<Point>> approx_contours;
    //vector<Vec4i> hierarchy;
    //Detect the dots!!
    cv::HoughCircles(gray,dots,CV_HOUGH_GRADIENT,2,15,60.0,30.0,5,17);
    //Detect the contours
    cv::findContours(gray, contours,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
    //cout<<"Number of Contours detected: "<<contours.size()<<endl;
    //we don't want to draw the contour that bounds the total image, so we remove it from the list with a check of contour areas
    unsigned int area_of_image=output.rows*output.cols;
    unsigned int area_of_contour;
    int background_index=-1;
   //Detect Rectangles using approxpolyDP
    for (int i = 0; i < contours.size(); i++)
    {
        //check area
        area_of_contour = cv::contourArea(contours[i]);
        if(area_of_contour<0.90*area_of_image)
        {
            approxPolyDP(Mat(contours[i]), approx,arcLength(Mat(contours[i]), true) * 0.02, true);
            //check if it has 4 vertices,convex and has a reasonable area
            if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 800 && isContourConvex(Mat(approx)))
            {
                double maxCosine = 0;

                //loop for vertices 2,3,4
                for (int j = 2; j < 5; j++)
                {
                    double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                    maxCosine = MAX(maxCosine, cosine);
                }

                if (maxCosine < 0.3)
                    approx_contours.push_back(approx);
            }
        }
        else
        {
            background_index=i;
        }
        
    }

    cout<<"Number of detected Dice: "<<approx_contours.size()<<endl;
    //Now we have the detected dots and the dice!!
    //Proceeding to display functions

    for(unsigned int k=0;k<contours.size();k++)
    {
        if(k!=background_index)
        {//cv::drawContours(output,approx_contours,k,cv::Scalar(0,255,0),3,8); // display dice as approx rectangles!!
        cv::drawContours(output, contours, k, cv::Scalar(0, 255, 0), 3, 8); // display dice as contours themselves!!
        }
    } 
    
    cout << "Number of dots: " << dots.size() << endl;
    
    //display detected dots
    std::vector<cv::Vec3f>::const_iterator iterator_dots = dots.begin();
    while (iterator_dots != dots.end())
    {

        circle(output,Point((*iterator_dots)[0], (*iterator_dots)[1]),(*iterator_dots)[2],Scalar(0, 255, 0),2);                   
        ++iterator_dots;
    }
    
    
    //TODO: display number of dots near the dice
    /*
    Each contour is stored as a series of points. To get the number of dots within a dice face,
     we need to find the number of dots that are within the contour boundaries.
     Also, each dot has info as (x,y,radius)
     Thus, simple application of pointPolygonTest for the dots should do the trick
    */
    vector<int> dots_in_die(approx_contours.size());
    vector<vector<Point>> dots_coord_in_die(approx_contours.size());
    float distance_to_rectangle;
    vector<float> x_max(approx_contours.size()), y_max(approx_contours.size());

    for (int k = 0; k < approx_contours.size(); k++)
    {
       // cout<<"Dice Index "<<k<<endl;
        iterator_dots = dots.begin();
        while(iterator_dots != dots.end())
        {
            //cout<<"Loop"<<endl;
            cv::Vec3f dot_val=*iterator_dots;
            
            distance_to_rectangle=pointPolygonTest(approx_contours[k], Point2f(dot_val[0],dot_val[1]), true);
            //cout<<distance_to_rectangle<<endl;
            if(fabs(distance_to_rectangle)<85)//inside
            {
                dots_in_die[k]++;
                dots_coord_in_die[k].push_back(Point(dot_val[0],dot_val[1]));
                if (dot_val[0] > x_max[k])
                {
                    x_max[k] = dot_val[0];
                }
                if (dot_val[1] > y_max[k])
                {
                    y_max[k] = dot_val[1];
                }
            }
            ++iterator_dots;
        }
        //cout<<x_max[k]<<" "<<y_max[k]<<endl;
    }

    //Add the text at top left of the image
    std::string label = format("Sum: %d", dots.size());
    putText(output, label, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0),2);
    for(int k=0;k<dots_in_die.size();k++)
    {
        putText(output, format("%d", dots_in_die[k]), Point(x_max[k]+70,y_max[k]-70 ), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0),2);//displaying 80 pixels to right and above the max x and y dot.
    }
    //end text overlay

    //now display the image
    cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
    cv::imshow("image", output);
    cv::waitKey(500);
    //write it to disk
    string output_path="../output images/output_"+to_string(i)+".png";
    cv::imwrite(output_path,output);
//} //end else
}
    return 0;
}