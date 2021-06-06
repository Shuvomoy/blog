---
layout: post 
title: C++ Kernel for Jupyter Notebook
categories: [programming] 
comments: true 
---

In this blog, we will talk about how to have a C++ kernel in Jupyter notebook, which would allow us to run C++ code in a dynamic manner. We proceed as follows. First, we install Jupyter notebook, and then install C++ kernel for Jupyter notebook. I ran my code on 64 bit Ubuntu 14.04.

<!-- more -->

### Installing Jupyter notebook

The easiest way to install Jupyter notebook is to install Anaconda. Just go to https://www.continuum.io/downloads and then download the Python 3.5 version of Anaconda, and follow the installation instruction. Simple!

### Installing C++ kernel for Jupyter notebook: Cling

Cling is an interactive C++ interpreter; it allows us to type and execute C++ code dynamically, like Python or Julia. The developers provide binary snapshots for Cling at https://root.cern.ch/download/cling/. Download and extract the one associated with your platform and operating system.  Let us pretend that the path name for the extracted folder is `/home/ubuntu_user/cling_ubuntu`. There will be a folder named `bin` in it, which would contain the Cling binary. We need to add the path, i.e., `/home/ubuntu_user/cling_ubuntu/bin` in our path. So, open up a terminal and type the following.

``export PATH=/home/ubuntu_user/cling_ubuntu/bin:$PATH``

Now we need to install Cling kernel for Jupyter. (alternatively add the line to .bashrc)

First, in terminal we change directory to it and then install the kernel as follows.

`cd /cling-install-prefix/share/cling/Jupyter/kernel`

`pip install -e .`

Finally register for the kernelspec:

`jupyter-kernelspec install --user cling-cpp11`.

Though I have used C++ version 11, one could alternatively use C++ 14 or C++ 17.

If you have not done it already, install a C++ compiler such as g++ from terminal or software center. 

Okay, we are done. To start a jupyter notebook with C++ kernel, just type the following in the terminal:

`jupyter notebook`

A webpage named `Home` will open. On the top right corner, we will have a button titled `new`, click on it and select C++. 


### First example

Our first example is to write a simple addition function, which additionally will print "Hello World!" (of course). First comes the obvious stuff.


```c++
#include <iostream>
```




​    




```c++
using namespace std;
```




​    



To write a function, we need to execute a special Cling command name `.rawInput`, before and after the function definition.


```c++
.rawInput
```

`Out:    Using raw input`





​    




```c++
int addition (int a, int b)
{
  int r;
  r=a+b;
  cout << "Hello World!" << '\n'; 
  return r;
}
```




​    




```c++
.rawInput
```

`Out:     Not using raw input`





​    



Let us check if the function is working.


```c++
addition(5,6)
```

`Out:     Hello World!`
   `(int) 11`


It is working!

Now we are going to look at a slightly complicated example, where we are going to do some computer vision stuff using OpenCV.

### Installing OpenCV

Install OpenCV from http://opencv.org/ by following the instructions available at this [link](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation).

We will start with loading the necessary header files.


```c++
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/features2d/features2d.hpp>
```




​    



We need to provide the location of the associated libraries.


```c++
.L /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4
```




​    




```c++
.L /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4
```




​    




```c++
.L /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4
```




​    




```c++
.L /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4
```




​    



The rest of the code is as usual, nothing fancy going on there. The code is heavily commented, so hopefully there would not be any issue understanding it.


```c++
using namespace cv;
```




​    




```c++
VideoCapture input("GOPR0030.MP4"); // Create a video capture object that is going 
// to take the video input. GOPR0030.MP4 is the video file that I am considering for my
// code, any other video file could also be used. 
```




`Out:     (cv::VideoCapture &) @0x7ff133d56018`





```c++
Mat frameVideo, frameVideo_prev; // The matrix object associated with the frames of the video

// Mat frameVideo_prev;
```




`Out:     (cv::Mat &) @0x7ff133d56090`





```c++
vector<Point2f> points, points_prev; 
// The optical flow function expects vectors of points as features

```




`Out:     (std::vector<cv::Point2f> &) {  }`





```c++
vector<uchar> status; // Vector of unsigned charecters, that is going to contain the status 

```




`Out:     (std::vector<uchar> &) {  }`





```c++
vector<float> error_das; //error vector in optical flow algorithm

```




`Out:     (std::vector<float> &) {  }`





```c++
OrbFeatureDetector detector; // In this detector object we are going to store the feature vectors in a loop

```




`Out:     (cv::OrbFeatureDetector &) @0x7ff133d56150`





```c++
vector<KeyPoint> keypoints; 
// keypoints is a vector. It is a vector of KeyPoint-s where each KeyPoint correspond to a point detected 
// by a feature detector, e.g., Harris Corner Detector. Each keypoint is charecterized by the position, scale,
// orientation in the image etc. In each of the detector objects we essentially store etector vectors

```




`Out:     (std::vector<cv::KeyPoint> &) {  }`





```c++
input.read(frameVideo); // This essentially will read the first frame of the video
```




`Out:     (bool) true`





```c++
detector(frameVideo, Mat(), keypoints); // detect the features in the first frame of the video 
// object using OrbFeatureDetector by focusing on the entire frameVideo (denoted by Mat()), and put
// those features in the keypoints vector

```




`Out:     (void) @0x7ffdfad0d850`





```c++
//The optical flow algorihtm expects the feature vector as a vector of Point2f-s, and not a keypoints vector, 
// so we first convert the keypoints vector to Point2f vector

KeyPoint::convert(keypoints, points);
```




`Out:     (void) @0x7ffdfad0d850`





```c++
frameVideo.copyTo(frameVideo_prev); // copy the contents of the first frame to frameVideo_prev

```




`Out:     (void) @0x7ffdfad0d850`





```c++
points_prev=points; 
```




`Out:     (std::vector &) { @0x7ffdfad0d2f0, some long output ..., @0x7ffdfad0d2f0 }`





```c++
for (;;)
{
if(!input.read(frameVideo)) //if the frames of the videos are messed up for some reason
{break;} // Stop!

// Calculate the optical flow between frameVideo and frameVideo_prev
calcOpticalFlowPyrLK(frameVideo_prev, frameVideo,points_prev, points,status, error_das);

// ---------------------------------------------------------------------------

// Parameter description of calcOpticalPyrLK
// void calcOpticalFlowPyrLK(InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts, 
    // OutputArray status, OutputArray err)
//
//
// prevImg – first 8-bit input image or pyramid constructed by buildOpticalFlowPyramid().
// -------
// nextImg – second input image or pyramid of the same size and the same type as prevImg.
// -------
// prevPts – vector of 2D points for which the flow needs to be found; point coordinates must be
// -------
// single-precision floating-point numbers.

// nextPts – output vector of 2D points (with single-precision floating-point coordinates) containing
// -------
// the calculated new positions of input features in the second image; 
    // when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have the same size as in the input.

// status – output status vector (of unsigned chars); each element of the vector is set to 1 if the
// ------
 //flow for the corresponding features has been found, otherwise, it is set to 0.

// err – output vector of errors; each element of the vector is set to an error for the corresponding
// --- 
//feature, type of the error measure can be set in flags parameter; if the flow wasn’t found then the error 
    // is not defined (use the status parameter to find such cases).

// --------------------------------------------------------------------------

// We are interested to see how the features are moving in successive frames, so we are going to 
// make the current frameVideo as the frameVideo_prev in the subsequent loops

frameVideo.copyTo(frameVideo_prev);

points_prev=points; // in the next loop points_prev becomes points


// detector(frameVideo, Mat(), keypoints); // detect the features in frameVideo using OrbFeatureDetector by not 
// focusing on the entire frameVideo (denoted by Mat()), and put those features in the keypoints vector

for (int i=0; i < points.size(); i++) // Denote all the features detected by a red circle and show them in the video
{
circle(frameVideo, points[i], 2, Scalar(0,0,255), 1); 
}

imshow("Window", frameVideo); // Note that if it does not break the program will flow to this line, 
    // so this line acts like an else part, which will show the frames of the video

char c=waitKey(30);// The video will be shown until the keyboard charecter associated with c 
    // is pressed, which we will set to ESC

if(c==27)// 27 corresponds to ESC
{break;} // Stop the video then
}
```




​    



We should see an output video, where the features of the video frames will be shown as red circles, and the corresponding optical flow of the features will be displayed as the video progresses.
