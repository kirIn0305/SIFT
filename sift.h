#ifndef __SIFT_H

#define __SIFT_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

//================================================================================
// Descriptor Struct
//================================================================================
struct DESCRIPTOR
{
    double x; // cordinate x
    double y; // cordinate y
    double sig; // sigma
    double org; // orientation

    double v[128]; // SIFT Descriptor

    DESCRIPTOR(){x=y=0.0; sig=org=0.0;}
    DESCRIPTOR(double _x, double _y, double _sig, double _org)
    {
        x=_x; y=_y; sig=_sig; org=_org;
    }
    ~DESCRIPTOR(){}
};


void SIFT(cv::Mat &src, list<DESCRIPTOR*> &des);

#endif
