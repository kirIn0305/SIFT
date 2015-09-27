#include <iostream>
using namespace std;

#include "sift.h"

int main(int argc, char const* argv[])
{
    cv::Mat src1 = cv::imread( "./image2.png", CV_LOAD_IMAGE_GRAYSCALE );

    cout << "SIFT Start::" << endl;

    clock_t str=clock();

    SIFT(src1);

    clock_t end=clock();
    cout << "time = " << (end-str) / (double)CLOCKS_PER_SEC << " [sec]" << endl;

    return 0;
}
