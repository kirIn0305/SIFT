#include <iostream>
using namespace std;

#include "sift.h"

void plot_image(cv::Mat &src,list<DESCRIPTOR*> &keys)
{
    FILE *gp=popen("gnuplot","w");

    int W=src.cols;
    int H=src.rows;

    fprintf(gp,"set yrange [] reverse\n");
    fprintf(gp,"set format x ''\n");
    fprintf(gp,"set format y ''\n");
    fprintf(gp,"set size ratio %lf\n",H/(double)W);
    fprintf(gp,"set palette gray\n");
    fprintf(gp,"set xrange [0:%d]\n",W-1);
    /* fprintf(gp,"set yrange [0:%d]\n",H-1); */
    /* fprintf(gp,"set xrange [%d:0]\n",W-1); */
    fprintf(gp,"set yrange [%d:0]\n",H-1);
    fprintf(gp,"unset key\n");
    fprintf(gp,"unset colorbox\n");
    fprintf(gp,"plot '-' matrix with image,'-' w l\n");

    //画像の表示
    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
        fprintf(gp,"%f ", float(src.at<uchar>(y, x)));
        }
        fprintf(gp,"\n");
    }
    fprintf(gp,"e\n");
    fprintf(gp,"e\n");

    for(list<DESCRIPTOR*>::iterator it=keys.begin();it!=keys.end();it++){
        double size = (*it)->sig;

        //円を描く
        for(int i=0;i<20;i++){

            double x=(*it)->x + size*cos((i/19.0)*2*M_PI);
            double y=(*it)->y + size*sin((i/19.0)*2*M_PI);

            fprintf(gp,"%f %f\n",x,y);
        }
        fprintf(gp,"\n");

        // Gradient Orientation draw
        fprintf(gp, "%f %f\n", (*it)->x, (*it)->y);
        fprintf(gp, "%f %f\n", (*it)->x+size*cos((*it)->org), (*it)->y+size*sin((*it)->org));
        fprintf(gp, "\n");
    }
    fprintf(gp,"e\n");

    fflush(gp);
    cout<<endl<<"enter>>";
    getchar();
    pclose(gp);
}


//================================================================================
// main
//================================================================================
int main(int argc, char const* argv[])
{
    cv::Mat src1 = cv::imread( "./lena.png", CV_LOAD_IMAGE_GRAYSCALE );
    list<DESCRIPTOR*> key1;

    cout << "SIFT Start::" << endl;

    clock_t str=clock();

    SIFT(src1, key1);

    clock_t end=clock();
    cout << "time = " << (end-str) / (double)CLOCKS_PER_SEC << " [sec]" << endl;

    plot_image(src1, key1);

    return 0;
}
