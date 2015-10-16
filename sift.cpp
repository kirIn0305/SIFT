#include <iostream>
#include <sstream>
using namespace std;

#include <vector>
#include <list>
#include <math.h>
#include "sift.h"

//================================================================================
// SIFT parameter
//================================================================================
const double SIG0 = 1.6;
const int S = 3;
const int MINW = 20;

int OCT;

const double Rth = 10.0;
const double TH_EDG = (Rth+1)*(Rth+1) / Rth;
/* const double TH_POW = 0.03; */
const double TH_POW = 5.5;

# define KEYPOINT_BIN 36
//================================================================================
// Key Point Candidate struct 
//================================================================================
struct KEYPOINT {
    double x;
    double y;

    int o;
    int s;

    double hst[KEYPOINT_BIN]; //histgram

    //constract
    KEYPOINT(){x=y=0.0; o=s=0;}
    KEYPOINT(double _x, double _y, int _o, int _s)
    {
        x=_x; y=_y; o=_o; s=_s;
    }
    ~KEYPOINT(){}
};

//================================================================================
// calculate inverse matrix
//================================================================================
void calc_inv(double (*mat)[3], double (*inv)[3])
{
    int i,j,k;
    double buf;

    // initialization
    for(i = 0; i < 3; i++){
        for (j = 0; j < 3; j++) {
            inv[i][j] = 0;
            inv[i][i] = 1;
        }
    }

    for(i=0;i<3;i++){
        buf=1/mat[i][i];
        for(j=0;j<3;j++){
        mat[i][j]*=buf;
        inv[i][j]*=buf;
        }
          
        for(j=0;j<3;j++){
            if(i!=j){
                buf=mat[j][i];
                for(k=0;k<3;k++){
                    mat[j][k]-=mat[i][k]*buf;
                    inv[j][k]-=inv[i][k]*buf;
                }
            }
        }
    }
}

//================================================================================
// down sampling image generator for octave
//================================================================================
void down_sampling(cv::Mat &src)
{
    int W = src.cols;
    int H = src.rows;

    cv::vector<cv::Mat> dst;
    //---------------------- Image pyramid level 1  ----------------------
    cv::buildPyramid(src, dst, 1);
    src = dst[1];
}

//================================================================================
// Get KeyPoints candidate
//================================================================================
void get_keypoints(cv::Mat &src, cv::vector<cv::vector<cv::Mat>> &L, cv::vector<cv::vector<cv::Mat>> &DoG, list<KEYPOINT*> &keys)
{
    cout << "step1:keypoint detection" << endl;

    //---------------------- Rate of increase ----------------------
    double k=pow(2,1/(double)S);

    //---------------------- smoothing image ----------------------
    for (int o = 0; o < OCT; o++) {
        cout << "   octave " << o << " : " <<endl;
        double sig=SIG0;
        for(int s=0;s<S+3;s++){
            //gaussian filter
            cv::GaussianBlur(src, L[o][s], cv::Size(0,0), sig, sig);
            sig*=k;
            cout << "   sig : " << sig << endl;

            // debug
            stringstream file_name;
            file_name << "gaussian/" << o << s << ".png";
            cv::imwrite(file_name.str(), L[o][s]);
        }
        // down sampling
        down_sampling(src);
    }

    //---------------------- Diffrence Of Gaussian ----------------------
    for(int o = 0; o < OCT; o++){
        for (int i = 0; i < S + 2; i++) {
           DoG[o][i] = L[o][i+1] - L[o][i]; 
        }
    }

    //---------------------- flag (alredy detecting cordinate of Key Points) ----------------------
    bool **flg = new bool *[DoG[0][0].cols];
    flg[0] = new bool[DoG[0][0].cols * DoG[0][0].rows];
    for(int x=1;x<DoG[0][0].cols;x++){
        flg[x]=flg[x-1]+DoG[0][0].rows;
    }

    //---------------------- detecting Max from DoG ----------------------
    for(int o=0; o<OCT; o++){
        int W = DoG[o][0].cols;
        int H = DoG[o][0].rows;

        // initialize flag
        for (int x = 1; x < W-1; x++) {
            for (int y = 1; y < H-1; y++) {
               flg[x][y] = false; 
            }
        }

        for (int s = 1; s < S+1; s++) {
            for (int x = 1; x < W-1; x++) {
                for (int y = 1; y < H-1; y++) {
                    if(flg[x][y]) continue;

                    // search 26 px neighbor ( via 3 images)
                    bool is_max = true; //the maximum value ?
                    bool is_min = true; //the minmum value ?
                    for (int ds = s-1; ds <= s+1; ds++) {
                        for (int dx = x-1; dx <= x+1; dx++) {
                            for (int dy = y-1; dy < y+1; dy++) {
                                // pass when center px
                                if(ds==s && dx==x && dy==y) continue;

                                // check maximum and minimum value
                                if (is_max && DoG[o][s].at<uchar>(y, x) <= DoG[o][ds].at<uchar>(dy, dx)) {
                                    is_max = false;
                                }
                                if (is_min && DoG[o][s].at<uchar>(y, x) >= DoG[o][ds].at<uchar>(dy, dx)) {
                                    is_min = false;
                                }
                                // not found values
                                /* if(!is_max && !is_min) */
                                /*     goto next; */
                            }
                        }
                    }

                    // registerate cordinates of KeyPoints
                    if(is_max || is_min){
                        KEYPOINT *tmp = new KEYPOINT((double)x, (double)y, o, s);
                        keys.push_back(tmp);
                        flg[x][y]=true;
                    }

                    /* next:; */
                }
            }
        }
    }
    
    cout << "   number of keypoints = " << keys.size() << endl << endl;

    delete [] flg[0];
    delete [] flg;
}

void localize_keypoints(cv::vector<cv::vector<cv::Mat>> &DoG, list<KEYPOINT*> &keys)
{
    cout << "step2:localize" << endl;

    double mD[3][3]; // Derivative matrix
    double iD[3][3]; // inv Derivative matrix
    double xD[3];    // keypoint cordinate
    double X[3];     // subpixel cordinate

    /* for (int o = 0; o < OCT; o++) { */
    /*     for (int i = 0; i < S + 2; i++) { */
    /*        // debug */
    /*        stringstream file_name; */
    /*        file_name << "DoG/" << o << i << ".png"; */
    /*        cv::imwrite(file_name.str(), DoG[o][i]); */
    /*     } */
    /* } */

    for(list<KEYPOINT*>::iterator it=keys.begin();it!=keys.end();){
        //---------------------- Principal curve ----------------------
        int o = (*it)->o;
        int s = (*it)->s;
        int x = (int)((*it)->x);
        int y = (int)((*it)->y);

        //---------------------- 2D Hessian matrix ----------------------
        // change method of Differential for simplify calculation
        double Dxx = DoG[o][s].at<uchar>(y, x-2) + DoG[o][s].at<uchar>(y, x+2) - 2 * DoG[o][s].at<uchar>(y,x); 
        double Dyy = DoG[o][s].at<uchar>(y-2, x) + DoG[o][s].at<uchar>(y+2, x) - 2 * DoG[o][s].at<uchar>(y,x); 
        double Dxy = DoG[o][s].at<uchar>(y-1, x-1) - DoG[o][s].at<uchar>(y+1, x-1) - DoG[o][s].at<uchar>(y-1, x+1) + DoG[o][s].at<uchar>(y+1, x+1);

        //---------------------- trace and Determinant ----------------------
        double trc = Dxx + Dyy;
        double det = Dxx * Dyy - Dxy * Dxy;

        //---------------------- Decision erase keypoints ----------------------
        if(trc*trc / det >= TH_EDG){
            delete (*it);
            it = keys.erase(it);
            continue;
        }
        //---------------------- Decision erase keypoints when low contrast keypoints ----------------------
        int sm1 = (s-1<0)? 0 : s-1;
        int sm2 = (s-2<0)? 0 : s-2;
        int sp1 = (s+1 >= S+2)? S+1 : s+1;
        int sp2 = (s+2 >= S+2)? S+1 : s+2;

        //---------------------- sub pixel estimate ----------------------
        double Dx = DoG[o][s].at<uchar>(y, x-1) - DoG[o][s].at<uchar>(y, x+1);
        double Dy = DoG[o][s].at<uchar>(y-1, x) - DoG[o][s].at<uchar>(y+1, x);
        double Ds = DoG[o][sm1].at<uchar>(y, x) - DoG[o][sp1].at<uchar>(y, x);

        double Dss = DoG[o][sm2].at<uchar>(y, x) - DoG[o][sp2].at<uchar>(y, x) + 2 * DoG[o][s].at<uchar>(y, x);
        double Dxs = DoG[o][sm1].at<uchar>(y, x-1) - DoG[o][sm1].at<uchar>(y, x+1) - DoG[o][sp1].at<uchar>(y, x-1) + DoG[o][sp1].at<uchar>(y, x+1);
        double Dys = DoG[o][sm1].at<uchar>(y-1, x) - DoG[o][sm1].at<uchar>(y+1, x) - DoG[o][sp1].at<uchar>(y-1, x) + DoG[o][sp1].at<uchar>(y+1, x);

        mD[0][0]=Dxx; mD[0][1]=Dxy; mD[0][2]=Dxs;
        mD[1][0]=Dxy; mD[1][1]=Dyy; mD[1][2]=Dys;
        mD[2][0]=Dxs; mD[2][1]=Dys; mD[2][2]=Dss;

        xD[0]=-Dx; xD[1]=-Dy; xD[2]=-Ds;

        //---------------------- inverse matrix ----------------------
        calc_inv(mD, iD);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cout << "iD[" << i << j << "] : " << iD[i][j] << endl;
                
            }
        }

        //---------------------- subpixel cordinate ----------------------
        X[0]=iD[0][0]*xD[0]+iD[0][1]*xD[1]+iD[0][2]*xD[2];
        X[1]=iD[1][0]*xD[0]+iD[1][1]*xD[1]+iD[1][2]*xD[2];
        X[2]=iD[2][0]*xD[0]+iD[2][1]*xD[1]+iD[2][2]*xD[2];
        /* cout << "X[0] : " << X[0] << endl; */
        /* cout << "X[1] : " << X[1] << endl; */
        /* cout << "X[2] : " << X[2] << endl; */

        //---------------------- DoG (subpixel cordinate) ----------------------
        double Dpow=fabs(DoG[o][s].at<uchar>(y, x) + (xD[0]*X[0]+xD[1]*X[1]+xD[2]*X[2])/2);
        cout << "Dpow : " << Dpow << endl;

        //---------------------- Threshold Processing ----------------------
        if(Dpow<TH_POW){
            delete (*it);
            it=keys.erase(it);
        } else it++;
    }


    cout << "   number of keypoints (erased) = " << keys.size() << endl;

    }

cv::Mat gaussian_filter(double sig)
{
    int w=int(ceil(3.0*sig)*2+1);
    int c=(w-1)/2;

    cv::Mat mask(w,w, CV_64F);

    sig=2*sig*sig;

    for (int x = 0; x < w; x++) {
        int px=x-c;
        for (int y = 0; y < w; y++) {
            int py=y-c;
            double dst=px*px+py+py;

            mask.at<double>(y, x) = exp(-dst/sig)/(sig*M_PI);
        }
    }

    return mask;
}

void calc_orientation(cv::vector<cv::vector<cv::Mat>> &L, cv::vector<cv::vector<cv::Mat>> &Fpow, cv::vector<cv::vector<cv::Mat>> &Farg, list<KEYPOINT*> &keys )
{
    cout << "step3 : calculation Orientations" << endl;

    //---------------------- calculation Gradient strength and Gradient direction   ----------------------
    // make Gaussian Filter
    cv::vector<cv::Mat> G(S+3);
    double k=pow(2, 1/(double)S);
    for (int s = 0; s < S+3; s++) {
        double sig=pow(k,s+1)*SIG0;
        G[s] = gaussian_filter(sig);
    }


    for(list<KEYPOINT*>::iterator it=keys.begin();it!=keys.end();it++){
        int u = (*it)->x;
        int v = (*it)->y;
        int o = (*it)->o;
        int s = (*it)->s;

        int Rm = (G[s].cols-1) / 2;
        cv::Mat Fpow_tmp(G[s].size(), G[s].type());
        cv::Mat Farg_tmp(G[s].size(), G[s].type());
        double fu, fv;

        for (int i = v-Rm; i < v+Rm; i++) {
            int my = i - v + Rm;
            for (int j = u-Rm; j < u+Rm; j++) {
                int mx = j - u + Rm;

                // ToDO!! Adjust boundary conditions
                if (i < 0) {
                    if (j < 0) {
                        fu = L[o][s+1].at<uchar>(0, 1) - L[o][s+1].at<uchar>(0, 0);
                        fv = L[o][s+1].at<uchar>(1, 0) - L[o][s+1].at<uchar>(0, 0);
                    } else if(j > L[o][s+1].cols) {
                        fu = L[o][s+1].at<uchar>(0, L[o][s+1].cols) - L[o][s+1].at<uchar>(0, L[o][s+1].cols - 1);
                        fv = L[o][s+1].at<uchar>(1, L[o][s+1].cols) - L[o][s+1].at<uchar>(0, L[o][s+1].cols);
                    } else {
                        fu = L[o][s+1].at<uchar>(0, j+1) - L[o][s+1].at<uchar>(0, j-1);
                        fv = L[o][s+1].at<uchar>(1, j) - L[o][s+1].at<uchar>(0, j);
                    }
                } else if(i > L[o][s+1].rows) {
                    if (j < 0) {
                        fu = L[o][s+1].at<uchar>(L[o][s+1].rows, 1) - L[o][s+1].at<uchar>(L[o][s+1].rows, 0);
                        fv = L[o][s+1].at<uchar>(L[o][s+1].rows, 0) - L[o][s+1].at<uchar>(L[o][s+1].rows - 1, 0);
                    } else if(j > L[o][s+1].cols) {
                        fu = L[o][s+1].at<uchar>(L[o][s+1].rows, L[o][s+1].cols) - L[o][s+1].at<uchar>(L[o][s+1].rows, L[o][s+1].cols - 1);
                        fv = L[o][s+1].at<uchar>(L[o][s+1].rows, L[o][s+1].cols) - L[o][s+1].at<uchar>(L[o][s+1].rows-1, L[o][s+1].cols);
                    } else {
                        fu = L[o][s+1].at<uchar>(L[o][s+1].rows, j+1) - L[o][s+1].at<uchar>(L[o][s+1].rows, j-1);
                        fv = L[o][s+1].at<uchar>(L[o][s+1].rows, j) - L[o][s+1].at<uchar>(L[o][s+1].rows-1, j);
                    }
                } else {
                    fu = L[o][s+1].at<uchar>(i, j+1) - L[o][s+1].at<uchar>(i, j-1);
                    fv = L[o][s+1].at<uchar>(i+1, j) - L[o][s+1].at<uchar>(i-1, j);
                } 

                // calc Fpow Farg 
                Fpow_tmp.at<double>(my, mx) = sqrt(fu*fu + fv*fv);
                Farg_tmp.at<double>(my, mx) = (atan2(fv, fu) / M_PI + 1) / 2;
            }
        }

        Fpow[o][s].push_back(Fpow_tmp);
        Farg[o][s].push_back(Farg_tmp);

    }

    cout << "       start histgraming..." << endl;

    for(list<KEYPOINT*>::iterator it=keys.begin();it!=keys.end();it++){
        int x = (*it)->x;
        int y = (*it)->y;
        int o = (*it)->o;
        int s = (*it)->s;

        // initialize histgram
        for (int bin = 0; bin < KEYPOINT_BIN; bin++) {
            (*it)->hst[bin] = 0;
        }


        // Gaussian filter radius
        int Rm = (G[s].cols-1)/2;

        for (int i = x-Rm; i <= x+Rm; i++) {
            int mx = i-x+Rm;
            for (int j = y-Rm; j < y+Rm; j++) {
                int my = j-y+Rm;

                // select bin number from angle
                int bin = (int)floor(KEYPOINT_BIN*Farg[o][s].at<uchar>(my,mx)) % KEYPOINT_BIN;
                /* cout << "       bin : " << bin << endl; */
                (*it)->hst[bin] += (double)(G[s].at<uchar>(my, mx)) * Fpow[o][s].at<double>(my,mx);
                /* cout << "hist[bin] : " << (*it)->hst[bin] << endl; */
            }
        }

        // add histgram smoothing (not writing in reference papaer)
        const int num_smooth = 6;
       for (int i = 0; i < num_smooth; i++) {
            (*it)->hst[0] = ((*it)->hst[KEYPOINT_BIN-1]+(*it)->hst[0]+(*it)->hst[1])/ 3.0;
            for (int bin = 1; bin < KEYPOINT_BIN; bin++) {
                (*it)->hst[bin] = ((*it)->hst[bin-1]+(*it)->hst[bin]+(*it)->hst[(bin+1)%KEYPOINT_BIN]) / 3.0;
            }

       } 
    }
   
    cout << "   number of keypoints = " << keys.size() << endl;
    cout << endl;
}

void calc_descriptor(cv::vector<cv::vector<cv::Mat>> &L, cv::vector<cv::vector<cv::Mat>> &Fpow, cv::vector<cv::vector<cv::Mat>> &Farg, list<KEYPOINT*> &keys, list<DESCRIPTOR*> &des )
{
    cout << "step4 : calculation Description" << endl;
}

void plot_image(cv::Mat &src,list<KEYPOINT*> &keys)
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

    for(list<KEYPOINT*>::iterator it=keys.begin();it!=keys.end();it++){
        double dlt =pow(2,(*it)->o);
        double xo  =dlt*(*it)->x;
        double yo  =dlt*(*it)->y;
        double size=dlt*pow(2,((*it)->s-1)/(double)S);

        //円を描く
        for(int i=0;i<20;i++){

            double x=xo + size*cos((i/19.0)*2*M_PI);
            double y=yo + size*sin((i/19.0)*2*M_PI);

            fprintf(gp,"%f %f\n",x,y);
        }
        fprintf(gp,"\n");
    }
    fprintf(gp,"e\n");

    fflush(gp);
    cout<<endl<<"enter>>";
    getchar();
    pclose(gp);
}


void SIFT(cv::Mat &_src)
{
    cv::Mat src(_src);

    //---------------------- octave ----------------------
    int W = (src.rows < src.cols)? src.rows : src.cols;
    for (OCT = 0; W > MINW; OCT++, W /= 2);
    cout << "octave : " << OCT << endl;
        
    //---------------------- buffer ----------------------
    //gaussian
    cv::vector<cv::vector<cv::Mat>> L(OCT, cv::vector<cv::Mat>(S + 3));
    //DoG
    cv::vector<cv::vector<cv::Mat>> DoG(OCT, cv::vector<cv::Mat>(S + 2));

    //---------------------- Key Point Candidate ----------------------
    list<KEYPOINT*> keys;
    list<DESCRIPTOR*> des;

   
    //---------------------- detection ----------------------
    get_keypoints(src, L, DoG, keys);
    localize_keypoints(DoG, keys);


    //---------------------- description ----------------------
    // Orientations buffer
    /* cv::vector<double> Fpow(keys.size()); */
    /* cv::vector<double> Farg(keys.size()); */
    cv::vector<cv::vector<cv::Mat>> Fpow(OCT, cv::vector<cv::Mat>(S+3));
    cv::vector<cv::vector<cv::Mat>> Farg(OCT, cv::vector<cv::Mat>(S+3));

    calc_orientation(L, Fpow, Farg, keys);
    calc_descriptor(L, Fpow, Farg, keys, des);

    plot_image(_src, keys);
}
