#include <opencv2/opencv.hpp>
#include <iostream>
#include <limits>
#include <csignal>
#include <cfloat>
#include <cstdio>
#include <vector>
#include <array>

static bool stop = false;
void sigIntHandler(int signal) {
  stop = true;
}


//inline
//void find_min(cv::Mat &mat, double *min) {
//  *min = std::numeric_limits<double>::max();
//  double *p = nullptr;
//  for (int i = 0; i < mat.rows; ++i)
//    p = mat.ptr<double>(i);
//    for (int j = 0; j < mat.cols; ++j) {
//      if (p[j] < *min)
//        *min = p[j];
//    }
//}
//
//inline
//void find_max(cv::Mat &mat, double *max) {
//  *max = std::numeric_limits<double>::min();
//  double *p = nullptr;
//  for (int i = 0; i < mat.rows; ++i)
//    p = mat.ptr<double>(i);
//    for (int j = 0; j < mat.cols; ++j) {
//      if (p[j] > *max)
//        *max = p[j];
//    }
//}




int main(int argc, char **argv) {
  cv::VideoCapture cap(0);
  if (!cap.isOpened())
    return -1;

  std::signal(SIGINT, sigIntHandler);
  
  cv::Mat img;
  cv::Mat gray;
  cv::Mat blurred(640, 480, CV_64F);
  cv::Mat rx(640, 480, CV_64F);
  cv::Mat ry(640, 480, CV_64F);
  cv::Mat rxx(640, 480, CV_64F);
  cv::Mat rxy(640, 480, CV_64F);
  cv::Mat ryx(640, 480, CV_64F);
  cv::Mat ryy(640, 480, CV_64F);
  cv::Mat rxxryy(640, 480, CV_64F);
  cv::Mat rxyryx(640, 480, CV_64F);
  //cv::Mat A(640, 480, CV_64F);
  //cv::Mat B(640, 480, CV_64F);
  //cv::Mat lambda1(640, 480, CV_64F);
  //cv::Mat lambda2(640, 480, CV_64F);
  cv::Mat S(640, 480, CV_64F);
  double s_min = 0.;
  double s_max = 0.;
  cv::Scalar RED = cv::Scalar(0, 0, 255);
  cv::Scalar GREEN = cv::Scalar(0, 255, 0);


  std::vector<cv::Point> points;
  for (;!stop;)
  {
    points.clear();
    cap >> img;
    // img = cv::imread("/home/fans/Pictures/kinect_chessboard.png");
    cv::cvtColor(img, gray, CV_BGR2GRAY);
    // cv::equalizeHist(gray, gray);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Scharr(blurred, rx, CV_64F, 1, 0);
    cv::Scharr(blurred, ry, CV_64F, 0, 1);
    cv::Scharr(rx, rxx, CV_64F, 1, 0);
    cv::Scharr(rx, rxy, CV_64F, 0, 1);
    cv::Scharr(ry, ryx, CV_64F, 1, 0);
    cv::Scharr(ry, ryy, CV_64F, 0, 1);
    //A = rxx + ryy;
    //cv::sqrt((rxx - ryy).mul(rxx-ryy) + rxy.mul(4*ryx), B);
    //lambda1 = A + B;
    //lambda2 = A - B;
    
    rxxryy = rxx.mul(ryy);
    rxyryx = rxy.mul(ryx);
    S = rxxryy - rxyryx;
    cv::minMaxIdx(S, &s_min, &s_max);
    s_min *= 0.6;

    double *s = nullptr;
    for (int r = 0; r < img.rows; ++r) {
      s = S.ptr<double>(r);
      for (int c = 0; c < img.cols; ++c) {
        if (s[c] < -1000. && s[c] < s_min) {
          //cv::circle(img, cv::Point(c, r), 1, GREEN);
          points.push_back(cv::Point(c, r));
        }
      }
    }
    std::vector<cv::Point> final_points;
    std::vector<bool> accessed(points.size(), false);
    for (int i = 0; i < points.size(); ++i) {
      if (accessed[i])
        continue;
    
      accessed[i] = true;
      
      double x = points[i].x;
      double y = points[i].y;
    
      int counter = 1;
      for (int j = i; j < points.size(); ++j) {
        if (accessed[j])
          continue;
        else if (std::abs(points[j].x - points[i].x) < 5 &&
                 std::abs(points[j].y - points[i].y) < 5) {
          accessed[j] = true;
          x += points[j].x;
          y += points[j].y;
          counter++;
        }
      }
      x /= counter;
      y /= counter;
      final_points.push_back(cv::Point(x, y));
    }

    std::cout << "{" << points.size() << ", "
              << final_points.size() << "}" << std::endl;
    
    for (auto ele = final_points.begin(); ele != final_points.end(); ele++)
      cv::circle(img, *ele, 5, RED);
        
    

    cv::imshow("img", img);
    cv::waitKey(10);
  }
  
  
  return 0;
}
