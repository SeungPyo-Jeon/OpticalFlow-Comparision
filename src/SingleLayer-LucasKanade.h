#ifndef SINGLELAYER_LK_H_
#define SINGLELAYER_LK_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#define PATCH_SIZE 27

using namespace std;

class SingleLayer_LucasKanade
{
public:
    vector<vector<float>> calcOpticalFlow(cv::Mat pre, cv::Mat cur);
    vector<vector<float>> LucasKanade(int stage, cv::Mat pre, cv::Mat cur);
    cv::Mat *drawOpticalFlow(cv::Mat *OF, vector<vector<float>> local_optical_flow, int idx);
};
#endif