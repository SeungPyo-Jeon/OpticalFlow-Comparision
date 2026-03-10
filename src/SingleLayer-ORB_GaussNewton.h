#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

class OpticalFlowTracker
{
public:
    OpticalFlowTracker(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const vector<cv::KeyPoint> &kp1_,
        vector<cv::KeyPoint> &kp2_,
        vector<int> &success_,
        bool inverse_ = true, bool has_initial_ = false) : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_) {}

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const vector<cv::KeyPoint> &kp1;
    vector<cv::KeyPoint> &kp2;
    vector<int> &success;
    bool inverse = true;
    bool has_initial = false;
};

class SingleLayer_ORB_GaussNewton
{
public:
    void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const vector<cv::KeyPoint> &kp1,
        vector<cv::KeyPoint> &kp2,
        vector<int> &success,
        bool inverse, bool has_initial);

    void OpticalFlowMultiLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const vector<cv::KeyPoint> &kp1,
        vector<cv::KeyPoint> &kp2,
        vector<int> &success,
        bool inverse);
    vector<vector<float>> calcOpticalFlow(cv::Mat pre, cv::Mat cur, bool multi, int idx);
    cv::Mat *drawOpticalFlow(cv::Mat *OF, vector<vector<float>> local_optical_flow);
};
