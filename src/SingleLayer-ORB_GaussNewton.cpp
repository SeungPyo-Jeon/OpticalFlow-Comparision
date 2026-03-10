#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <filesystem>
#include "SingleLayer-ORB_GaussNewton.h"

using namespace std;

vector<vector<float>> SingleLayer_ORB_GaussNewton::calcOpticalFlow(cv::Mat pre, cv::Mat cur, bool multi, int idx)
{
    // 초기화
    vector<vector<float>> optical_flow;
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    // cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
    //  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    //  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // 1단계 Oriented FAST 코너 검출
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(pre, keypoints_1);
    /* ## feature extraction & feature matching
    // detector->detect(cur, keypoints_2);

    // 2단계 코너 기준으로 BRIEF descriptor 계산( Binary robust invariant elementary feature )
    descriptor->compute(pre, keypoints_1, descriptors_1);
    descriptor->compute(cur, keypoints_2, descriptors_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost= " << time_used.count() << " seconds." << endl;

    cv::Mat outimg1;
    cv::drawKeypoints(pre, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg1);

    // 3단계 해밍 거리 사용해 두 이미지의 BRIEF 설명자 일치
    vector<cv::DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_1, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds." << endl;

    // 4 단계 매치 포인트 페어 스크리닝
    // 최소 최대 거리계산
    auto min_max = minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2)
                                  { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    cout << "Max dist: " << max_dist << endl;
    cout << "Min dist: " << min_dist << endl;

    vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // 5단계
    cv::Mat img_match;
    cv::Mat img_goodmatch;

    cv::drawMatches(pre, keypoints_1, cur, keypoints_2, matches, img_match);
    cv::drawMatches(pre, keypoints_1, cur, keypoints_2, good_matches, img_goodmatch);
    cv::imshow("all matches", img_match);
    */
    vector<cv::KeyPoint> kp2_single;
    vector<int> success_single;
    if (multi)
    {
        SingleLayer_ORB_GaussNewton::OpticalFlowMultiLevel(pre, cur, keypoints_1, kp2_single, success_single, true);
    }
    else
    {
        SingleLayer_ORB_GaussNewton::OpticalFlowSingleLevel(pre, cur, keypoints_1, kp2_single, success_single, true, false);
    }
    cv::Mat img2_single;
    cv::cvtColor(cur, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++)
    {
        if (success_single[i])
        {
            cv::Point2f flow = -keypoints_1[i].pt + kp2_single[i].pt;
            cv::line(img2_single, kp2_single[i].pt, kp2_single[i].pt + flow * 2, cv::Scalar(0, 250, 0), 1);
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 255, 255), 1);
        }
    }

    cv::imshow("tracked single level", img2_single);
    /*
    if (multi)
    {
        std::filesystem::create_directories("./ORB_MultiLayer");
        cv::imwrite("./ORB_MultiLayer/" + to_string(idx) + ".png", img2_single);
    }
    else
    {
        std::filesystem::create_directories("./ORB_SingleLayer");
        cv::imwrite("./ORB_SingleLayer/" + to_string(idx) + ".png", img2_single);
    }*/
    return optical_flow;
}
cv::Mat *SingleLayer_ORB_GaussNewton::drawOpticalFlow(cv::Mat *OF, vector<vector<float>> local_optical_flow)
{
    return OF;
}

void SingleLayer_ORB_GaussNewton::OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<int> &success,
    bool inverse, bool has_initial)
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    // cv::parallel_for_(cv::Range(0, kp1.size()), bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
    cv::parallel_for_(cv::Range(0, kp1.size()), [&](const cv::Range &range)
                      { tracker.calculateOpticalFlow(range); });
}

void SingleLayer_ORB_GaussNewton::OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const vector<cv::KeyPoint> &kp1,
    vector<cv::KeyPoint> &kp2,
    vector<int> &success,
    bool inverse)
{
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    vector<cv::Mat> pyr1, pyr2;
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp : kp1)
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--)
    {
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        if (level > 0)
        {
            for (auto &kp : kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp : kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp : kp2_pyr)
        kp2.push_back(kp);
}

inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols - 1)
        x = img.cols - 2;
    if (y >= img.rows - 1)
        y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    // get pixel value by bilinear interpolation
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) +
           xx * (1 - yy) * img.at<uchar>(y, x_a1) +
           (1 - xx) * yy * img.at<uchar>(y_a1, x) +
           xx * yy * img.at<uchar>(y_a1, x_a1);
}

void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range)
{
    int half_patch_size = 8;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0;
        if (has_initial)
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero(); // Hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero(); // bias
        Eigen::Vector2d J;                           // Jacobian
        for (int iter = 0; iter < iterations; iter++)
        {
            if (inverse == false)
            {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }
            else
            {
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;

            for (int x = -half_patch_size; x <= half_patch_size; x++)
            {
                for (int y = -half_patch_size; y <= half_patch_size; y++)
                {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    if (inverse == false)
                    {
                        J = -1.0 * Eigen::Vector2d(
                                       0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                              GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                       0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                              GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                    }
                    else // if (iter == 0)
                    {
                        J = -1.0 * Eigen::Vector2d(
                                       0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                              GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                       0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                              GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1)));
                    }
                    if (inverse == false || iter == 0)
                    {
                        H += J * J.transpose();
                    }
                    cost += error * error;
                    b += -error * J;
                }
            }
            Eigen::Vector2d update = H.ldlt().solve(b);
            if (isnan(update[0]))
            {
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost)
            {
                break;
            }

            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2)
            {
                break;
            }
            if (i == 0)
            { // 첫 번째 포인트만 디버깅
                cout << "Iter: " << iter << ", dx: " << dx << ", dy: " << dy << ", Cost: " << cost << endl;
            }
        }

        success[i] = succ;

        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}
