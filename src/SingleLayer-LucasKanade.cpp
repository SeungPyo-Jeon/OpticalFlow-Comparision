#include "SingleLayer-LucasKanade.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

vector<vector<float>> SingleLayer_LucasKanade::calcOpticalFlow(cv::Mat pre, cv::Mat cur)
{
    int h = pre.size().height;
    int w = pre.size().width;
    cv::Mat TEMP_CUR, TEMP_PRE;
    cv::Mat I_x, I_y, I_t;

    // cv::Mat kernel_x = (cv::Mat_<double>(3, 3) << 0, 1, -1, 0, 1, -1, 0, 0, 0);
    // cv::Mat kernel_y = (cv::Mat_<double>(3, 3) << 0, 1, 1, 0, -1, -1, 0, 0, 0);
    cv::Mat kernel_x = (cv::Mat_<double>(3, 3) << 0, -1, 1, 0, -1, 1, 0, -1, 1);
    cv::Mat kernel_y = (cv::Mat_<double>(3, 3) << 1, 1, 1, -1, -1, -1, 0, 0, 0);
    cv::Mat kernel_t = (cv::Mat_<double>(3, 3) << 0, 1, 1, 0, 1, 1, 0, 0, 0);
    kernel_x = kernel_x / 6;
    kernel_y = kernel_y / 6;
    kernel_t = kernel_t / 6;

    cv::filter2D(cur, TEMP_CUR, CV_32F, kernel_x, cv::Point(-1, -1), 0, 4);
    cv::filter2D(pre, TEMP_PRE, CV_32F, kernel_x, cv::Point(-1, -1), 0, 4); // CV_16S
    I_x = TEMP_PRE + TEMP_CUR;

    cv::filter2D(cur, TEMP_CUR, CV_32F, kernel_y, cv::Point(-1, -1), 0, 4);
    cv::filter2D(pre, TEMP_PRE, CV_32F, kernel_y, cv::Point(-1, -1), 0, 4); // CV_16S
    I_y = TEMP_PRE + TEMP_CUR;

    cv::filter2D(cur, TEMP_CUR, CV_32F, kernel_t, cv::Point(-1, -1), 0, 4);
    cv::filter2D(pre, TEMP_PRE, CV_32F, -kernel_t, cv::Point(-1, -1), 0, 4); // CV_16S
    I_t = TEMP_PRE + TEMP_CUR;

    cv::Mat A, B, u;
    vector<vector<float>> local_of;
    for (int dh = 0; dh < h / PATCH_SIZE; dh++)
    {
        for (int dw = 0; dw < w / PATCH_SIZE; dw++)
        {
            A = I_x(cv::Rect(dw * PATCH_SIZE, dh * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)).clone();
            B = I_y(cv::Rect(dw * PATCH_SIZE, dh * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)).clone();
            A = A.reshape(1, PATCH_SIZE * PATCH_SIZE);
            B = B.reshape(1, PATCH_SIZE * PATCH_SIZE);
            cv::hconcat(A, B, A);
            B = -I_t(cv::Rect(dw * PATCH_SIZE, dh * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)).clone();
            B = B.reshape(1, PATCH_SIZE * PATCH_SIZE);

            u = (A.t() * A + cv::Mat::eye(cv::Size(2, 2), CV_32F) * 0.01).inv() * A.t() * B;

            vector<float> vec_u;
            vec_u.push_back(u.at<float>(0, 0));
            vec_u.push_back(u.at<float>(1, 0));
            local_of.push_back(vec_u);
        }
    }

    return local_of;
}

cv::Mat *SingleLayer_LucasKanade::drawOpticalFlow(cv::Mat *OF, vector<vector<float>> local_optical_flow, int idx)
{
    int w = OF->size().width;
    int h = OF->size().height;
    // patch 마다 optical flow 그리기.
    for (int dh = 0; dh < h / PATCH_SIZE; dh++)
    {
        for (int dw = 0; dw < w / PATCH_SIZE; dw++)
        {
            cv::Point origin = cv::Point((dw * PATCH_SIZE) + PATCH_SIZE / 2, (dh * PATCH_SIZE) + PATCH_SIZE / 2);
            vector<float> vec_u = local_optical_flow[dh * (w / PATCH_SIZE) + dw];
            cv::Point optical_flow = cv::Point(vec_u[0] * 3, vec_u[1] * 3);
            cout << vec_u[0] << " " << vec_u[1] << endl;
            cv::line(*OF, origin, optical_flow + origin, cv::Scalar(0, 0, 255), 1.85);
            cv::circle(*OF, origin, 1.5, cv::Scalar(0, 255, 255), 1);
        }
    }
    cv::imshow("tracked single level", *OF);
    // std::filesystem::create_directories("./LK_SingleLayer");
    //  cv::imwrite("./LK_SingleLayer/" + to_string(idx) + ".png", *OF);
    return OF;
}
