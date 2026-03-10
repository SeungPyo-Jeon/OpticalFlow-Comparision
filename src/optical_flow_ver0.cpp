#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

using namespace std;

#define PATCH_SIZE 55

vector<vector<float>> calcOpticalFlow(cv::Mat pre, cv::Mat cur);
vector<vector<float>> LucasKanade(int stage, cv::Mat pre, cv::Mat cur);

int main()
{
    cv::Mat cur_img, pre_img, OF;             // 현재 t시점의 frame, t-1 시점의 frame, Optical flow 표시할 frame
    vector<vector<float>> local_optical_flow; // patch에 해당하는 Optical flow vector를 담은 리스트
    int w, h;                                 // 이미지 크기

    // 데이터 불러오고 정렬하기
    string base_path = "../data/KITTI07/image_0/";
    // string base_path = "../data/Mirae/";
    vector<string> paths;
    for (const auto &entry : std::filesystem::directory_iterator(base_path))
    {
        paths.push_back(entry.path().stem());
    }
    sort(paths.begin(), paths.end());

    // t = 0 frame 미리 불러오기
    pre_img = cv::imread(base_path + string(paths[0]) + string(".png"), cv::IMREAD_GRAYSCALE);
    w = pre_img.size().width;
    h = pre_img.size().height;

    for (int idx = 1; idx < paths.size(); idx++)
    {
        // 현재 t 시점 frame grayscale 로드
        cur_img = cv::imread(base_path + string(paths[idx]) + string(".png"), cv::IMREAD_GRAYSCALE);
        cv::cvtColor(cur_img, OF, cv::COLOR_GRAY2BGR);

        // Optical flow 구하기
        local_optical_flow = LucasKanade(1, pre_img, cur_img);
        // patch 마다 optical flow 그리기.
        for (int dh = 0; dh < h / PATCH_SIZE; dh++)
        {
            for (int dw = 0; dw < w / PATCH_SIZE; dw++)
            {
                cv::Point origin = cv::Point((dw * PATCH_SIZE) + PATCH_SIZE / 2, (dh * PATCH_SIZE) + PATCH_SIZE / 2);
                vector<float> vec_u = local_optical_flow[dh * (w / PATCH_SIZE) + dw];
                cv::Point optical_flow = cv::Point(vec_u[0], vec_u[1]);
                cout << vec_u[0] << " " << vec_u[1] << endl;
                cv::line(OF, origin, optical_flow + origin, cv::Scalar(255, 255, 0), 3);
            }
        }

        cv::imshow("image", cur_img);
        cv::imshow("low", OF);
        cv::waitKey();
        pre_img = cur_img;
    }

    return 0;
}

vector<vector<float>> calcOpticalFlow(cv::Mat pre, cv::Mat cur)
{
    int h = pre.size().height;
    int w = pre.size().width;
    cv::Mat TEMP_CUR, TEMP_PRE;
    cv::Mat I_x, I_y, I_t;

    cv::Mat kernel_x = (cv::Mat_<double>(3, 3) << 0, 1, -1, 0, 1, -1, 0, 0, 0);
    cv::Mat kernel_y = (cv::Mat_<double>(3, 3) << 0, 1, 1, 0, -1, -1, 0, 0, 0);
    cv::Mat kernel_t = (cv::Mat_<double>(3, 3) << 0, 1, 1, 0, 1, 1, 0, 0, 0);
    kernel_x = kernel_x / 4;
    kernel_y = kernel_y / 4;
    kernel_t = kernel_t / 4;

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

vector<vector<float>> LucasKanade(int stage, cv::Mat pre, cv::Mat cur)
{
    int h = pre.size().height;
    int w = pre.size().width;

    cv::Mat low_pre, low_cur, low_OF;
    cv::GaussianBlur(pre, low_pre, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_DEFAULT);
    cv::GaussianBlur(cur, low_cur, cv::Size(3, 3), 1.0, 1.0, cv::BORDER_DEFAULT);

    cv::resize(low_pre, low_pre, cv::Size(w / 2, h / 2), 0, 0, cv::INTER_LINEAR);
    cv::resize(low_cur, low_cur, cv::Size(w / 2, h / 2), 0, 0, cv::INTER_LINEAR);
    // if (stage == 8)
    //{
    //     cv::Sobel()
    // }
    //
    // low_OF = LucasKanade(stage + 1, low_pre, low_cur);

    return calcOpticalFlow(pre, cur);
}