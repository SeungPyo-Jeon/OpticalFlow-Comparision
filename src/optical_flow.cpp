#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

#define MODE 1
// Mode 0 :
// OpenCV calacOpticalFlowPyrLK
// MODE 1
#include "SingleLayer-LucasKanade.h"
// MODE 2, 3
#include "SingleLayer-ORB_GaussNewton.h"
using namespace std;

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
        switch (MODE)
        {

        case 0:
            break;
        case 1: // SingleLayer-LucasKanade
            SingleLayer_LucasKanade lk;
            local_optical_flow = lk.calcOpticalFlow(pre_img, cur_img);
            OF = *lk.drawOpticalFlow(&OF, local_optical_flow, idx);
            break;
        case 2:
            SingleLayer_ORB_GaussNewton orb_single;
            local_optical_flow = orb_single.calcOpticalFlow(pre_img, cur_img, false, idx);
            OF = *orb_single.drawOpticalFlow(&OF, local_optical_flow);
            break;

        case 3:
            SingleLayer_ORB_GaussNewton orb_multi;
            local_optical_flow = orb_multi.calcOpticalFlow(pre_img, cur_img, true, idx);
            OF = *orb_multi.drawOpticalFlow(&OF, local_optical_flow);
            break;
        }

        // cv::imshow("image", cur_img);
        // cv::imshow("Optical flow", OF);
        cv::waitKey();
        pre_img = cur_img;
    }

    return 0;
}
