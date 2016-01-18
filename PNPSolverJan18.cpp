#include "PNPSolver.h"
#include "opencv2/calib3d/calib3d.hpp"

PNPSolver::PNPSolver(KinectCamera * kinectCamera)
 : camera(kinectCamera)
{

}

PNPSolver::~PNPSolver()
{

}

void PNPSolver::getRelativePose(isam::Pose3d &pose,
                                std::vector<std::pair<int2, int2> > & inliers,
                                std::vector<InterestPoint *> & scene,
                                std::vector<InterestPoint *> & model)
{
    assert(scene.size() == model.size());

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;

    for(size_t i = 0; i < scene.size(); i++)
    {
        points2d.push_back(cv::Point2f(model.at(i)->u, model.at(i)->v));
        points3d.push_back(cv::Point3f(scene.at(i)->X, scene.at(i)->Y, scene.at(i)->Z));
    }

    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1); //Zero distortion: Deal with it

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);

    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

    cv::Mat inliersCv;

    cv::solvePnPRansac(points3d,
                       points2d,
                       *camera->intrinsicMatrix,
                       distCoeffs,
                       rvec,
                       tvec,
                       false,
                       2000,
                       7,
                       20,
                       inliersCv);

    for(int i=0; i<points3d.size();i++)
    {
        cout<<points3d[i]<<endl;
    }
  //  cout<<*camera->intrinsicMatrix


    cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
    cv::Rodrigues(rvec, R_matrix);

    {
        // draw reprojection error
        /*
        CvDraw::draw_reprojection_error(const vector<cv::Point3f> &pts_3d,
                                     const vector<cv::Point2f> &pts_2d,
                                     const cv::Mat & camera_intrinsic_matrix,
                                     const cv::Mat & rotation,
                                     const cv::Mat & translation,
                                     const cv::Mat & distCoeffs,
                                     vector<float> & reproj_errors,
                                     cv::Mat & error_image)
                                     */
        printf("image size, width, height: %d %d\n", curImage_.cols, curImage_.rows);
        assert(curImage_.rows == 480);
        assert(curImage_.cols == 640);

        vector<float> reproj_errors;
        CvDraw::draw_reprojection_error(points3d, points2d, *camera->intrinsicMatrix, rvec, tvec, distCoeffs, reproj_errors, curImage_);

        int num = rand()%256;
        char buf[1024] = {NULL};
        sprintf(buf, "/home/lili/workspace/SLAM/evaluation/fr1_desk/trial2/PnP/reproj_%d.jpg", num);
        cv::imwrite(buf, curImage_);
        printf("save image %s\n", buf);



    }

    T(0,0) = R_matrix.at<double>(0,0);
    T(0,1) = R_matrix.at<double>(0,1);
    T(0,2) = R_matrix.at<double>(0,2);
    T(1,0) = R_matrix.at<double>(1,0);
    T(1,1) = R_matrix.at<double>(1,1);
    T(1,2) = R_matrix.at<double>(1,2);
    T(2,0) = R_matrix.at<double>(2,0);
    T(2,1) = R_matrix.at<double>(2,1);
    T(2,2) = R_matrix.at<double>(2,2);
    T(0,3) = tvec.at<double>(0);
    T(1,3) = tvec.at<double>(1);
    T(2,3) = tvec.at<double>(2);
    T(3,0) = 0;
    T(3,1) = 0;
    T(3,2) = 0;
    T(3,3) = 1;

    //cout<<"T is correct "<<endl;

    Eigen::MatrixXd isamM(3, 3);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            isamM(i, j) = R_matrix.at<double>(i, j);
        }
    }

    isam::Rot3d R(isamM);

    pose = isam::Pose3d(tvec.at<double>(0),
                        tvec.at<double>(1),
                        tvec.at<double>(2),
                        R.yaw(),
                        R.pitch(),
                        R.roll());

    cout<<"Pose is correct"<<endl;

    for(int i = 0; i < inliersCv.rows; ++i)
    {
        int n = inliersCv.at<int>(i);
        int2 corresp1 = {int(scene.at(n)->u), int(scene.at(n)->v)};
        int2 corresp2 = {int(model.at(n)->u), int(model.at(n)->v)};
        inliers.push_back(std::pair<int2, int2>(corresp1, corresp2));
    }
    //cout<<"solvePnP is correct"<<endl;
}


using namespace cv;
void CvDraw::draw_match_vertical(const cv::Mat &image1, const cv::Mat &image2,
                                 const vector< cv::Point2d > & pts1,
                                 const vector< cv::Point2d > & pts2,
                                 cv::Mat &matches, const int sample_num)
{
    assert(image1.channels() == 3);
    assert(image2.channels() == 3);
    assert(image1.type() == CV_8UC3);
    assert(image2.type() == CV_8UC3);
    assert(pts1.size() == pts2.size());

    int gap = 10;
    int w = std::max(image1.cols, image2.cols);
    int h = image1.rows + image2.rows + gap;
    matches = cv::Mat(h, w, CV_8UC3);

    // copy images
    cv::Mat roi(matches, Rect(0, 0, image1.cols, image1.rows));
    image1.copyTo(roi);

    roi = matches(Rect(0, image1.rows + gap, image2.cols, image2.rows));
    image2.copyTo(roi);

    // draw lines
    for (int i = 0; i<pts1.size(); i += sample_num) {
        cv::Point p1(pts1[i].x, pts1[i].y);
        cv::Point p2(pts2[i].x, pts2[i].y + image1.rows + gap);
        cv::line(matches, p1, p2, cv::Scalar(0, 0, 255));
    }
}

void CvDraw::draw_reprojection_error(const vector<cv::Point3f> &pts_3d,
                                     const vector<cv::Point2f> &pts_2d,
                                     const cv::Mat & camera_intrinsic_matrix,
                                     const cv::Mat & rotation,
                                     const cv::Mat & translation,
                                     const cv::Mat & distCoeffs,
                                     vector<float> & reproj_errors,
                                     cv::Mat & error_image)
{
    assert(pts_3d.size() == pts_2d.size());

    // project world point to image space
    vector<Point2f> projectedPoints;
    projectedPoints.resize(pts_3d.size());
    cv::projectPoints(Mat(pts_3d), rotation, translation, camera_intrinsic_matrix, distCoeffs, projectedPoints);

    CvDraw::draw_cross(error_image, pts_2d, CvDraw::red());
    CvDraw::draw_cross(error_image, projectedPoints, CvDraw::green());

    // draw reprojection error
    for (int i = 0; i<pts_2d.size(); i++) {
        double dx = pts_2d[i].x - projectedPoints[i].x;
        double dy = pts_2d[i].y - projectedPoints[i].y;
        double err = sqrt(dx * dx + dy * dy);
        reproj_errors.push_back(err);

        cv::Point p1 = cv::Point(pts_2d[i].x, pts_2d[i].y);
        cv::Point p2 = cv::Point(projectedPoints[i].x, projectedPoints[i].y);
        cv::line(error_image, p1, p2, CvDraw::blue());
    }
}

void CvDraw::draw_cross(cv::Mat & image,
                        const vector<cv::Point2f> & pts,
                        const cv::Scalar & color,
                        const int length)
{
    assert(image.channels() == 3);

    for (unsigned int i = 0; i<pts.size(); i++)
    {
        //center point
        int px = pts[i].x;
        int py = pts[i].y;

        cv::Point p1, p2, p3, p4;

        int h_l = length/2;
        p1 = cv::Point(px - h_l, py);
        p2 = cv::Point(px + h_l, py);
        p3 = cv::Point(px, py - h_l);
        p4 = cv::Point(px, py + h_l);

        cv::line(image, p1, p2, color);
        cv::line(image, p3, p4, color);
    }
}

void CvDraw::copy_rgb_image(const unsigned char * data,
                            int img_width,
                            int img_height,
                            cv::Mat &image)
{
//     Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);
    cv::Mat temp = cv::Mat(img_height, img_width, CV_8UC3, (void*)data);
    image = temp.clone();
}

cv::Scalar CvDraw::red()
{
    return cv::Scalar(0, 0, 255);

}

cv::Scalar CvDraw::green()
{
    return cv::Scalar(0, 255, 0);
}

cv::Scalar CvDraw::blue()
{
    return cv::Scalar(255, 0, 0);
}


