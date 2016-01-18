#include "RawLogReader.h"
#include "KeyframeMap.h"
#include "PoseGraph/iSAMInterface.h"
#include "PlaceRecognition/PlaceRecognition.h"
#include "Odometry/FOVISOdometry.h"
#include "Odometry/DVOdometry.h"
#include "PlaceRecognition/KinectCamera.h"
#include "PlaceRecognition/PNPSolver.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;


void drawPoses(std::vector<std::pair<uint64_t, Eigen::Matrix4f> > & poses,
               pcl::visualization::PCLVisualizer & cloudViewer, double r, double g, double b)
{
    static int count = 0;

    for(size_t i = 1; i < poses.size(); i++)
    {
        pcl::PointXYZ p1, p2;

        p1.x = poses.at(i - 1).second(0, 3);
        p1.y = poses.at(i - 1).second(1, 3);
        p1.z = poses.at(i - 1).second(2, 3);

        p2.x = poses.at(i).second(0, 3);
        p2.y = poses.at(i).second(1, 3);
        p2.z = poses.at(i).second(2, 3);

        std::stringstream strs;

        strs << "l" << count++;

        cloudViewer.addLine(p1, p2, r, g, b, strs.str());
    }
}

struct RGBDFile {

    uint64_t rgb_timestamp;
    string real_timestamp;
    string rgb_frame;
    string depth_frame;
};

vector<RGBDFile> RGBDFileInfo;

void readRGBDdata(const string &filename)
{

    string line, word;

    fstream fin;

    fin.open(filename.c_str());

    if(!fin)
    {
        cout<<"cannot open the file"<<endl;
    }
    else
    {
        cout<<"file is open"<<endl;
    }

    istringstream istr;

    string str;

    while(getline(fin, line))
    {
       RGBDFile fileInfo;
       istringstream record(line); //bind record to the line we just read
       record >> fileInfo.rgb_timestamp; //read the name
       record >> fileInfo.real_timestamp;
       record >> fileInfo.rgb_frame;
       record >> fileInfo.depth_frame;
       RGBDFileInfo.push_back(fileInfo);

    }

    fin.close();
}


cv::Mat readDepthImage(const char *fileName)
{
    cv::Mat depth = cv::imread(fileName, -1);
    assert(depth.type() == 2);
    return depth;
}

cv::Mat matToDouble(const cv::Mat& depthMap)
{
    int row = depthMap.rows;
    int col = depthMap.cols;

    cv::Mat ret = cv::Mat(row, col, CV_64F);

    assert(depthMap.channels() == 1);

    if(depthMap.type()==0)
    {
        //unsigned char
        for(int y = 0; y<row; y++)
        {
            for(int x=0; x<col; x++)
            {
                ret.at<double>(y,x) = depthMap.at<unsigned char>(y,x);
            }
        }
    }
    else if(depthMap.type() == 2)
    {
        //unsigned short
        for(int y = 0; y < row; y++)
        {
            for(int x =0; x<col; x++)
            {
                ret.at<double>(y,x) = depthMap.at<unsigned short>(y,x);
            }

        }
    }
    else
    {
        assert(0);
    }

    return ret;
}

cv::Mat matToUnsignedShort(const cv::Mat& depthMap)
{
    int row = depthMap.rows;
    int col = depthMap.cols;
    cv::Mat ret = cv::Mat(row, col, CV_16U);

    assert(depthMap.channels() == 1);

    if(depthMap.type() == 0)
    {
        //unsigned char
        for(int y = 0; y< row; y++)
        {
            for(int x = 0; x< col; x++)
            {
                ret.at<unsigned short>(y, x) = depthMap.at<unsigned char>(y,x);
            }

        }

    }
    else if(depthMap.type()==2)
    {
        ret = depthMap;
    }
    else
    {
        assert(0);
    }

    return ret;
}

void printCovariance(ofstream &printName,  const Eigen::MatrixXd &covariance)
{
    for(int i=0; i<covariance.rows(); i++)
    {
        for(int j=0; j<covariance.cols(); j++)
        {
             printName<<covariance(i,j)<<" ";
        }
    }
    printName<<endl;

}


void featureMatching(Mat img_1, Mat img_2, Mat depth_1, Mat depth_2, Mat K, int frame_index, Mat RVO, Mat tVO)
{

        vector<KeyPoint> keypoints_1, keypoints_2;

        //Camera Intrinsics
        double fx=K.at<double>(0,0);
        double fy=K.at<double>(1,1);
        double cx=K.at<double>(0,2);
        double cy=K.at<double>(1,2);

        double min_dis = 500;
        double max_dis = 50000;

        double X1, Y1, Z1, X2, Y2, Z2;
        double factor = 5000;

        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 400;

        SurfFeatureDetector detector( minHessian );


        detector.detect( img_1, keypoints_1 );
        detector.detect( img_2, keypoints_2 );
        //imshow( "Good Matches", img_matches );

        //-- Step 2: Calculate descriptors (feature vectors)
        SurfDescriptorExtractor extractor;

        Mat descriptors_1, descriptors_2;

        extractor.compute(img_1, keypoints_1, descriptors_1 );
        extractor.compute(img_2, keypoints_2, descriptors_2 );



        //-- Step 3: Matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
        std::vector< DMatch > matches;

        matcher.match( descriptors_1, descriptors_2, matches );

         // compute homography using RANSAC
        cv::Mat mask;
        int ransacThreshold=9;

        vector<Point2d> imgpts1beforeRANSAC, imgpts2beforeRANSAC;

        for( int i = 0; i < (int)matches.size(); i++ )
        {
            imgpts1beforeRANSAC.push_back(keypoints_1[matches[i].queryIdx].pt);
            imgpts2beforeRANSAC.push_back(keypoints_2[matches[i].trainIdx].pt);
        }

        cv::Mat H12 = cv::findHomography(imgpts1beforeRANSAC, imgpts2beforeRANSAC, CV_RANSAC, ransacThreshold, mask);

        int numMatchesbeforeRANSAC=(int)matches.size();
        cout<<"The number of matches before RANSAC"<<numMatchesbeforeRANSAC<<endl;

        int numRANSACInlier=0;
        for(int i=0; i<(int)matches.size(); i++)
        {
            if((int)mask.at<uchar>(i, 0) == 1)
            {
                numRANSACInlier+=1;
            }
        }

        cout<<"The number of matches after RANSAC"<<numRANSACInlier<<endl;

        double max_dist = 0; double min_dist = 100;

        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < descriptors_1.rows; i++ )
        {
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
        //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
        //-- small)
        //-- PS.- radiusMatch can also be used here.
        std::vector< DMatch > good_matches;

        for( int i = 0; i < descriptors_1.rows; i++)
        {
          if( matches[i].distance <= max(2*min_dist, 0.02)&& (int)mask.at<uchar>(i, 0) == 1)  //consider RANSAC
            { good_matches.push_back(matches[i]); }
        }


        vector<int> matchedKeypointsIndex1, matchedKeypointsIndex2;

        vector<Point2f> imgpts1, imgpts2;

        for( int i = 0; i < (int)good_matches.size(); i++ )
        {
           //printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
            matchedKeypointsIndex1.push_back(good_matches[i].queryIdx);
            matchedKeypointsIndex2.push_back(good_matches[i].trainIdx);
            imgpts1.push_back(Point2f(keypoints_1[good_matches[i].queryIdx].pt.x, keypoints_1[good_matches[i].queryIdx].pt.y));
            imgpts2.push_back(Point2f(keypoints_2[good_matches[i].trainIdx].pt.x, keypoints_2[good_matches[i].trainIdx].pt.y));
        }


        vector<Point3f>  feature_world3D_1;
       // project the matched 2D keypoint in image1 to the 3D points in world coordinate(the camera coordinate equals to the world coordinate in this case) using back-projection
        for(int i=0; i<(int)matchedKeypointsIndex1.size(); i++)
         {
            unsigned short depthValue1 = depth_1.at<unsigned short>(keypoints_1[matchedKeypointsIndex1[i]].pt.y, keypoints_1[matchedKeypointsIndex1[i]].pt.x);
            double worldZ1=0;
            if(depthValue1 > min_dis && depthValue1 < max_dis )
            {
               worldZ1=depthValue1/factor;
            }


            double worldX1=(keypoints_1[matchedKeypointsIndex1[i]].pt.x-cx)*worldZ1/fx;
            double worldY1=(keypoints_1[matchedKeypointsIndex1[i]].pt.y-cy)*worldZ1/fy;

            //cout<<i<<"th matchedKeypointsIndex1  "<<matchedKeypointsIndex1[i]<<"   worldX1  "<<worldX1<<"  worldY1   "<<worldY1<<"  worldZ1   "<<worldZ1<<endl;

            //store point cloud
            feature_world3D_1.push_back(Point3f(worldX1,worldY1,worldZ1));
        }

    {

        vector<float> reproj_errors;
        cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
        printf("3\n");
        cv::Mat rod;
        cv::Rodrigues(RVO, rod);
        CvDraw::draw_reprojection_error(feature_world3D_1, imgpts2, K, rod, tVO, distCoeffs, reproj_errors, img_2);
        printf("4\n");


        char fileName1[1024] = {NULL};
        sprintf(fileName1, "/home/lili/workspace/SLAM/evaluation/fr1_desk/trial3/reproj/reproj_%05d.png", frame_index);
        cv::imwrite(fileName1, img_2);
        printf("save image %s\n", fileName1);

   }
}


int main(int argc, char * argv[])
{
    int width = 640;
    int height = 480;

    Resolution::getInstance(width, height);

    //Intrinsics::getInstance(528, 528, 320, 240);

    //The Intrinsics for the RGBD Benchmark
    Intrinsics::getInstance(525,525,319.5,239.5);

    cv::Mat intrinsicMatrix = cv::Mat(3,3,CV_64F);

    intrinsicMatrix.at<double>(0,0) = Intrinsics::getInstance().fx();
    intrinsicMatrix.at<double>(1,1) = Intrinsics::getInstance().fy();

    intrinsicMatrix.at<double>(0,2) = Intrinsics::getInstance().cx();
    intrinsicMatrix.at<double>(1,2) = Intrinsics::getInstance().cy();

    intrinsicMatrix.at<double>(0,1) =0;
    intrinsicMatrix.at<double>(1,0) =0;

    intrinsicMatrix.at<double>(2,0) =0;
    intrinsicMatrix.at<double>(2,1) =0;
    intrinsicMatrix.at<double>(2,2) =1;

    readRGBDdata("/home/lili/workspace/SLAM/evaluation/fr1_desk/fr1deskTimeAssoData.txt");

    //Bytef * decompressionBuffer = new Bytef[Resolution::getInstance().numPixels() * 2];
    //IplImage * deCompImage = 0;

    //std::string logFile="/home/lili/Kinect_Logs/2015-11-05.00.klg";

   // assert(pcl::console::parse_argument(argc, argv, "-l", logFile) > 0 && "Please provide a log file");

    /*RawLogReader logReader(decompressionBuffer,
                           deCompImage,
                           logFile,
                           true);*/

    cv::Mat1b tmp(height, width);
    //cv::Mat3b depthImg(height, width);

    PlaceRecognition placeRecognition(&intrinsicMatrix);

    iSAMInterface isam;

    //Keyframes
    KeyframeMap map(true);
    Eigen::Vector3f lastPlaceRecognitionTrans = Eigen::Vector3f::Zero();
    Eigen::Matrix3f lastPlaceRecognitionRot = Eigen::Matrix3f::Identity();
    int64_t lastTime = 0;

    FOVISOdometry * odom = 0;

    cout<<"RGBDFileInfo.size() "<<RGBDFileInfo.size()<<endl;

    if(true)
    {
        int frame_index = 0;
        odom = new FOVISOdometry;

        if(frame_index != RGBDFileInfo.size())
        {
            frame_index++;

            cv::Mat rgb_frame=cv::imread(RGBDFileInfo[frame_index].rgb_frame, 1);
            cv::Mat depth_frame1=cv::imread(RGBDFileInfo[frame_index].depth_frame, -1);
           // cout<<"depth_frame1.type "<<depth_frame1.type()<<endl;
            assert(depth_frame1.type() == 2 || depth_frame1.type() == 0);
            cv::Mat depth_frame = matToUnsignedShort(depth_frame1);
          //  cout<<"depth_frame.type "<<depth_frame.type()<<endl;

            unsigned char *imageData = rgb_frame.data;
            unsigned short *depthData =(unsigned short*) depth_frame.data;

            //cout<<" depthData.size() is " <<depthData.length()<<endl;

            uint64_t timestamp = RGBDFileInfo[frame_index].rgb_timestamp;

            Eigen::Matrix3f Rcurr = Eigen::Matrix3f::Identity();
            Eigen::Vector3f tcurr = Eigen::Vector3f::Zero();

            cout<<"timestamp is "<<timestamp<<endl;

            odom->getIncrementalTransformation(tcurr,
                                               Rcurr,
                                               timestamp,
                                               imageData,
                                               depthData);

            cout<<"odom->getIncrementalTransformation is OK"<<endl;


        }
    }

    ofstream fout1("fr1desk_FOVISJan17Trial3.txt");
    ofstream fout2("fr1desk_KeyframeMotionMetric0.0Jan17Trial3.txt");
    ofstream fout3("fr1desk_closure_transformationJan17Trial3.txt");
    ofstream fout4("fr1desk_pose_after_optimizationJan17Trial3.txt");
    ofstream fout5("fr1desk_covarianceJan17Trial3.txt");



    //ofstream fout11("fr1xyz_pose_FOVISJan12Trial1.csv");
    //ofstream fout21("fr1xyz_pose_KeyframeMotionMetric0.3Jan13Trial1.csv");
    //ofstream fout31("fr1xyz_closure_transformationJan13Trial1.csv");
    //ofstream fout41("fr1xyz_pose_after_optimizationJan13Trial1.csv");

    int frame_index = 0;
    int loopClosureCount=0;
    int keyframe_count =0;

    Mat first_rgb;
    Mat first_depth;

    Mat R_abs = cv::Mat::eye(3,3, CV_64F);
    Mat T_abs = cv::Mat::zeros(3, 1, CV_64F);


    while(frame_index<RGBDFileInfo.size())
    {
        //logReader.getNext();

        //cv::Mat3b rgbImg(height, width, (cv::Vec<unsigned char, 3> *)logReader.deCompImage->imageData);
        //cv::Mat1w depth(height, width, (unsigned short *)&decompressionBuffer[0]);


        cout<<"frame_index is "<<frame_index<<endl;

        cv::Mat rgbImg=cv::imread(RGBDFileInfo[frame_index].rgb_frame, 1);
        cv::Mat depthImg1=cv::imread(RGBDFileInfo[frame_index].depth_frame, -1);
        assert(depthImg1.type() == 2 || depthImg1.type() == 0);
        cv::Mat depthImg = matToUnsignedShort(depthImg1);
        unsigned char *imageData = rgbImg.data;
        unsigned short *depthData = (unsigned short*)depthImg.data;

        if(frame_index == 0)
        {
            first_rgb = rgbImg;
            first_depth = depthImg;
        }

        cv::normalize(depthImg, tmp, 0, 255, cv::NORM_MINMAX, 0);

        cv::cvtColor(tmp, depthImg, CV_GRAY2RGB);

        cv::imshow("RGB", rgbImg);
        cv::imshow("Depth", depthImg);

        char key = cv::waitKey(1);

        if(key == 'q')
        {
            break;
        }
        else if(key == ' ')
        {
            key = cv::waitKey(0);
        }


        Eigen::Matrix3f Rcurr = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcurr = Eigen::Vector3f::Zero();

   //     Eigen::Matrix3f RVOprev = Eigen::Matrix3f::Identity();
   //     Eigen::Vector3f tVOprev = Eigen::Vector3f::Zero();


        uint64_t timestamp = RGBDFileInfo[frame_index].rgb_timestamp;

//        #1
        odom->getIncrementalTransformation(tcurr,
                                           Rcurr,
                                           timestamp,
                                           imageData,
                                           depthData);

     //   R_abs = Rcurr * R_abs;
    //    T_abs = T_abs + Rcurr;


        if(odom->getState())
        {
          Eigen::Quaternionf quatFrame(Rcurr);
          fout1<<RGBDFileInfo[frame_index].real_timestamp<<" "<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<quatFrame.w()<<" "<<quatFrame.x()<<" "<<quatFrame.y()<<" "<<quatFrame.z()<<endl;
          //fout11<<frame_index<<" "<<RGBDFileInfo[frame_index].real_timestamp<<" "<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<quatFrame.w()<<" "<<quatFrame.x()<<" "<<quatFrame.y()<<" "<<quatFrame.z()<<endl;

          if(frame_index != 0)
          {
            cv::Mat R_cv = cv::Mat(3, 3, CV_64F);
            cv::Mat T_cv = cv::Mat(3, 1, CV_64F);
            for(int i = 0; i<3; i++)
            {
                T_cv.at<double>(i, 0) = tcurr[i];
                for(int j = 0; j<3; j++)
                {
                    R_cv.at<double>(i, j) = Rcurr(i,j);
                }
            }

            R_abs = R_cv;
            T_abs = T_cv;
            printf("frame number %d, translation %f %f %f\n", frame_index, T_abs.at<double>(0, 0), T_abs.at<double>(1, 0), T_abs.at<double>(2, 0));

                printf("0\n");
                featureMatching(first_rgb,rgbImg,first_depth,depthImg, intrinsicMatrix, frame_index, R_abs, T_abs);
                printf("1\n");
          }


        frame_index++;
        if(frame_index>10)
        {
            break;
        }
    }

 }


    return 0;
}

