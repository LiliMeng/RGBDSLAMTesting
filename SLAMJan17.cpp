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

    cout<<"OdometryProvider is OK"<<endl;

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
                                               timestamp, // dumb parameter
                                               imageData,
                                               depthData);

            cout<<"odom->getIncrementalTransformation is OK"<<endl;

            {
                  // draw reprojection error
            cv::Mat image3D;

            KinectCamera camera = KinectCamera(&intrinsicMatrix);

            camera.computeImage3D(depth_frame, image3D);

            int rnd_num = 100;
            vector<Point3f> points3d;
            vector<Point2f> points2d;
            for(int i = 0; i<rnd_num; i++)
            {
                int x = rand()%640;
                int y = rand()%480;

                float * row3d = (float *)image3D.ptr(y);
                float x_3d = row3d[x * 3];
                float y_3d = row3d[x * 3 + 1];
                float z_3d = row3d[x * 3 + 2];
                if(x_3d < 500)
                {
                    points3d.push_back(Point3f(x_3d, y_3d, z_3d));
                    points2d.push_back(Point2f(x, y));
                }
            }

            /*
            void draw_reprojection_error(const vector<cv::Point3f> &pt_3d,
                                        const vector<cv::Point2f> &pt_2d,
                                        const cv::Mat & camera_intrinsic_matrix,
                                        const cv::Mat & rotation,
                                        const cv::Mat & translation,
                                        const cv::Mat & distCoeffs,
                                        vector<float> & reproj_errors,
                                        cv::Mat & error_image);
            */

             cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
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
            vector<float> reproj_errors;
            CvDraw::draw_reprojection_error(points3d,
                                            points2d,
                                            intrinsicMatrix,
                                            R_cv,
                                            T_cv,
                                            distCoeffs,
                                            reproj_errors,
                                            rgb_frame);
            cv::imshow("first frame", rgb_frame);
            cv::waitKey(0);

         //   int num = rand()%256;
         //   char buf[1024] = {NULL};
          //  sprintf(buf, "/home/lili/workspace/SLAM/evaluation/fr1_desk/trial2/PnP/reproj_%d.jpg", num);
          //  cv::imwrite(buf, curImage_);
          //  printf("save image %s\n", buf);

            }

        }
    }
    /*else
    {
        odom = new DVOdometry;

        if(logReader.hasMore())
        {
            logReader.getNext();

            DVOdometry * dvo = static_cast<DVOdometry *>(odom);

            dvo->firstRun((unsigned char *)logReader.deCompImage->imageData,
                          (unsigned short *)&decompressionBuffer[0]);
        }
    }*/

    ofstream fout1("fr1desk_FOVISJan17Trial2.txt");
    ofstream fout2("fr1desk_KeyframeMotionMetric0.0Jan17Trial2.txt");
    ofstream fout3("fr1desk_closure_transformationJan17Trial2.txt");
    ofstream fout4("fr1desk_pose_after_optimizationJan17Trial2.txt");
    ofstream fout5("fr1desk_covarianceJan17Trial2.txt");



    //ofstream fout11("fr1xyz_pose_FOVISJan12Trial1.csv");
    //ofstream fout21("fr1xyz_pose_KeyframeMotionMetric0.3Jan13Trial1.csv");
    //ofstream fout31("fr1xyz_closure_transformationJan13Trial1.csv");
    //ofstream fout41("fr1xyz_pose_after_optimizationJan13Trial1.csv");

    int frame_index = 0;
    int loopClosureCount=0;
    int keyframe_count =0;

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

        uint64_t timestamp = RGBDFileInfo[frame_index].rgb_timestamp;

//        #1
        odom->getIncrementalTransformation(tcurr,
                                           Rcurr,
                                           timestamp,
                                           imageData,
                                           depthData);

       // printf("odom finish frame %d\n", frame_index);

        if(odom->getState())
        {
          Eigen::Quaternionf quatFrame(Rcurr);
          fout1<<RGBDFileInfo[frame_index].real_timestamp<<" "<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<quatFrame.w()<<" "<<quatFrame.x()<<" "<<quatFrame.y()<<" "<<quatFrame.z()<<endl;
          //fout11<<frame_index<<" "<<RGBDFileInfo[frame_index].real_timestamp<<" "<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<quatFrame.w()<<" "<<quatFrame.x()<<" "<<quatFrame.y()<<" "<<quatFrame.z()<<endl;


        Eigen::Matrix3f Rdelta = Rcurr.inverse() * lastPlaceRecognitionRot;
        Eigen::Vector3f tdelta = tcurr - lastPlaceRecognitionTrans;

        Eigen::MatrixXd covariance = odom->getCovariance();

        //fout5<<covariance(0,0)<<endl;
        printCovariance(fout5,  covariance);


      //  if((Projection::rodrigues2(Rdelta).norm() + tdelta.norm())  >= 0.3)
        {

            isam.addCameraCameraConstraint(lastTime,
                                           timestamp,
                                           lastPlaceRecognitionRot,
                                           lastPlaceRecognitionTrans,
                                           Rcurr,
                                           tcurr,
                                           covariance);

            //lastTime = logReader.timestamp;
            lastTime=timestamp;

            lastPlaceRecognitionRot = Rcurr;
            lastPlaceRecognitionTrans = tcurr;

//          #2
            map.addKeyframe(imageData,
                            depthData,
                            Rcurr,
                            tcurr,
                            timestamp);

           Eigen::Quaternionf quatKeyframe(Rcurr);

           fout2<<RGBDFileInfo[frame_index].real_timestamp<<" "<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<quatKeyframe.w()<<" "<<quatKeyframe.x()<<" "<<quatKeyframe.y()<<" "<<quatKeyframe.z()<<endl;
          // fout21<<frame_index<<" "<<RGBDFileInfo[frame_index].real_timestamp<<" "<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<quatKeyframe.w()<<" "<<quatKeyframe.x()<<" "<<quatKeyframe.y()<<" "<<quatKeyframe.z()<<endl;
           /*
            //Save keyframe
           {
            cv::Mat3b rgbImgKeyframe(height, width, (cv::Vec<unsigned char, 3> *)logReader.deCompImage->imageData);

            cv::Mat1w depthImgKeyframe(height, width, (unsigned short *)&decompressionBuffer[0]);

            //save keyframe depth
            char fileName[1024] = {NULL};
            sprintf(fileName, "keyframe_depth_%06d.png", frame_index);
            cv::imwrite(fileName, depthImgKeyframe);

            //save keyframe rgb

            sprintf(fileName, "keyframe_rgb_%06d.png", frame_index);
            cv::imwrite(fileName, rgbImgKeyframe);
            frame_index ++;

           }
        */

            int64_t matchTime;
            Eigen::Matrix4d transformation;
            Eigen::MatrixXd cov=Eigen::MatrixXd::Identity(6,6);

            keyframe_count++;
            cout<<"map.addKeyframe is OK, the number of keyframe is  "<<keyframe_count<<endl;

         //#3
            if(placeRecognition.detectLoop(imageData,
                                           depthData,
                                           timestamp,
                                           matchTime,
                                           transformation,
                                           cov,
                                           loopClosureCount))
            {
               loopClosureCount++;

               isam.addLoopConstraint(timestamp, matchTime, transformation, cov);
               fout3<<frame_index<<" "<<RGBDFileInfo[frame_index].real_timestamp<<" "<<timestamp<<" "<<matchTime<<" "<<RGBDFileInfo[matchTime-1].real_timestamp<<" "<<transformation(0,0)<<" "<<transformation(0,1)<<" "<<transformation(0,2)<<" "<<transformation(0,3)<<" "<<transformation(1,0)<<" "<<transformation(1,1)<<" "<<transformation(1,2)<<" "<<transformation(1,3)<<" "<<transformation(2,0)<<" "<<transformation(2,1)<<" "<<transformation(2,2)<<" "<<transformation(2,3)<<" "<<transformation(3,0)<<" "<<transformation(3,1)<<" "<<transformation(3,2)<<" "<<transformation(3,3)<<endl;
              // fout31<<frame_index<<" "<<RGBDFileInfo[frame_index].real_timestamp<<" "<<timestamp<<" "<<matchTime<<" "<<RGBDFileInfo[matchTime-1].real_timestamp<<" "<<transformation(0,0)<<" "<<transformation(0,1)<<" "<<transformation(0,2)<<" "<<transformation(0,3)<<" "<<transformation(1,0)<<" "<<transformation(1,1)<<" "<<transformation(1,2)<<" "<<transformation(1,3)<<" "<<transformation(2,0)<<" "<<transformation(2,1)<<" "<<transformation(2,2)<<" "<<transformation(2,3)<<" "<<transformation(3,0)<<" "<<transformation(3,1)<<" "<<transformation(3,2)<<" "<<transformation(3,3)<<endl;

               cout<<matchTime<<endl;
               cout<<"loopClosureCount"<<loopClosureCount<<endl;
            }

        }

        }

        frame_index++;
        /*
        if(loopClosureCount>=5)
        {
            break;
        }*/

    }

    std::vector<std::pair<uint64_t, Eigen::Matrix4f> > posesBefore;
    isam.getCameraPoses(posesBefore);

    cout<<"The program is OK before optimization "<<endl;
//    #4
    isam.optimise();

    cout<<"The program is OK after isam.optimise() "<<endl;

   // map.applyPoses(isam);

   // pcl::PointCloud<pcl::PointXYZRGB> * cloud = map.getMap();

   // pcl::visualization::PCLVisualizer cloudViewer;

   // cloudViewer.setBackgroundColor(1, 1, 1);
   // cloudViewer.initCameraParameters();
  // cloudViewer.addCoordinateSystem(0.1, 0, 0, 0);

    //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> color(cloud->makeShared());
  //  cloudViewer.addPointCloud<pcl::PointXYZRGB>(cloud->makeShared(), color, "Cloud Viewer");

    std::vector<std::pair<uint64_t, Eigen::Matrix4f> > poses;

    isam.getCameraPoses(poses);

   cout<<"The number of optimized poses"<<poses.size()<<endl;

    for(std::vector<std::pair<uint64_t, Eigen::Matrix4f> >::iterator ite=poses.begin(); ite!=poses.end(); ite++)
    {
        Eigen::Matrix3f Roptimized;
        Roptimized<<ite->second(0,0), ite->second(0,1), ite->second(0,2),
                    ite->second(1,0), ite->second(1,1), ite->second(1,2),
                    ite->second(2,0), ite->second(2,1), ite->second(2,2);

         Eigen::Quaternionf quatOptimized(Roptimized);

         fout4<<ite->second(0,3)<<" "<<ite->second(1,3)<<" "<<ite->second(2,3)<<" "<<quatOptimized.w()<<" "<<quatOptimized.x()<<" "<<quatOptimized.y()<<" "<<quatOptimized.z()<<endl;

    }



    //drawPoses(poses, cloudViewer, 1.0, 0, 0);
    //drawPoses(posesBefore, cloudViewer, 0, 0, 1.0);

   // cloudViewer.spin();

    //delete [] decompressionBuffer;

    return 0;
}

