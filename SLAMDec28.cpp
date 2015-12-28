#include "RawLogReader.h"
#include "KeyframeMap.h"
#include "PoseGraph/iSAMInterface.h"
#include "PlaceRecognition/PlaceRecognition.h"
#include "Odometry/FOVISOdometry.h"
#include "Odometry/DVOdometry.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

using namespace std;


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

    cout<<"row "<<row<<endl;
    cout<<"col "<<col<<endl;

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


int main(int argc, char * argv[])
{
    int width = 640;
    int height = 480;

    Resolution::getInstance(width, height);

    Intrinsics::getInstance(528, 528, 320, 240);

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

    readRGBDdata("/home/lili/workspace/SLAM/src/outputAssociatedData.txt");

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

    OdometryProvider * odom = 0;

    cout<<"OdometryProvider is OK"<<endl;
    cout<<"RGBDFileInfo.size() "<<RGBDFileInfo.size()<<endl;

    if(true)
    {
        int frame_index = 0;
        odom = new FOVISOdometry;

        if(frame_index!=RGBDFileInfo.size())
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

            // test
            {
                for(int i = 0; i<height; i++)
                {
                    for(int j = 0; j<width; j++)
                    {
                        int  tmp = depthData[i*width + j];
                        printf("%d\n", tmp);
                    }
                }
            }

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

    ofstream fout1("camera_pose_FOVISDec27.csv");
    ofstream fout2("camera_pose_KeyframeMotionMetric0.1Dec27.csv");
    ofstream fout3("loop_closure_transformationDec27.csv");
    ofstream fout4("camera_pose_after_optimizationDec27.csv");


    int frame_index = 0;
    int loopClosureCount=0;

    while(frame_index<RGBDFileInfo.size())
    {
        //logReader.getNext();

        //cv::Mat3b rgbImg(height, width, (cv::Vec<unsigned char, 3> *)logReader.deCompImage->imageData);
        //cv::Mat1w depth(height, width, (unsigned short *)&decompressionBuffer[0]);

        frame_index++;

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


       fout1<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<Rcurr(0,0)<<" "<<Rcurr(0,1)<<" "<<Rcurr(0,2)<<" "<<Rcurr(1,0)<<" "<<Rcurr(1,1)<<" "<<Rcurr(1,2)<<" "<<Rcurr(2,0)<<" "<<Rcurr(2,1)<<" "<<Rcurr(2,2)<<endl;

        Eigen::Matrix3f Rdelta = Rcurr.inverse() * lastPlaceRecognitionRot;
        Eigen::Vector3f tdelta = tcurr - lastPlaceRecognitionTrans;

        Eigen::MatrixXd covariance = odom->getCovariance();

        if((Projection::rodrigues2(Rdelta).norm() + tdelta.norm())  >= 0.1)
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

//            #2
            map.addKeyframe(imageData,
                            depthData,
                            Rcurr,
                            tcurr,
                            timestamp);

           fout2<<tcurr[0]<<" "<<tcurr[1]<<" "<<tcurr[2]<<" "<<Rcurr(0,0)<<" "<<Rcurr(0,1)<<" "<<Rcurr(0,2)<<" "<<Rcurr(1,0)<<" "<<Rcurr(1,1)<<" "<<Rcurr(1,2)<<" "<<Rcurr(2,0)<<" "<<Rcurr(2,1)<<" "<<Rcurr(2,2)<<endl;

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
            Eigen::MatrixXd cov(6, 6);

            cout<<"map.addKeyframe is OK"<<endl;

//            #3
            if(placeRecognition.detectLoop(imageData,
                                           depthData,
                                           timestamp,
                                           matchTime,
                                           transformation,
                                           cov))
            {
               isam.addLoopConstraint(timestamp, matchTime, transformation, cov);
               fout3<<transformation(0,0)<<" "<<transformation(0,1)<<" "<<transformation(0,2)<<" "<<transformation(0,3)<<" "<<transformation(1,0)<<" "<<transformation(1,1)<<" "<<transformation(1,2)<<" "<<transformation(1,3)<<" "<<transformation(2,0)<<" "<<transformation(2,1)<<" "<<transformation(2,2)<<" "<<transformation(2,3)<<" "<<transformation(3,0)<<" "<<transformation(3,1)<<" "<<transformation(3,2)<<" "<<transformation(3,3)<<endl;
               cout<<matchTime<<endl;
               loopClosureCount++;
            }

        }

        if(loopClosureCount>=5)
        {
            break;
        }
    }

    std::vector<std::pair<uint64_t, Eigen::Matrix4f> > posesBefore;
    isam.getCameraPoses(posesBefore);

//    #4
    isam.optimise();


    map.applyPoses(isam);

    pcl::PointCloud<pcl::PointXYZRGB> * cloud = map.getMap();

    pcl::visualization::PCLVisualizer cloudViewer;

    cloudViewer.setBackgroundColor(1, 1, 1);
    cloudViewer.initCameraParameters();
   cloudViewer.addCoordinateSystem(0.1, 0, 0, 0);

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> color(cloud->makeShared());
    cloudViewer.addPointCloud<pcl::PointXYZRGB>(cloud->makeShared(), color, "Cloud Viewer");

    std::vector<std::pair<uint64_t, Eigen::Matrix4f> > poses;

    isam.getCameraPoses(poses);

    for(std::vector<std::pair<uint64_t, Eigen::Matrix4f> >::iterator ite=poses.begin(); ite!=poses.end(); ite++)
    {

        fout4<<ite->second(0,0)<<" "<<ite->second(0,1)<<" "<<ite->second(0,2)<<" "<<ite->second(0,3)<<" "<<ite->second(1,0)<<" "<<ite->second(1,1)<<" "<<ite->second(1,2)<<" "<<ite->second(1,3)<<" "<<ite->second(2,0)<<" "<<ite->second(2,1)<<" "<<ite->second(2,2)<<" "<<ite->second(2,3)<<" "<<ite->second(3,0)<<" "<<ite->second(3,1)<<" "<<ite->second(3,2)<<" "<<ite->second(3,3)<<endl;
    }

    cout<<"The number of optimized poses"<<poses.size()<<endl;


    drawPoses(poses, cloudViewer, 1.0, 0, 0);
    //drawPoses(posesBefore, cloudViewer, 0, 0, 1.0);

    cloudViewer.spin();

   //delete [] decompressionBuffer;

    return 0;
}

