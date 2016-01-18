/*
 * Surf3DTools.h
 *
 *  Created on: 12 May 2012
 *      Author: thomas
 */

#ifndef SURF3DTOOLS_H_
#define SURF3DTOOLS_H_

#include "DBowInterfaceSurf.h"
#include <opencv2/opencv.hpp>
#include <sstream>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
//#include "opencv2/nonfree/nonfree.hpp"

#include "opencv2/calib3d/calib3d.hpp"

#if CV24
#include <opencv2/nonfree/features2d.hpp>
#endif
#include "KinectCamera.h"

using namespace std;

class InterestPoint
{
    public:
        InterestPoint(float X, float Y, float Z, float u, float v)
             : X(X), Y(Y), Z(Z), u(u), v(v)
            {}
            float X, Y, Z, u, v;
};


class Surf3DTools
{
    public:
        class Surf3DImage
        {
            public:
                Surf3DImage(std::vector<float> & imageDescriptor,
                            std::vector<cv::KeyPoint> & imageKeyPoints)
                 : descriptor(imageDescriptor),
                   keyPoints(imageKeyPoints)
                {}

                class PointCorrespondence
                {
                    public:
                        PointCorrespondence(CvPoint3D32f point3d,
                                            CvPoint2D32f coordIm)
                         : point3d(point3d),
                           coordIm(coordIm)
                        {}

                        CvPoint3D32f point3d;
                        CvPoint2D32f coordIm;
                };

                std::vector<float> & descriptor;
                std::vector<cv::KeyPoint> & keyPoints;
                std::vector<PointCorrespondence> pointCorrespondences;
        };

        static Surf3DImage * calculate3dPointsSURF(KinectCamera * kinectCamera,
                                                   cv::Mat * depthMap,
                                                   std::vector<float> & imageDescriptor,
                                                   std::vector<cv::KeyPoint> & imageKeyPoints)
        {
            Surf3DImage * newSurf3DImage = new Surf3DImage(imageDescriptor, imageKeyPoints);

            cv::Mat * image3D = new cv::Mat();

            kinectCamera->computeImage3D(*depthMap, *image3D);

            const double maxZ = 10.0;
            const double minZ = 0.3;

            for(int y = 0; y < image3D->rows; y++)
            {
                for(int x = 0; x < image3D->cols; x++)
                {
                    cv::Vec3f point = image3D->at<cv::Vec3f> (y, x);

                    //cout<<"fabs(point[2]-maxZ) "<<fabs(point[2]-maxZ)<<endl;
                    //cout<<"fabs(point[2]) "<<fabs(point[2])<<endl;

                    //if(fabs(point[2] - maxZ) < FLT_EPSILON || fabs(point[2]) > maxZ)
                    if(fabs(point[2])<minZ||fabs(point[2])>maxZ)
                    {
                        continue;
                    }
                    else
                    {
                        //cout<<"fabs(point[2]) "<<fabs(point[2])<<endl;
                        newSurf3DImage->pointCorrespondences.push_back(Surf3DImage::PointCorrespondence(cvPoint3D32f(point[0],
                                                                                                                     point[1],
                                                                                                                     point[2]),
                                                                                                        cvPoint2D32f(x, y)));
                    }
                }
            }

            delete image3D;

            return newSurf3DImage;
        }

        static void surfMatch3D(Surf3DImage * one,
                                Surf3DImage * two,
                                std::vector<std::vector<float> > & matches1,
                                std::vector<std::vector<float> > & matches2)
        {
            cv::FlannBasedMatcher matcher;

            std::vector< cv::DMatch > matchesRaw;

            cv::Mat descriptors_1 = cv::Mat(one->descriptor.size()/128, 128, CV_32FC1);
            memcpy(descriptors_1.data, one->descriptor.data(), one->descriptor.size()*sizeof(float));

            cv::Mat descriptors_2 = cv::Mat(two->descriptor.size()/128, 128, CV_32FC1);
            memcpy(descriptors_2.data, two->descriptor.data(), two->descriptor.size()*sizeof(float));

            matcher.match(descriptors_1, descriptors_2, matchesRaw);
            /*matcher.knnMatch(descriptors_1, descriptors_2, matchesRaw, 2); //find the 2 nearest neighbors


            //discard invalid results, basically we have to filter out the good matches. To do that we will be using Nearest Neighbor Distance Ratio.
            vector< DMatch > good_matches;
            float nndrRatio = 0.70f;
            good_matches.reserve(matchesRaw.size());

            for (size_t i = 0; i < matchesRaw.size(); ++i)
            {
                if (matchesRaw[i].size() < 2)
                    continue;

                const DMatch &m1 = matchesRaw[i][0];
                const DMatch &m2 = matchesRaw[i][1];

                if(m1.distance <= nndrRatio * m2.distance)
                    good_matches.push_back(m1);
            }
            */

           // cout<<"matchesRaw.size()  "<<matchesRaw.size()<<endl;

            // compute homography using RANSAC
            cv::Mat mask;
            int ransacThreshold=9;

            vector<cv::Point2d> imgpts1beforeRANSAC, imgpts2beforeRANSAC;

            for( int i = 0; i < (int)matchesRaw.size(); i++ )
            {
                imgpts1beforeRANSAC.push_back(one->keyPoints[matchesRaw[i].queryIdx].pt);
                imgpts2beforeRANSAC.push_back(two->keyPoints[matchesRaw[i].trainIdx].pt);
            }

            //cv::Mat H12 = cv::findHomography(imgpts1beforeRANSAC, imgpts2beforeRANSAC, CV_RANSAC, ransacThreshold, mask);
            cv::Mat F12= cv::findFundamentalMat(imgpts1beforeRANSAC, imgpts2beforeRANSAC, CV_FM_RANSAC, ransacThreshold, 0.99, mask);
            cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
            int numMatchesbeforeRANSAC=(int)matchesRaw.size();

         //   cout<<"The number of matches before RANSAC"<<numMatchesbeforeRANSAC<<endl;

            int numRANSACInlier=0;
            std::vector< cv::DMatch > matches;

            for(int i=0; i<(int)matchesRaw.size(); i++)
            {
                if((int)mask.at<uchar>(i, 0) == 1)
                {
                    numRANSACInlier+=1;
                    matches.push_back(matchesRaw[i]);
                }
            }

            cout<<"The number of matches after RANSAC"<<numRANSACInlier<<endl;


            matches1.resize(matches.size());
            matches2.resize(matches.size());

            cout<<" matches1.size() "<<matches1.size()<<"  matches2.size() "<<matches2.size()<<endl;
            assert(matches1.size()==matches2.size());

            cout<<"one->pointCorrespondences.size() "<<one->pointCorrespondences.size()<<endl;
           // cout<<"two->pointCorrespondences.size() "<<two->pointCorrespondences.size()<<endl;

           // cout<<" one->keyPoints.size() "<<one->keyPoints.size()<<endl;
          //  cout<<" two->keyPoints.size() "<<two->keyPoints.size()<<endl;


            for(unsigned int i = 0; i < matches.size(); i++)
            {
                matches1[i].resize(5);
                assert(matches[i].queryIdx<one->pointCorrespondences.size());
                assert(matches[i].queryIdx<one->keyPoints.size());
                matches1[i][0] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.x;
                matches1[i][1] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.y;
                matches1[i][2] = one->pointCorrespondences.at(matches[i].queryIdx).point3d.z;
                matches1[i][3] = one->keyPoints.at(matches[i].queryIdx).pt.x;
                matches1[i][4] = one->keyPoints.at(matches[i].queryIdx).pt.y;
                matches2[i].resize(5);
                assert(matches[i].trainIdx<two->pointCorrespondences.size());
                assert(matches[i].trainIdx<two->keyPoints.size());
                matches2[i][0] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.x;
                matches2[i][1] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.y;
                matches2[i][2] = two->pointCorrespondences.at(matches[i].trainIdx).point3d.z;
                matches2[i][3] = two->keyPoints[matches[i].trainIdx].pt.x;
                matches2[i][4] = two->keyPoints[matches[i].trainIdx].pt.y;
            }
        }


        static void displayMatches(cv::Mat * im1,
                                   cv::Mat * im2,
                                   std::vector<std::vector<float> > & matches1,
                                   std::vector<std::vector<float> > & matches2, isam::Pose3d pose,
                                   bool save = false,
                                   int loop_closure_count=0)
        {
            cv::namedWindow("Loop Closure", CV_WINDOW_AUTOSIZE);

            cv::Mat * full_image;

            cv::Size full_size(im1->cols*2, im1->rows);

            full_image = new cv::Mat(full_size, CV_8UC3);

            cv::Mat left = full_image->colRange(0, im1->cols);
            im1->copyTo(left);
            cv::Mat right = full_image->colRange(im1->cols,im1->cols*2);
            im2->copyTo(right);

            for(unsigned int i = 0; i < matches1.size(); i++)
            {
                cv::line(*full_image, cv::Point(matches1[i][3], matches1[i][4]), cv::Point(matches2[i][3] + im1->cols, matches2[i][4]), cv::Scalar(0, 0, 255), 1);
            }

            cv::imshow("Loop Closure", *full_image);

            //save keyframe depth
            char fileName1[1024] = {NULL};
            sprintf(fileName1, "/home/lili/workspace/SLAM/evaluation/fr1_desk/trial2/loopClosure/loopClosure_largeWithLoop_first_%05d.png", loop_closure_count);
            cv::imwrite(fileName1, *im1);

            char fileName2[1024] = {NULL};
            sprintf(fileName2, "/home/lili/workspace/SLAM/evaluation/fr1_desk/trial2/loopClosure/loopClosure_largeWithLoop_second_%05d.png", loop_closure_count);
            cv::imwrite(fileName2, *im2);

            char fileName3[1024] = {NULL};
            sprintf(fileName3, "/home/lili/workspace/SLAM/evaluation/fr1_desk/trial2/loopClosure/loopClosure_largeWithLoop_full_%05d.png", loop_closure_count);
            cv::imwrite(fileName3, *full_image);


            cvWaitKey(3);

            delete full_image;


        }



    private:
        Surf3DTools()
        {}
};

#endif /* SURF3DTOOLS_H_ */
