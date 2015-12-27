/*
 * OdometryProvider.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#ifndef ODOMETRYPROVIDER_H_
#define ODOMETRYPROVIDER_H_

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "../Utils/Resolution.h"
#include "../Utils/Intrinsics.h"

class OdometryProvider
{
    public:
        OdometryProvider()
        {
            Rprev = Eigen::Matrix3f::Identity();
            tprev = Eigen::Vector3f::Zero();
        }

        virtual ~OdometryProvider()
        {}

        virtual void getIncrementalTransformation(Eigen::Vector3f & trans,
                                                  Eigen::Matrix3f & rot,
                                                  uint64_t timestamp,
                                                  unsigned char * rgbImage,
                                                  unsigned short * depthData) =0;

        virtual Eigen::MatrixXd getCovariance() =0;

        virtual void reset() =0;

        inline static void computeProjectiveMatrix(const cv::Mat& ksi, cv::Mat& Rt)
        {
            CV_Assert(ksi.size() == cv::Size(1, 6) && ksi.type() == CV_64FC1);

            // for infinitesimal transformation
            Rt = cv::Mat::eye(4, 4, CV_64FC1);

            cv::Mat R = Rt(cv::Rect(0, 0, 3, 3));
            cv::Mat rvec = ksi.rowRange(3, 6);

            cv::Rodrigues(rvec, R); //rvec is input rotation matrix, R is output rotation matrix

            Rt.at<double>(0,3) = ksi.at<double>(0);
            Rt.at<double>(1,3) = ksi.at<double>(1);
            Rt.at<double>(2,3) = ksi.at<double>(2);
        }

    protected:
        Eigen::Matrix3f Rprev;
        Eigen::Vector3f tprev;

};

#endif /* ODOMETRYPROVIDER_H_ */
