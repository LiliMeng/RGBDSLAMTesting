/*
 * FOVISOdometry.cpp
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#include "FOVISOdometry.h"

FOVISOdometry::FOVISOdometry()
{
    input_camera_params.cx = Intrinsics::getInstance().cx();
    input_camera_params.cy = Intrinsics::getInstance().cy();
    input_camera_params.fx = Intrinsics::getInstance().fx();
    input_camera_params.fy = Intrinsics::getInstance().fy();
    input_camera_params.height = Resolution::getInstance().rows();
    input_camera_params.width = Resolution::getInstance().cols();
    rect = new fovis::Rectification(input_camera_params);

    fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();

    odom = new fovis::VisualOdometry(rect, options);

    depthImg = new fovis::DepthImage(input_camera_params, Resolution::getInstance().cols(), Resolution::getInstance().rows());

    gray_buf = new uint8_t[Resolution::getInstance().numPixels()];
    depth_data = new float[Resolution::getInstance().numPixels()];
}

FOVISOdometry::~FOVISOdometry()
{
    delete odom;
    delete rect;
    delete depthImg;
    delete [] gray_buf;
    delete [] depth_data;
}

fovis::VisualOdometry * FOVISOdometry::getFovis()
{
    return odom;
}

void FOVISOdometry::reset()
{
    delete odom;

    fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();

    odom = new fovis::VisualOdometry(rect, options);
}

void FOVISOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                 Eigen::Matrix3f & rot,
                                                 uint64_t timestamp,
                                                 unsigned char * rgbImage,
                                                 unsigned short * depthData)
{
    for(int i = 0; i < input_camera_params.width * input_camera_params.height; i++)
    {
        uint16_t d = depthData[i];

        if(d != 0)
        {
            depth_data[i] = d * 1e-3;
        }
        else
        {
            depth_data[i] = NAN;
        }
    }

    unsigned char * rgb_data = rgbImage;

    for(int i = 0; i < input_camera_params.width * input_camera_params.height; i++)
    {
        gray_buf[i] = (int)round(0.2125 * rgb_data[2] +
                                 0.7154 * rgb_data[1] +
                                 0.0721 * rgb_data[0]);
        rgb_data++;
        rgb_data++;
        rgb_data++;
    }

    depthImg->setDepthImage(depth_data);

    odom->processFrame(gray_buf, depthImg);

    Eigen::Isometry3f fovisOdom = odom->getMotionEstimate().cast<float>();

    if(odom->getMotionEstimateStatus() == fovis::SUCCESS)
    {
        Eigen::Isometry3f current;
        current.setIdentity();
        current.rotate(Rprev);
        current.translation() = tprev;

        current = current * fovisOdom;

        trans = current.translation();
        rot = current.rotation();

        tprev = trans;
        Rprev = rot;
    }
}

Eigen::MatrixXd FOVISOdometry::getCovariance()
{
    return odom->getMotionEstimateCov();
}
