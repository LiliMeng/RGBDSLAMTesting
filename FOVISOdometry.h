/*
 * FOVISOdometry.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#ifndef FOVISODOMETRY_H_
#define FOVISODOMETRY_H_

#include "OdometryProvider.h"
#include <fovis/fovis.hpp>

class FOVISOdometry : public OdometryProvider
{
    public:
        FOVISOdometry();

        virtual ~FOVISOdometry();

        void getIncrementalTransformation(Eigen::Vector3f & trans,
                                          Eigen::Matrix3f & rot,
                                          uint64_t timestamp,
                                          unsigned char * rgbImage,
                                          unsigned short * depthData);

        Eigen::MatrixXd getCovariance();

        void reset();

        fovis::VisualOdometry * getFovis();

    private:
        fovis::VisualOdometry * odom;
        fovis::DepthImage * depthImg;
        fovis::CameraIntrinsicsParameters input_camera_params;
        fovis::Rectification * rect;
        uint8_t * gray_buf;
        float * depth_data;
};

#endif /* FOVISODOMETRY_H_ */
