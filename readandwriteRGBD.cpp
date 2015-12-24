 //only part of read and write RGBD info.
 
 cv::Mat readDepthImage(const char *fileName)
{
    cv::Mat depth = cv::imread(fileName, -1);
    assert(depth.type() == 2);
    return depth;
}
 
 
 
    {   // save frame
            char fileName[1024] = {NULL};
            sprintf(fileName, "frame_depth_%06d.png", frame_index);
            cv::imwrite(fileName, depth);
            frame_index ++;
            // save frame
            char fileName[1024] = {NULL};
            sprintf(fileName, "frame_depth_%06d.png", frame_index);
            cv::imwrite(fileName, depth);
            frame_index ++;

            // read from saved image
            cv::Mat depth2 = readDepthImage(fileName);
            assert(depth.type() == depth2.type());

            for(int y = 0; y < depth.rows; y++)
            {
                for(int x = 0; x< depth.cols; x++)
                {
                    if(depth.at<unsigned short>(y, x) != depth2.at<unsigned short>(y, x))
                    {
                        printf("depth depth2 is %d %d\n", depth.at<unsigned short>(y, x), depth2.at<unsigned short>(y, x));
                    }
                    else
                    {
                        printf("read write ok!\n");
                    }
                }
            }
            break;

        }
            // read from saved image
            cv::Mat depth2 = cv::imread(fileName, -1);
            printf("depth  type is %d ", depth.type());
            printf("depth2 type is %d ", depth2.type());
            assert(depth.type() == depth2.type());



            for(int y = 0; y < depth.rows; y++)
            {
                for(int x = 0; x< depth.cols; x++)
                {
                    if(depth.at<unsigned short>(y, x) != depth2.at<unsigned short>(y, x))
                    {
                        printf("depth depth2 is %d %d\n", depth.at<unsigned short>(y, x), depth2.at<unsigned short>(y, x));
                    }
                    else
                    {
                        printf("read write ok!\n");
                    }
                }
            }
            break;

        }


        {   // save rgb frame
            char fileName[1024] = {NULL};
            sprintf(fileName, "frame_%06d.jpg", frame_index);
            cv::imwrite(fileName, rgbImg);
            frame_index ++;
        }
