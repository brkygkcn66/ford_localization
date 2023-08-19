#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <iostream>
  
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

class ICPMatcher
{
    ros::NodeHandle nh_;
    ros::Subscriber target_cloud_sub;
    ros::Publisher transformed_pub;

    std::string target_pc_topic;
    std::string transformed_pc_topic;

    int ndt_max_iterations{35};
    float leaf_size = 0.1;

    using Point_T = pcl::PointXYZ;


    pcl::PointCloud<Point_T>::Ptr prev_cloud{new pcl::PointCloud<Point_T>};
    pcl::ApproximateVoxelGrid<Point_T> approximate_voxel_filter;

    bool is_first{true};

    
    Eigen::Matrix4d lidar_odom_matrix = Eigen::Matrix4d::Identity ();

    ros::Publisher odom_pub;
    tf::TransformBroadcaster odom_broadcaster;
    ros::Time current_time;

public:
    ICPMatcher():nh_("~"){

        read_params();

        // Subscribe to the target point cloud topic
        target_cloud_sub = nh_.subscribe<sensor_msgs::PointCloud2>(
            target_pc_topic, 1, &ICPMatcher::pointCloudCallback, this);

        // Create a publisher for the transformed point cloud
        transformed_pub = nh_.advertise<sensor_msgs::PointCloud2>(
        transformed_pc_topic, 1);

        odom_pub = nh_.advertise<nav_msgs::Odometry>("odom_ndt", 50);

        
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& current_cloud_msg)
    {
        if(is_first)
        {
            prev_cloud->clear();
            pcl::fromROSMsg(*current_cloud_msg, *prev_cloud);
            is_first = false;
        }
        else
        {
            // Convert ROS PointCloud2 message to PCL point cloud
            pcl::PointCloud<Point_T>::Ptr current_cloud(new pcl::PointCloud<Point_T>);
            pcl::fromROSMsg(*current_cloud_msg, *current_cloud);

            // Filtering input scan to increase speed of registration.
            pcl::PointCloud<Point_T>::Ptr filtered_prev_cloud (new pcl::PointCloud<Point_T>);
            pcl::PointCloud<Point_T>::Ptr filtered_current_cloud (new pcl::PointCloud<Point_T>);
            
            approximate_voxel_filter.setLeafSize (leaf_size, leaf_size, leaf_size);
            
            approximate_voxel_filter.setInputCloud (prev_cloud);
            approximate_voxel_filter.filter (*filtered_prev_cloud);

            approximate_voxel_filter.setInputCloud (current_cloud);
            approximate_voxel_filter.filter (*filtered_current_cloud);


            // Initializing Normal Distributions Transform (NDT).
            pcl::NormalDistributionsTransform<Point_T, Point_T> ndt;

            // Setting scale dependent NDT parameters
            // Setting minimum transformation difference for termination condition.
            ndt.setTransformationEpsilon (0.1);
            // Setting maximum step size for More-Thuente line search.
            ndt.setStepSize (0.4);
            //Setting Resolution of NDT grid structure (VoxelGridCovariance).
            ndt.setResolution (5.0);

            // Setting max number of registration iterations.
            ndt.setMaximumIterations (ndt_max_iterations);

            // ndt.setNumThreads(4);

            // Setting point cloud to be aligned.
            ndt.setInputSource (filtered_current_cloud);
            // Setting point cloud to be aligned to.
            ndt.setInputTarget (filtered_prev_cloud);

            // Calculating required rigid transform to align the input cloud to the target cloud.
            pcl::PointCloud<Point_T>::Ptr output_cloud (new pcl::PointCloud<Point_T>);
            ndt.align (*output_cloud);

            if (ndt.hasConverged())
            {
                ROS_INFO("ICP converged with score: %f", ndt.getFitnessScore());

                Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();

                transformation_matrix = ndt.getFinalTransformation ().cast<double>();

                // lidar_odom_matrix = transformation_matrix.inverse() * lidar_odom_matrix;
                print4x4Matrix (transformation_matrix);

                // Convert transformed PCL point cloud back to ROS PointCloud2 message
                sensor_msgs::PointCloud2 transformed_cloud_msg;
                pcl::PointCloud<Point_T>::Ptr transformed_cloud(new pcl::PointCloud<Point_T>);
                pcl::transformPointCloud(*prev_cloud, *transformed_cloud, transformation_matrix);
                pcl::toROSMsg(*transformed_cloud, transformed_cloud_msg);
                transformed_cloud_msg.header = current_cloud_msg->header;

                // Publish the transformed point cloud
                transformed_pub.publish(transformed_cloud_msg);

                publish_odom(transformation_matrix);
            }
            else
            {
                ROS_ERROR_STREAM("ICP did not converge");
            }
            // replace previous cloud with current cloud
            prev_cloud.swap(current_cloud);
        }
        
    }

    void publish_odom(const Eigen::Matrix4d &matrix)
    {
        // Extract the rotation matrix from the transformation
        Eigen::Matrix3d rotationMatrix = matrix.block<3, 3>(0, 0);

        // Convert the rotation matrix to a quaternion using Eigen
        Eigen::Quaterniond eigenQuaternion(rotationMatrix);

        // Convert Eigen quaternion to ROS quaternion
        geometry_msgs::Quaternion odom_quat;
        odom_quat.x = eigenQuaternion.x();
        odom_quat.y = eigenQuaternion.y();
        odom_quat.z = eigenQuaternion.z();
        odom_quat.w = eigenQuaternion.w();

        //next, we'll publish the odometry message over ROS
        nav_msgs::Odometry odom;
        odom.header.stamp = ros::Time::now();
        odom.header.frame_id = "VLS128_right_lidar";

        //set the position
        odom.pose.pose.position.x = matrix (0, 3);
        odom.pose.pose.position.y = matrix (1, 3);
        odom.pose.pose.position.z = matrix (2, 3);
        odom.pose.pose.orientation = odom_quat;

        //publish the message
        odom_pub.publish(odom);
    }

    void print4x4Matrix (const Eigen::Matrix4d & matrix) const
    {
        printf ("Rotation matrix :\n");
        printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
        printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
        printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
        printf ("Translation vector :\n");
        printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
    }

    void read_params()
    {
        if (!nh_.getParam("target_pc_topic", target_pc_topic))
        {
            ROS_ERROR_STREAM("Failed to get parameter target_pc_topic");
        }
        ROS_INFO_STREAM("target_pc_topic name: " << target_pc_topic);

        if (!nh_.getParam("transformed_pc_topic", transformed_pc_topic))
        {
            ROS_ERROR_STREAM("Failed to get parameter transformed_pc_topic");
        }
        ROS_INFO_STREAM("transformed_pc_topic name: " << transformed_pc_topic);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "icp_matcher_node");
    
    ICPMatcher matcher;

    ros::spin();

    return 0;
}