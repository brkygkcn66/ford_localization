#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/search/flann_search.h>

#include <pcl/keypoints/iss_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

class ICPMatcher
{
    ros::NodeHandle nh_;
    ros::Subscriber target_cloud_sub;
    ros::Publisher keypoints_pub;

    std::string target_pc_topic;
    std::string transformed_pc_topic;

    using Point_T = pcl::PointXYZ;

    pcl::PointCloud<Point_T>::Ptr prev_cloud{new pcl::PointCloud<Point_T>};
    pcl::PointCloud<Point_T>::Ptr filtered_prev_cloud {new pcl::PointCloud<Point_T>};

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr prev_fpfhs{new pcl::PointCloud<pcl::FPFHSignature33>()};

    bool is_first{true};

    ros::Publisher odom_pub;
    tf::TransformBroadcaster odom_broadcaster;
    ros::Time current_time;

    Eigen::Matrix4d current_pose;

    pcl::ApproximateVoxelGrid<Point_T> voxel_filter;

    float leaf_size = 0.1;

public:
    ICPMatcher():nh_("~"){

        read_params();

        // Subscribe to the target point cloud topic
        target_cloud_sub = nh_.subscribe<sensor_msgs::PointCloud2>(
            target_pc_topic, 1, &ICPMatcher::pointCloudCallback, this);

        // Create a publisher for the transformed point cloud
        keypoints_pub = nh_.advertise<sensor_msgs::PointCloud2>(
        "keypoints", 1);

        odom_pub = nh_.advertise<nav_msgs::Odometry>("odom", 50);

        current_pose = Eigen::Matrix4d::Identity ();
    }
    
    void computeFPFH(pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs, pcl::PointCloud<Point_T>::Ptr cloud)
    {
        fpfhs->clear();
        // Compute surface normals for the downsampled cloud
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimationOMP<Point_T, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(cloud);
        pcl::search::KdTree<Point_T>::Ptr tree(new pcl::search::KdTree<Point_T>());
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setKSearch(3); 
        normal_estimator.setNumberOfThreads(6);
        normal_estimator.compute(*normals);

        // ROS_INFO_STREAM("cloud_normals size" << normals->size ());

        // FPFH özellik çıkarımını uygulayın
        
        pcl::FPFHEstimationOMP<Point_T, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
        fpfh_estimation.setInputCloud(cloud);
        fpfh_estimation.setInputNormals(normals);
        fpfh_estimation.setSearchMethod(tree);
        fpfh_estimation.setRadiusSearch(0.05); 
        fpfh_estimation.setNumberOfThreads(6);
        fpfh_estimation.compute(*fpfhs);

        // ROS_INFO_STREAM("fpfhs size" << fpfhs->size ());
    }
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& current_cloud_msg)
    {
        if(is_first)
        {
            
            pcl::fromROSMsg(*current_cloud_msg, *prev_cloud);
            
            voxel_filter.setInputCloud(prev_cloud);
            voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
            voxel_filter.filter(*filtered_prev_cloud);

            computeFPFH(prev_fpfhs, filtered_prev_cloud);

            is_first = false;
            // ROS_INFO("First PC Received!!!");
        }
        else
        {
            // Convert ROS PointCloud2 message to PCL point cloud
            pcl::PointCloud<Point_T>::Ptr current_cloud{new pcl::PointCloud<Point_T>};
            pcl::fromROSMsg(*current_cloud_msg, *current_cloud);

            // Apply voxel filter to downsample the point cloud
            pcl::PointCloud<Point_T>::Ptr filtered_current_cloud {new pcl::PointCloud<Point_T>};
            voxel_filter.setInputCloud(current_cloud);
            voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size); 
            voxel_filter.filter(*filtered_current_cloud);

            // landmark detection

            pcl::PointCloud<pcl::FPFHSignature33>::Ptr current_fpfhs{new pcl::PointCloud<pcl::FPFHSignature33>()};

            computeFPFH(current_fpfhs, filtered_current_cloud);

            ROS_INFO_STREAM("prev_fpfhs size    :" << prev_fpfhs->size ());
            ROS_INFO_STREAM("current_fpfhs size :" << current_fpfhs->size ());


            // Correspondence estimation
            pcl::CorrespondencesPtr correspondences{new pcl::Correspondences};

            pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> correspondence_estimation;
            correspondence_estimation.setInputSource(prev_fpfhs);
            correspondence_estimation.setInputTarget(current_fpfhs);
            correspondence_estimation.determineCorrespondences(*correspondences);

            ROS_INFO_STREAM("Correspondence completed!!!");

            // Transformation estimation
            pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> transform_estimation;
            Eigen::Matrix4f transformation_matrix;
            transform_estimation.estimateRigidTransformation(*filtered_prev_cloud, *filtered_current_cloud, *correspondences, transformation_matrix);

            print4x4Matrix (transformation_matrix.cast<double>());

            prev_fpfhs->clear();
            prev_fpfhs = current_fpfhs;



            // // Publish landmarks using ROS publishers
            // sensor_msgs::PointCloud2 landmarks_msg;
            // pcl::toROSMsg(*keypoints, landmarks_msg);
            // landmarks_msg.header.frame_id = "VLS128_right_lidar";

            // keypoints_pub.publish(landmarks_msg);

            // ROS_INFO("4");

            // // ICP registration
            // pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            // icp.setInputSource(current_cloud);
            // icp.setInputTarget(map_ptr);
            // icp.setMaximumIterations (iterations);
            // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_aligned(new pcl::PointCloud<pcl::PointXYZ>);
            // ROS_INFO("2!");
            // icp.align(*cloud_aligned);

            // ROS_INFO("3!");

            // if (icp.hasConverged())
            // {
            //     ROS_INFO("ICP converged with score: %f", icp.getFitnessScore());

            //     transformation_matrix = icp.getFinalTransformation ().cast<double>();

            //     lidar_odom_matrix = transformation_matrix;
            //     // print4x4Matrix (transformation_matrix);

            //     // Convert transformed PCL point cloud back to ROS PointCloud2 message
            //     sensor_msgs::PointCloud2 transformed_cloud_msg;
            //     pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            //     pcl::transformPointCloud(*current_cloud, *transformed_cloud, transformation_matrix);

            //     map_ptr->clear();
            //     *map_ptr = *cloud_aligned;

            //     pcl::toROSMsg(*map_ptr, transformed_cloud_msg);
            //     transformed_cloud_msg.header = current_cloud_msg->header;

            //     // Publish the transformed point cloud
            //     transformed_pub.publish(transformed_cloud_msg);

            //     // publish_odom(lidar_odom_matrix);
            // }
            // else
            // {
            //     ROS_ERROR_STREAM("ICP did not converge");
            // }
            // // replace source cloud with new cloud
            // // prev_cloud.swap(current_cloud);
        }
        
    }

    void publish_odom(const Eigen::Matrix4d &matrix)
    {
        current_time = ros::Time::now();
        // Extract the rotation matrix from the transformation
        Eigen::Matrix3d rotationMatrix = matrix.block<3, 3>(0, 0);

        // Convert the rotation matrix to a quaternion using Eigen
        Eigen::Quaterniond eigenQuaternion(rotationMatrix);

        // Convert Eigen quaternion to ROS tf2 quaternion
        geometry_msgs::Quaternion odom_quat;
        odom_quat.x = eigenQuaternion.x();
        odom_quat.y = eigenQuaternion.y();
        odom_quat.z = eigenQuaternion.z();
        odom_quat.w = eigenQuaternion.w();

        //first, we'll publish the transform over tf
        geometry_msgs::TransformStamped odom_trans;
        odom_trans.header.stamp = current_time;
        odom_trans.header.frame_id = "odom_icp";
        odom_trans.child_frame_id = "rear_axle";

        odom_trans.transform.translation.x = matrix (0, 3);
        odom_trans.transform.translation.y = matrix (1, 3);
        odom_trans.transform.translation.z = matrix (2, 3);
        odom_trans.transform.rotation = odom_quat;

        //send the transform
        odom_broadcaster.sendTransform(odom_trans);

        //next, we'll publish the odometry message over ROS
        nav_msgs::Odometry odom;
        odom.header.stamp = current_time;
        odom.header.frame_id = "odom_icp";

        //set the position
        odom.pose.pose.position.x = matrix (0, 3);
        odom.pose.pose.position.y = matrix (1, 3);
        odom.pose.pose.position.z = matrix (2, 3);
        odom.pose.pose.orientation = odom_quat;

        //set the velocity
        odom.child_frame_id = "rear_axle";
        odom.twist.twist.linear.x = 0;
        odom.twist.twist.linear.y = 0;
        odom.twist.twist.angular.z = 0;

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