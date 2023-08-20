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

#include <pcl/keypoints/sift_keypoint.h>

#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <cmath>

using Point_T = pcl::PointXYZ;

// working incrementally
class GPSOdometryEstimator 
{

public:
    GPSOdometryEstimator() {
        ros::NodeHandle nh;
        
        gps_sub = nh.subscribe("/gps/fix", 10, &GPSOdometryEstimator::gpsCallback, this);
        
        previous_latitude = 0.0;
        previous_longitude = 0.0;
    }

    void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& msg) {

        if(is_first)
        {
            previous_latitude = msg->latitude;
            previous_longitude = msg->longitude;
            is_first = false;
        }
        else
        {
            double current_latitude = msg->latitude;
            double current_longitude = msg->longitude;
            
            diff_displacement = haversineDistance(previous_latitude, previous_longitude, current_latitude, current_longitude);
            
            totol_displacement += diff_displacement;
            ROS_DEBUG("Displacement: %.2f meters", totol_displacement);
            
            previous_latitude = current_latitude;
            previous_longitude = current_longitude;
        }
        
    }

    double haversineDistance(double lat1, double lon1, double lat2, double lon2) {
        const double R = 6371000.0; // Earth's radius in meters
        
        double dlat = degToRad(lat2 - lat1);
        double dlon = degToRad(lon2 - lon1);
        
        double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
                   std::cos(degToRad(lat1)) * std::cos(degToRad(lat2)) *
                   std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
        
        double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
        
        return R * c;
    }

    inline double degToRad(double deg) {
        return deg * (M_PI / 180.0);
    }

    double get_total_displacement() const
    {
        return totol_displacement;
    }

    double get_diff_displacement() const
    {
        return diff_displacement;
    }

private:
    ros::Subscriber gps_sub;
    double previous_latitude;
    double previous_longitude;
    bool is_first{true};
    double totol_displacement{0};
    double diff_displacement{0};
};

class ICPMatcher
{
    ros::NodeHandle nh_;
    ros::Subscriber target_cloud_sub;
    ros::Publisher transformed_pub;

    std::string target_pc_topic;
    std::string transformed_pc_topic;

    pcl::PointCloud<Point_T>::Ptr prev_cloud{new pcl::PointCloud<Point_T>};
    pcl::PointCloud<Point_T>::Ptr filtered_prev_cloud {new pcl::PointCloud<Point_T>};

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr prev_fpfhs{new pcl::PointCloud<pcl::FPFHSignature33>()};

    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_previous{new pcl::PointCloud<pcl::PointWithScale>};

    bool is_first{true};

    ros::Publisher odom_pub;
    tf::TransformBroadcaster odom_broadcaster;
    ros::Time current_time;

    initial_position_ = Eigen::Vector3f(0, 0, 0);
    initial_orientation_ = Eigen::Quaternionf(1, 0, 0, 0);

    pcl::ApproximateVoxelGrid<Point_T> voxel_filter;

    float leaf_size = 0.25;

    GPSOdometryEstimator gps_odometry_calculator;

    std::string odom_frame_ = "odom_brky";
    std::string child_frame_ = "rear_axle";

public:
    ICPMatcher():nh_("~"){

        read_params();

        // Subscribe to the target point cloud topic
        target_cloud_sub = nh_.subscribe<sensor_msgs::PointCloud2>(
            target_pc_topic, 1, &ICPMatcher::pointCloudCallback, this);

        // Create a publisher for the transformed point cloud
        transformed_pub = nh_.advertise<sensor_msgs::PointCloud2>(
        "transformed_pc", 1);

        odom_pub = nh_.advertise<nav_msgs::Odometry>("odom", 50);

        current_pose = Eigen::Matrix4d::Identity ();
    }

    // This function by Tommaso Cavallari and Federico Tombari, taken from the tutorial
    // http://pointclouds.org/documentation/tutorials/correspondence_grouping.php
    double
    computeCloudResolution(const pcl::PointCloud<Point_T>::ConstPtr& cloud)
    {
        double resolution = 0.0;
        int numberOfPoints = 0;
        int nres;
        std::vector<int> indices(2);
        std::vector<float> squaredDistances(2);
        pcl::search::KdTree<Point_T> tree;
        tree.setInputCloud(cloud);
    
        for (size_t i = 0; i < cloud->size(); ++i)
        {
            if (! std::isfinite((*cloud)[i].x))
                continue;
    
            // Considering the second neighbor since the first is the point itself.
            nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
            if (nres == 2)
            {
                resolution += sqrt(squaredDistances[1]);
                ++numberOfPoints;
            }
        }
        if (numberOfPoints != 0)
            resolution /= numberOfPoints;
    
        return resolution;
    }
    
    void computeSIFT(pcl::PointCloud<Point_T>::Ptr cloud_xyz, pcl::PointCloud<pcl::PointWithScale> &keypoints)
    {
          // Parameters for sift computation
        constexpr float min_scale = 0.05f;
        constexpr int n_octaves = 3;
        constexpr int n_scales_per_octave = 4;
        constexpr float min_contrast = 0.001f;
        
        // Estimate the normals of the cloud_xyz
        pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());

        ne.setInputCloud(cloud_xyz);
        ne.setSearchMethod(tree_n);
        ne.setRadiusSearch(0.2);
        ne.compute(*cloud_normals);

        // Copy the xyz info from cloud_xyz and add it to cloud_normals as the xyz field in PointNormals estimation is zero
        for(std::size_t i = 0; i<cloud_normals->size(); ++i)
        {
            (*cloud_normals)[i].x = (*cloud_xyz)[i].x;
            (*cloud_normals)[i].y = (*cloud_xyz)[i].y;
            (*cloud_normals)[i].z = (*cloud_xyz)[i].z;
        }

        // Estimate the sift interest points using normals values from xyz as the Intensity variants
        pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
        
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
        sift.setSearchMethod(tree);
        sift.setScales(min_scale, n_octaves, n_scales_per_octave);
        sift.setMinimumContrast(min_contrast);
        sift.setInputCloud(cloud_normals);
        sift.compute(keypoints);

        std::cout << "No of SIFT points in the result are " << keypoints.size () << std::endl;
     
    }
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& current_cloud_msg)
    {
        if(is_first)
        {
            
            pcl::fromROSMsg(*current_cloud_msg, *prev_cloud);
            
            voxel_filter.setInputCloud(prev_cloud);
            voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
            voxel_filter.filter(*filtered_prev_cloud);
            keypoints_previous->clear();
            computeSIFT(filtered_prev_cloud, *keypoints_previous);

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

            pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_current{new pcl::PointCloud<pcl::PointWithScale>};

            computeSIFT(filtered_current_cloud, *keypoints_current);

            ROS_INFO_STREAM("keypoints_previous size    :" << keypoints_previous->size ());
            ROS_INFO_STREAM("keypoints_current size :" << keypoints_current->size ());

            pcl::IterativeClosestPoint<pcl::PointWithScale, pcl::PointWithScale> icp;
            icp.setInputSource(keypoints_previous);
            icp.setInputTarget(keypoints_current);

            pcl::PointCloud<pcl::PointWithScale> aligned_keypoints;
            icp.align(aligned_keypoints);

            

            if(icp.hasConverged())
            {
                Eigen::Matrix4f transformation_matrix = icp.getFinalTransformation();
                print4x4Matrix (transformation_matrix.cast<double>());

                // Apply transformation to cloud1
                pcl::PointCloud<Point_T>::Ptr transformed_cloud(new pcl::PointCloud<Point_T>);
                pcl::transformPointCloud(*current_cloud, *transformed_cloud, transformation_matrix);
                sensor_msgs::PointCloud2 transformed_cloud_msg;
                pcl::toROSMsg(*transformed_cloud, transformed_cloud_msg);
                transformed_cloud_msg.header = current_cloud_msg->header;

                // Publish the transformed point cloud
                transformed_pub.publish(transformed_cloud_msg);
            }


            keypoints_previous->clear();
            keypoints_previous = keypoints_current;

            // replace previous cloud with new cloud
            filtered_prev_cloud->clear();
            filtered_prev_cloud = filtered_current_cloud;

            prev_cloud->clear();
            prev_cloud = current_cloud;

            // // Correspondence estimation
            // pcl::CorrespondencesPtr correspondences{new pcl::Correspondences};

            // pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> correspondence_estimation;
            // correspondence_estimation.setInputSource(current_fpfhs);
            // correspondence_estimation.setInputTarget(prev_fpfhs);
            // correspondence_estimation.determineCorrespondences(*correspondences);

            // ROS_INFO_STREAM("Correspondence completed!!!");

            // // Transformation estimation
            // pcl::registration::TransformationEstimationSVD<Point_T, Point_T> transform_estimation;
            // Eigen::Matrix4f transformation_matrix;
            // transform_estimation.estimateRigidTransformation(*filtered_current_cloud, *filtered_prev_cloud, *correspondences, transformation_matrix);

            // print4x4Matrix (transformation_matrix.cast<double>());

            // // Apply transformation to cloud1
            // pcl::PointCloud<Point_T>::Ptr transformed_cloud(new pcl::PointCloud<Point_T>);
            // // pcl::transformPointCloud(*current_cloud, *transformed_cloud, transformation_matrix);

            // // Optionally, refine the alignment using ICP
            // pcl::IterativeClosestPoint<Point_T, Point_T> icp;
            // icp.setInputSource(filtered_current_cloud);
            // icp.setInputTarget(filtered_prev_cloud);
            // icp.align(*transformed_cloud, transformation_matrix);

            // // Print the transformation matrix and other results
            // // std::cout << "Transformation Matrix:\n" << transformation_matrix << std::endl;
            // std::cout << "ICP convergence: " << icp.hasConverged() << ", Score: " << icp.getFitnessScore() << std::endl;

            // if(icp.hasConverged())
            // {
            //     sensor_msgs::PointCloud2 transformed_cloud_msg;
            //     pcl::toROSMsg(*transformed_cloud, transformed_cloud_msg);
            //     transformed_cloud_msg.header = current_cloud_msg->header;

            //     // Publish the transformed point cloud
            //     transformed_pub.publish(transformed_cloud_msg);
            // }
            

            // prev_fpfhs->clear();
            // prev_fpfhs = current_fpfhs;

            // // replace previous cloud with new cloud
            // filtered_prev_cloud->clear();
            // filtered_prev_cloud = filtered_current_cloud;

            // prev_cloud->clear();
            // prev_cloud = current_cloud;


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

//     void publish_odom(const Eigen::Matrix4d &matrix)
//     {
//         current_time = ros::Time::now();
//         // Extract the rotation matrix from the transformation
//         Eigen::Matrix3d rotationMatrix = matrix.block<3, 3>(0, 0);

//         // Convert the rotation matrix to a quaternion using Eigen
//         Eigen::Quaterniond eigenQuaternion(rotationMatrix);

//         // Convert Eigen quaternion to ROS tf2 quaternion
//         geometry_msgs::Quaternion odom_quat;
//         odom_quat.x = eigenQuaternion.x();
//         odom_quat.y = eigenQuaternion.y();
//         odom_quat.z = eigenQuaternion.z();
//         odom_quat.w = eigenQuaternion.w();

//         //first, we'll publish the transform over tf
//         geometry_msgs::TransformStamped odom_trans;
//         odom_trans.header.stamp = current_time;
//         odom_trans.header.frame_id = "odom_icp";
//         odom_trans.child_frame_id = "rear_axle";

//         odom_trans.transform.translation.x = matrix (0, 3);
//         odom_trans.transform.translation.y = matrix (1, 3);
//         odom_trans.transform.translation.z = matrix (2, 3);
//         odom_trans.transform.rotation = odom_quat;

//         //send the transform
//         odom_broadcaster.sendTransform(odom_trans);

//         //next, we'll publish the odometry message over ROS
//         nav_msgs::Odometry odom;
//         odom.header.stamp = current_time;
//         odom.header.frame_id = "odom_icp";

//         //set the position
//         odom.pose.pose.position.x = matrix (0, 3);
//         odom.pose.pose.position.y = matrix (1, 3);
//         odom.pose.pose.position.z = matrix (2, 3);
//         odom.pose.pose.orientation = odom_quat;

//         //set the velocity
//         odom.child_frame_id = "rear_axle";
//         odom.twist.twist.linear.x = 0;
//         odom.twist.twist.linear.y = 0;
//         odom.twist.twist.angular.z = 0;

//         //publish the message
//         odom_pub.publish(odom);

//     }

//     void print4x4Matrix (const Eigen::Matrix4d & matrix) const
//     {
//         printf ("Rotation matrix :\n");
//         printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
//         printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
//         printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
//         printf ("Translation vector :\n");
//         printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
//     }

//     void read_params()
//     {
//         if (!nh_.getParam("target_pc_topic", target_pc_topic))
//         {
//             ROS_ERROR_STREAM("Failed to get parameter target_pc_topic");
//         }
//         ROS_INFO_STREAM("target_pc_topic name: " << target_pc_topic);

//         if (!nh_.getParam("transformed_pc_topic", transformed_pc_topic))
//         {
//             ROS_ERROR_STREAM("Failed to get parameter transformed_pc_topic");
//         }
//         ROS_INFO_STREAM("transformed_pc_topic name: " << transformed_pc_topic);
//     }
// };

class OdometryEstimator
{
    
    ros::NodeHandle nh;

    ICPMatcher icp_matcher;

    // FRAMES
    std::string frame_id_ = "odom_brky";
    std::string child_frame_id_ = "rear_axle";

    // TF
    geometry_msgs::TransformStamped odom_trans;
    tf2_ros::TransformBroadcaster tf_br;

    // ODOM
    ros::Publisher odom_pub;
    nav_msgs::Odometry odom;
    
public:
    OdometryEstimator()
    {   
        odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 50);
        initOdomTransaction();
    }

    void initOdomTransaction()
    {
        geometry_msgs::Quaternion odom_quat;
        odom_quat.x = 0.0;
        odom_quat.y = 0.0;
        odom_quat.z = 0.0;
        odom_quat.w = 1.0;

        odom_trans.transform.translation.x = 0.0;
        odom_trans.transform.translation.y = 0.0;
        odom_trans.transform.translation.z = 0.0;
        odom_trans.transform.rotation = odom_quat;

        odom_trans.header.frame_id = frame_id_
        odom_trans.child_frame_id = child_frame_id_;
    }

    void publish_tf()
    {
        odom_trans.header.stamp = ros::Time::now();
        tf_br.sendTransform(odom_trans);
    }

    void publish_odom()
    {
        odom.header.stamp = ros::Time::now();
        odom_pub.publish(odom);
    }
}

// class Mapper
// {
//     OdometryEstimator odom_estimator;

    
// }

class Localization
{
    ros::NodeHandle nh_;
    // POSE
    geometry_msgs::TransformStamped current_pose_;
    geometry_msgs::TransformStamped prev_pose_;
    ros::Publisher pose_pub;

    // FRAMES
    std::string frame_id_ = "map";
    std::string child_frame_id_ = "odom_brky";

    // TF
    tf2_ros::TransformBroadcaster tf_br;

    // POINTCLOUD
    pcl::PointCloud<Point_T>::Ptr map_cloud_{new pcl::PointCloud<Point_T>};
    pcl::PointCloud<Point_T>::Ptr last_cloud_;
    ros::Subscriber pc_sub;
    std::string pc_topic_ = "/pointcloud_VLS128_right_pcd";

    //GPS
    GPSOdometryEstimator gps_odometry_estimator;
    double last_displacement{0};
    double displacement_threshold{10.0};

    // ICP Matcher
    

public:
    Localization():nh_("~")
    {
        initPose();
        pose_pub = nh_.advertise<geometry_msgs::TransformStamped>("pose", 10);
        pc_sub = nh_.subscribe<sensor_msgs::PointCloud2>(
            pc_topic_, 1, &Localization::pointCloudCallback, this);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& current_cloud_msg)
    {
        if(map_cloud_->size() == 0)
        {
            pcl::fromROSMsg(*current_cloud_msg, *map_cloud_);
        }
        else
        {
            pcl::PointCloud<Point_T>::Ptr current_cloud{new pcl::PointCloud<Point_T>};
            pcl::fromROSMsg(*current_cloud_msg, *current_cloud);
            
            double diff_disp = propagateWithGPS();
            if( diff_disp>=displacement_threshold)
            {
                last_displacement = total_disp;

                triggerICP(map_cloud_, current_cloud, current_pose_);
            }
        }

        publish_pose_and_tf();
    }

    void triggerICP(const pcl::PointCloud<Point_T>::Ptr map_cloud,
                    const pcl::PointCloud<Point_T>::Ptr current_cloud,
                    const geometry_msgs::TransformStamped &current_pose)
    {


    }

    double propagateWithGPS()
    {

        double total_disp = gps_odometry_estimator.get_total_displacement();
        double diff_disp = total_disp - last_displacement;
        translate_pose_on_x_axis(diff_disp);
        ROS_DEBUG_STREAM("disp: " << diff_disp);
        return diff_disp;
    }

    void translate_pose_on_x_axis(double distance)
    {
        current_pose_.transform.translation.x += distance;
    }
    
    void initPose()
    {
        geometry_msgs::Quaternion odom_quat;
        odom_quat.x = 0.0;
        odom_quat.y = 0.0;
        odom_quat.z = 0.0;
        odom_quat.w = 1.0;

        current_pose_.transform.translation.x = 0.0;
        current_pose_.transform.translation.y = 0.0;
        current_pose_.transform.translation.z = 0.0;
        current_pose_.transform.rotation = odom_quat;

        current_pose_.header.frame_id = frame_id_;
        current_pose_.child_frame_id = child_frame_id_;
    }

    void publish_pose_and_tf()
    {
        current_pose_.header.stamp = ros::Time::now();
        tf_br.sendTransform(current_pose_);
        pose_pub.publish(current_pose_);
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "localization_node");
    
    Localization localization;

    ros::spin();

    return 0;
}