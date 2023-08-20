#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

#include <tf2_ros/transform_listener.h>

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

#include <sensor_msgs/NavSatFix.h>
#include <cmath>

#include <pcl/registration/ndt.h>

// #include <pcl/memory.h>  // for pcl::make_shared
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/common/transforms.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
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
    pcl::ApproximateVoxelGrid<Point_T> voxel_filter;

    float leaf_size = 0.25;
    int ndt_max_iterations{35};

public:
    ICPMatcher() = default;

    Eigen::Matrix4f calculate_tranformation(const pcl::PointCloud<Point_T>::Ptr map_cloud,
                    const pcl::PointCloud<Point_T>::Ptr current_cloud,
                    const geometry_msgs::TransformStamped &current_pose)
    {
        pcl::PointCloud<Point_T>::Ptr filtered_map_cloud {new pcl::PointCloud<Point_T>};
        pcl::PointCloud<Point_T>::Ptr filtered_current_cloud {new pcl::PointCloud<Point_T>};
        
        voxel_filter.setInputCloud(map_cloud);
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel_filter.filter(*filtered_map_cloud);

        voxel_filter.setInputCloud(current_cloud);
        voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
        voxel_filter.filter(*filtered_current_cloud);

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
        ndt.setInputTarget (filtered_map_cloud);

        // Calculating required rigid transform to align the input cloud to the target cloud.
        pcl::PointCloud<Point_T>::Ptr output_cloud (new pcl::PointCloud<Point_T>);
        ndt.align (*output_cloud);

        if (ndt.hasConverged())
        {
            ROS_INFO("ICP converged with score: %f", ndt.getFitnessScore());
            return ndt.getFinalTransformation ();
        }
        else
        {
            return Eigen::Matrix4f::Identity();
        }
    }

    void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
    {
        //
        // Downsample for consistency and speed
        // \note enable this for large datasets
        PointCloud::Ptr src (new PointCloud);
        PointCloud::Ptr tgt (new PointCloud);
        pcl::VoxelGrid<PointT> grid;
        if (downsample)
        {
            grid.setLeafSize(leaf_size, leaf_size, leaf_size);
            grid.setInputCloud (cloud_src);
            grid.filter (*src);

            grid.setInputCloud (cloud_tgt);
            grid.filter (*tgt);
        }
        else
        {
            src = cloud_src;
            tgt = cloud_tgt;
        }

        ROS_ERROR_STREAM("1");
        // Compute surface normals and curvature
        PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
        PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

        pcl::NormalEstimation<PointT, PointNormalT> norm_est;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        norm_est.setSearchMethod (tree);
        norm_est.setKSearch (30);
        
        norm_est.setInputCloud (src);
        norm_est.compute (*points_with_normals_src);
        pcl::copyPointCloud (*src, *points_with_normals_src);

        norm_est.setInputCloud (tgt);
        norm_est.compute (*points_with_normals_tgt);

        ROS_ERROR_STREAM("2");
        pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

        //
        // Instantiate our custom point representation (defined above) ...
        // MyPointRepresentation point_representation;
        // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
        // float alpha[4] = {1.0, 1.0, 1.0, 1.0};
        // point_representation.setRescaleValues (alpha);

        //
        // Align
        pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
        reg.setTransformationEpsilon (1e-6);
        // Set the maximum distance between two correspondences (src<->tgt) to 10cm
        // Note: adjust this based on the size of your datasets
        reg.setMaxCorrespondenceDistance (0.1);  
        //   // Set the point representation
        //   reg.setPointRepresentation (pcl::make_shared<const MyPointRepresentation> (point_representation));

        reg.setInputSource (points_with_normals_src);
        reg.setInputTarget (points_with_normals_tgt);

        //
        // Run the same optimization in a loop and visualize the results
        Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
        PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
        reg.setMaximumIterations (2);
        ROS_ERROR_STREAM("3");
        for (int i = 0; i < 30; ++i)
        {
            PCL_INFO ("Iteration Nr. %d.\n", i);

            // save cloud for visualization purpose
            points_with_normals_src = reg_result;

            // Estimate
            reg.setInputSource (points_with_normals_src);
            reg.align (*reg_result);

                //accumulate transformation between each Iteration
            Ti = reg.getFinalTransformation () * Ti;

                //if the difference between this transformation and the previous one
                //is smaller than the threshold, refine the process by reducing
                //the maximal correspondence distance
            if (std::abs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
            reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
            
            prev = reg.getLastIncrementalTransformation ();

            // visualize current state
            // showCloudsRight(points_with_normals_tgt, points_with_normals_src);
        }
        ROS_ERROR_STREAM("4");
            //
        // Get the transformation from target to source
        targetToSource = Ti.inverse();

        //
        // Transform target back in source frame
        pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);
        //add the source to the transformed target
        *output += *cloud_src;
        ROS_ERROR_STREAM("5");
        final_transform = targetToSource;
    }
};

// class OdometryEstimator
// {
    
//     ros::NodeHandle nh;

//     ICPMatcher icp_matcher;

//     // FRAMES
//     std::string frame_id_ = "odom_brky";
//     std::string child_frame_id_ = "rear_axle";

//     // TF
//     geometry_msgs::TransformStamped odom_trans;
//     tf2_ros::TransformBroadcaster tf_br;

//     // ODOM
//     ros::Publisher odom_pub;
//     nav_msgs::Odometry odom;
    
// public:
//     OdometryEstimator()
//     {   
//         odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 50);
//         initOdomTransaction();
//     }

//     void initOdomTransaction()
//     {
//         geometry_msgs::Quaternion odom_quat;
//         odom_quat.x = 0.0;
//         odom_quat.y = 0.0;
//         odom_quat.z = 0.0;
//         odom_quat.w = 1.0;

//         odom_trans.transform.translation.x = 0.0;
//         odom_trans.transform.translation.y = 0.0;
//         odom_trans.transform.translation.z = 0.0;
//         odom_trans.transform.rotation = odom_quat;

//         odom_trans.header.frame_id = frame_id_
//         odom_trans.child_frame_id = child_frame_id_;
//     }

//     void publish_tf()
//     {
//         odom_trans.header.stamp = ros::Time::now();
//         tf_br.sendTransform(odom_trans);
//     }

//     void publish_odom()
//     {
//         odom.header.stamp = ros::Time::now();
//         odom_pub.publish(odom);
//     }
// }

// class Mapper
// {
//     OdometryEstimator odom_estimator;

    
// }

class Localization
{
    ros::NodeHandle nh_;
    // POSE
    Eigen::Matrix4f current_pose_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f prev_pose_;
    ros::Publisher pose_pub;

    // FRAMES
    std::string frame_id_ = "odom_brky";
    std::string child_frame_id_ = "rear_axle";

    // TF
    geometry_msgs::TransformStamped pose_trans;
    tf2_ros::TransformBroadcaster tf_br;
    tf::TransformListener listener_;  
    std::string lidar_link = "/VLS128_right_lidar";
    std::string base_link = "/rear_axle";
    Eigen::Matrix4f lidar_to_base_tf_;

    // POINTCLOUD
    pcl::PointCloud<Point_T>::Ptr map_cloud_{new pcl::PointCloud<Point_T>};
    pcl::PointCloud<Point_T>::Ptr last_cloud_;
    ros::Subscriber pc_sub;
    ros::Publisher pc_pub;
    std::string pc_topic_ = "/pointcloud_VLS128_right_pcd";
    

    //GPS
    GPSOdometryEstimator gps_odometry_estimator;
    double last_displacement{0};
    double displacement_threshold{10.0};

    // ICP Matcher
    ICPMatcher icp_matcher;

public:
    Localization():nh_("~")
    {
        get_tf_from_lidar_to_base();

        pose_pub = nh_.advertise<geometry_msgs::TransformStamped>("pose", 10);
        pc_sub = nh_.subscribe<sensor_msgs::PointCloud2>(
            pc_topic_, 1, &Localization::pointCloudCallback, this);
        pc_pub = nh_.advertise<sensor_msgs::PointCloud2>(
        "brky_pc", 1);

        
    }

    void get_tf_from_lidar_to_base()
    {
        tf::StampedTransform transform;
        try {
            listener_.waitForTransform(base_link, lidar_link, ros::Time(0), ros::Duration(30.0) );
            listener_.lookupTransform(base_link, lidar_link, ros::Time(0), transform);

            tf::Vector3 translation = transform.getOrigin();
            tf::Quaternion rotation = transform.getRotation();

            lidar_to_base_tf_ = Eigen::Matrix4f::Identity();
            lidar_to_base_tf_(0, 3) = translation.x();
            lidar_to_base_tf_(1, 3) = translation.y();
            lidar_to_base_tf_(2, 3) = translation.z();

            Eigen::Quaternionf eigen_quaternion(rotation.w(), rotation.x(), rotation.y(), rotation.z());
            Eigen::Matrix3f rotation_matrix = eigen_quaternion.toRotationMatrix();
            lidar_to_base_tf_.block(0, 0, 3, 3) = rotation_matrix;

        } catch (tf::TransformException ex) {
            ROS_ERROR("%s",ex.what());
        }
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& current_cloud_msg)
    {
        if(map_cloud_->size() == 0)
        {
            pcl::fromROSMsg(*current_cloud_msg, *map_cloud_);
            transform_pc_to_baselink(map_cloud_);
            ROS_INFO_STREAM("first");
        }
        else
        {
            pcl::PointCloud<Point_T>::Ptr current_cloud{new pcl::PointCloud<Point_T>};
            pcl::fromROSMsg(*current_cloud_msg, *current_cloud);
            
            transform_pc_to_baselink(current_cloud);
            ROS_INFO_STREAM("2");
            // GPS Check and propagateWithGPS

            double total_disp = gps_odometry_estimator.get_total_displacement();
            double diff_disp = total_disp - last_displacement;
            ROS_INFO_STREAM("disp: " << diff_disp);
            translate_pose_on_x_axis(diff_disp);

            if( diff_disp>=displacement_threshold)
            {
                last_displacement = total_disp;

                // Transform the point cloud using the defined transformation matrix
                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_curr_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::transformPointCloud(*current_cloud, *transformed_curr_cloud, current_pose_);

                // Eigen::Matrix4f transform = triggerICP(map_cloud_, current_cloud, current_pose_);
                pcl::PointCloud<Point_T>::Ptr aligned_cloud{new pcl::PointCloud<Point_T>};
                Eigen::Matrix4f transform;
                icp_matcher.pairAlign(transformed_curr_cloud, map_cloud_, aligned_cloud, transform, true);
                map_cloud_->clear();
                map_cloud_ = aligned_cloud;

                current_pose_ = transform * current_pose_;

                print4x4Matrix(current_pose_);
                // add_cloud_to_map(current_cloud, transform);
                // std::cin.get();

                sensor_msgs::PointCloud2 map_cloud_msg;
                pcl::toROSMsg(*map_cloud_, map_cloud_msg);
                map_cloud_msg.header.stamp = ros::Time::now();
                map_cloud_msg.header.frame_id = frame_id_; 

                // Publish the transformed point cloud
                pc_pub.publish(map_cloud_msg);
            }
        }

        publish_pose_and_tf();
    }

    void transform_pc_to_baselink(pcl::PointCloud<Point_T>::Ptr& cloud)
    {
            
        pcl::PointCloud<Point_T>::Ptr transformed_cloud(new pcl::PointCloud<Point_T>);
        pcl::transformPointCloud(*cloud, *transformed_cloud, lidar_to_base_tf_);
        ROS_ERROR_STREAM("transformed_cloud size: " << transformed_cloud->size());
        cloud->clear();
        cloud = transformed_cloud;
    }

    Eigen::Matrix4f triggerICP(const pcl::PointCloud<Point_T>::Ptr map_cloud,
                    const pcl::PointCloud<Point_T>::Ptr current_cloud,
                    const geometry_msgs::TransformStamped &current_pose)
    {
        ROS_INFO_STREAM("triggerICP");
        Eigen::Matrix4f transformation 
            = icp_matcher.calculate_tranformation(map_cloud,
                                                  current_cloud,
                                                  current_pose
                                                  );    
        print4x4Matrix(transformation);
    }

    // double propagateWithGPS()
    // {

    //     double total_disp = gps_odometry_estimator.get_total_displacement();
    //     double diff_disp = total_disp - last_displacement;
    //     translate_pose_on_x_axis(diff_disp);
    //     ROS_INFO_STREAM("disp: " << diff_disp);
    //     if( diff_disp>=displacement_threshold)
    //     {
    //         last_displacement = total_disp;
    //     }
        
    //     return diff_disp;
    // }

    void translate_pose_on_x_axis(double distance)
    {
        // FIX IT brkygkcn
        current_pose_(0,3) += distance;
    }
    
    void print4x4Matrix (const Eigen::Matrix4f & matrix) const
    {
        printf ("Rotation matrix :\n");
        printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
        printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
        printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
        printf ("Translation vector :\n");
        printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
    }

    void publish_pose_and_tf()
    {
        // Extract the rotation matrix from the position matrix
        // Eigen::Matrix3f rotation_matrix = current_pose_.block<3, 3>(0, 0);

        // // Rotate the rotation matrix by 90 degrees around the Z-axis (Yaw)
        // float angle = M_PI_2; // 90 degrees in radians
        // Eigen::Matrix3f rotation_matrix_rotated;
        // rotation_matrix_rotated = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()) * rotation_matrix;

        // // Replace the rotation part of the position matrix with the rotated rotation matrix
        // current_pose_.block<3, 3>(0, 0) = rotation_matrix_rotated;


        // Convert the Eigen::Matrix4f to a tf2::Transform object
        tf2::Transform tf2_transform;
        tf2::Matrix3x3 rotation(current_pose_(0, 0), current_pose_(0, 1), current_pose_(0, 2),
                                current_pose_(1, 0), current_pose_(1, 1), current_pose_(1, 2),
                                current_pose_(2, 0), current_pose_(2, 1), current_pose_(2, 2));
        tf2::Vector3 translation(current_pose_(0, 3), current_pose_(1, 3), current_pose_(2, 3));
        tf2_transform.setOrigin(translation);
        tf2_transform.setBasis(rotation);

        // Convert the tf2::Transform to a geometry_msgs::TransformStamped message
        geometry_msgs::TransformStamped transform_msg;
        transform_msg.header.stamp = ros::Time::now();
        transform_msg.header.frame_id = frame_id_; 
        transform_msg.child_frame_id = child_frame_id_; 
        transform_msg.transform = tf2::toMsg(tf2_transform);

        // Publish the transformation
        tf_br.sendTransform(transform_msg);

        // pose_pub.publish(current_pose_);
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "localization_node");
    
    Localization localization;

    ros::spin();

    return 0;
}