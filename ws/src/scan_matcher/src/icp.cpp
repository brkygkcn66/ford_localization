#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

class ICPMatcher
{
    ros::NodeHandle nh_;
    ros::Subscriber target_cloud_sub;
    ros::Publisher transformed_pub;

    std::string target_pc_topic;
    std::string transformed_pc_topic;

    int iterations{10};

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud{new pcl::PointCloud<pcl::PointXYZ>};

    bool is_first{true};

    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d lidar_odom_matrix = Eigen::Matrix4d::Identity ();


public:
    ICPMatcher():nh_("~"){

        read_params();

        // Subscribe to the target point cloud topic
        target_cloud_sub = nh_.subscribe<sensor_msgs::PointCloud2>(
            target_pc_topic, 1, &ICPMatcher::pointCloudCallback, this);

        // Create a publisher for the transformed point cloud
        transformed_pub = nh_.advertise<sensor_msgs::PointCloud2>(
        transformed_pc_topic, 1);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& target_cloud_msg)
    {
        if(is_first)
        {
            source_cloud->clear();
            pcl::fromROSMsg(*target_cloud_msg, *source_cloud);
            is_first = false;
        }
        else
        {
            // Convert ROS PointCloud2 message to PCL point cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*target_cloud_msg, *target_cloud);

            // ICP registration
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(source_cloud);
            icp.setInputTarget(target_cloud);
            icp.setMaximumIterations (iterations);
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            icp.align(*transformed_cloud);

            if (icp.hasConverged())
            {
                ROS_INFO("ICP converged with score: %f", icp.getFitnessScore());

                transformation_matrix = icp.getFinalTransformation ().cast<double>();

                lidar_odom_matrix = transformation_matrix * lidar_odom_matrix;
                print4x4Matrix (lidar_odom_matrix);

                // Convert transformed PCL point cloud back to ROS PointCloud2 message
                sensor_msgs::PointCloud2 transformed_cloud_msg;
                pcl::toROSMsg(*transformed_cloud, transformed_cloud_msg);
                transformed_cloud_msg.header = target_cloud_msg->header;

                // Publish the transformed point cloud
                transformed_pub.publish(transformed_cloud_msg);
            }
            else
            {
                ROS_WARN("ICP did not converge");
            }
            // replace source cloud with new cloud
            source_cloud.swap(target_cloud);
        }
        
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