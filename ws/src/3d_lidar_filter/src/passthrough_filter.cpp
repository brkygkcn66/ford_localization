#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <sensor_msgs/PointCloud2.h>

class PassthroughFilter
{

    ros::NodeHandle nh_;
    ros::Subscriber point_cloud_sub;
    ros::Publisher filtered_pub;

    std::string input_pc_topic;
    std::string filtered_pc_topic;

public:

    PassthroughFilter(void):nh_("~")
    {
        read_params();
        // Subscribe to the input pc topic
        point_cloud_sub = nh_.subscribe<sensor_msgs::PointCloud2>(
        input_pc_topic, 1, &PassthroughFilter::pointCloudCallback, this);

        // Publisher for the filtered pc
        filtered_pub = nh_.advertise<sensor_msgs::PointCloud2>(
        filtered_pc_topic, 1);
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud_msg)
    {
        // Convert ROS PointCloud2 message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input_cloud_msg, *input_cloud);

        // Create a passthrough filter to filter points below a certain z threshold
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(input_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-30.0, 30.0); // Set your desired z-axis limits here // 3.52 m lidar yüksekliği + 0.8 m ile 4 m olan azami araç yüksekliği
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pass.filter(*filtered_cloud);

        pass.setInputCloud(filtered_cloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-10.0, 50.0);
        pass.filter(*filtered_cloud);

        // Convert filtered PCL point cloud back to ROS PointCloud2 message
        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
        filtered_cloud_msg.header = input_cloud_msg->header;

        // Publish the filtered point cloud
        filtered_pub.publish(filtered_cloud_msg);
    }

    void read_params()
    {
        if (!nh_.getParam("input_pc_topic", input_pc_topic))
        {
            ROS_ERROR_STREAM("Failed to get parameter input_pc_topic");
        }
        ROS_INFO_STREAM("input_pc_topic name: " << input_pc_topic);

        if (!nh_.getParam("filtered_pc_topic", filtered_pc_topic))
        {
            ROS_ERROR_STREAM("Failed to get parameter filtered_pc_topic");
        }
        ROS_INFO_STREAM("filtered_pc_topic name: " << filtered_pc_topic);
    }
    
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "passthrough_filter_node");

    PassthroughFilter passthrough_filter;

    ros::spin();

    return 0;
}
