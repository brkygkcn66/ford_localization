#include <ros/ros.h>
#include <highwaypilot_msgs/HPVehicleStatus.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>

class SensorExtractor
{
    ros::NodeHandle nh;

    // Create subscribers
    ros::Subscriber vehicleStatusSub;

    // Create publishers
    ros::Publisher imuPub;
    ros::Publisher gpsPub;

public:
    SensorExtractor():nh("~"){

        vehicleStatusSub = nh.subscribe("/vehicle_status", 10, &SensorExtractor::vehicleStatusCallback, this);
        imuPub = nh.advertise<sensor_msgs::Imu>("/imu", 10);
        gpsPub = nh.advertise<sensor_msgs::NavSatFix>("/gps/fix", 10);
    }


    void publish_imu(const highwaypilot_msgs::HPVehicleStatus::ConstPtr& msg)
    {
        // Create and publish IMU message
        sensor_msgs::Imu imu_msg;
        // Fill in the header
        imu_msg.header.stamp = ros::Time::now();
        imu_msg.header.frame_id = "novatel_gps";

        auto degree_to_rad = [](auto degree) -> double{
            return degree * (M_PI / 180.0);
        };

        double roll = degree_to_rad(msg->gps_roll);    
        double pitch = degree_to_rad(msg->gps_pitch);    
        double yaw = degree_to_rad(msg->gps_yaw);    

        // Create a Quaternion using tf2
        tf2::Quaternion quaternion;
        quaternion.setRPY(roll, pitch, yaw);

        // Fill in the orientation
        imu_msg.orientation.x = quaternion.x();
        imu_msg.orientation.y = quaternion.y();
        imu_msg.orientation.z = quaternion.z();
        imu_msg.orientation.w = quaternion.w();

        imu_msg.orientation_covariance[0] = degree_to_rad(msg->gps_roll_var);
        imu_msg.orientation_covariance[4] = degree_to_rad(msg->gps_pitch_var);
        imu_msg.orientation_covariance[9] = degree_to_rad(msg->gps_yaw_var);

        // Fill in the angular velocity
        imu_msg.angular_velocity.x = degree_to_rad(msg->gps_x_angrate);
        imu_msg.angular_velocity.y = degree_to_rad(msg->gps_y_angrate);
        imu_msg.angular_velocity.z = degree_to_rad(msg->gps_z_angrate);

        // Fill in the linear acceleration
        imu_msg.linear_acceleration.x = msg->gps_x_acc;
        imu_msg.linear_acceleration.y = msg->gps_y_acc;
        imu_msg.linear_acceleration.z = msg->gps_z_acc;

        imuPub.publish(imu_msg);
    }
    

    void publish_gps(const highwaypilot_msgs::HPVehicleStatus::ConstPtr& msg)
    {
        sensor_msgs::NavSatFix gps_msg;
        // Fill in gpsMsg with relevant data
        // Fill in the header
        gps_msg.header.stamp = ros::Time::now();
        gps_msg.header.frame_id = "novatel_gps";

        // Fill in the GPS data
        gps_msg.latitude = msg->gps_latitude;
        gps_msg.longitude = msg->gps_longitude;
        gps_msg.altitude = msg->gps_height;

        gpsPub.publish(gps_msg);
    }

    void vehicleStatusCallback(const highwaypilot_msgs::HPVehicleStatus::ConstPtr& msg)
    {
        publish_imu(msg);
        publish_gps(msg);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "sensor_extractor_node");
    
    SensorExtractor se;

    ros::spin();

    return 0;
}