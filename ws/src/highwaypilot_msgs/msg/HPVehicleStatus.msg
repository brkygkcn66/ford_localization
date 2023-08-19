Header header

# Powertrain
uint8 drivemode
uint8 steeringmode
uint8 MODE_MANUAL=0
uint8 MODE_AUTO=1
uint8 MODE_OVERRIDE=2
uint8 MODE_FAILURE=4

# Vehicle engine speed [rpm]
float32 engine_speed
# Vehicle front axle speed [km/h]
float32 speed 
# Vehicle steering wheel angle [rad]
float32 angle

# Vehicle throttle [percentage]
uint8 throttle
# Vehicle brake [percentage]
uint8 brake
# Vehicle current gear
uint8 current_gear
# Vehicle slope [percentage]
int8 slope
# Vehicle trailer status
uint8 trailer_status
uint8 TRAILER_NOT_CONNECTED=0
uint8 TRAILER_CONNECTED=1
uint8 TRAILER_NOT_AVAILABLE=3

# Lights (8 bits, each bit represent a light)
uint8 lights
uint8 LIGHTS_OFF=0
uint8 LIGHTS_PARKING=1
uint8 LIGHTS_DAYTIME=2
uint8 LIGHTS_LOW_BEAM=4
uint8 LIGHTS_HIGH_BEAM=8
uint8 LIGHTS_FRONT_FOG=16
uint8 LIGHTS_REAR_FOG=32
uint8 LIGHTS_LEFT_INDICATOR=64
uint8 LIGHTS_RIGHT_INDICATOR=128
# If both of the indicator lights are on, it's a hazard warning!
uint8 LIGHTS_HAZARD_WARNING=192

# Longitudinal velocity at rear axle - unit: m/s
float32 VLgt

# Lateral velocity at rear axle - unit: m/s
float32 VLat

# Longitudinal acceleration at rear axle - unit: m/s^2
float32 ALgt

# Lateral acceleration at rear axle - unit: m/s^2
float32 ALat

# NovAtel
# UTC GPS Time (integer part: seconds of the day, decimal part: miliseconds)
float64 gps_time
# Roll angle in IMU Frame (deg)
float32 gps_roll
# Pitch angle in IMU Frame (deg)
float32 gps_pitch
# Yaw angle in IMU Frame (deg)
float32 gps_yaw
# Geographic Longitude (decimal degrees)
float64 gps_longitude
# Geographic Latitude (decimal degrees)
float64 gps_latitude
# Geographic Altitude (m)
float32 gps_height
# Angular rate along the x-axis in IMU Frame (deg/s)
float32 gps_x_angrate
# Angular rate along the y-axis in IMU Frame (deg/s)
float32 gps_y_angrate
# Angular rate along the z-axis in IMU Frame (deg/s)
float32 gps_z_angrate
# Acceleration along the x-axis in IMU Frame (m/s^2)
float32 gps_x_acc
# Acceleration along the y-axis in IMU Frame (m/s^2)
float32 gps_y_acc
# Acceleration along the z-axis in IMU Frame (m/s^2)
float32 gps_z_acc
# East Velocity in Global Frame (m/s)
float32 gps_east_vel
# North Velocity in Global Frame (m/s)
float32 gps_north_vel
# Up Velocity in Global Frame (m/s)
float32 gps_up_vel
# Inertial Navigation System Status
uint8 gps_ins_status
# Number of Satellites in View
uint8 gps_satellites
# Solution Status
uint32 gps_sol_status
# Position/Velocity Status
uint32 gps_pos_status

# SCU State
uint8 scu_state
# Internal VCAN Bus Status
uint8 i_vcanbus_status
# Internal STCAN Bus Status
uint8 i_stcanbus_status
# VCAN Bus Status
uint8 vcanbus_status
# STCAN Bus Status
uint8 stcanbus_status
# Green Status Led Fault Code
uint8 green_statusled_faultcode
# Red Status Led Fault Code
uint8 red_statusled_faultcode
# Buzzer Fault Code
uint8 buzzer_faultcode
# Engagement Button Fault Code
uint8 engagementbutton_faultcode

# Brake Switch
uint8 brake_switch
# Engine Overrride Control Mode
uint8 eng_override_ctrl_mode
# External Control Mode Req for Servotwin
uint8 ext_ctrl_mode_req

# Yaw Rate
float32 yaw_rate
# Driver Torque of Servotwin
float32 driver_torque

# Variance of Roll angle in IMU Frame (deg)
float32 gps_roll_var
# Variance of Pitch angle in IMU Frame (deg)
float32 gps_pitch_var
# Variance of Yaw angle in IMU Frame (deg)
float32 gps_yaw_var
# Variance of east (m)
float32 gps_east_var
# Variance of north(m)
float32 gps_north_var
# Variance of up(m)
float32 gps_up_var
# Variance of East Velocity in Global Frame (m/s)
float32 gps_east_vel_var
# Variance of North Velocity in Global Frame (m/s)
float32 gps_north_vel_var
# Variance of Up Velocity in Global Frame (m/s)
float32 gps_up_vel_var

# X Velocity in User Frame (m/s)
float64 gps_x_vel
# Y velocity in User Frame (m/s)
float64 gps_y_vel
# Z Velocity in User Frame (m/s)
float64 gps_z_vel
#Variance of X Velocity in User Frame (m/s)
float32 gps_x_vel_var
#Variance of Y Velocity in User Frame (m/s)
float32 gps_y_vel_var
#Variance of Z Velocity in User Frame (m/s)
float32 gps_z_vel_var
# Slip Angle (deg)
float32 gps_slip_angle
