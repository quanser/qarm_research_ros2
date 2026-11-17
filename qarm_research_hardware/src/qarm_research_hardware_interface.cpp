// Copyright 2026 Quanser Consulting Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// C++ includes
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// ROS 2 includes
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/float64.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

// Quanser includes
#include "quanser/quanser_thread.h"
#include "quanser/quanser_persistent_stream.h"

using namespace std::chrono_literals;
using namespace std::placeholders;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class qarm_research_hardware_interface : public rclcpp_lifecycle::LifecycleNode
{
public:
    explicit qarm_research_hardware_interface(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
    : rclcpp_lifecycle::LifecycleNode("qarm_research_hardware_interface", options)
    {
        // Only declare parameters in the constructor.
        // All resource allocation happens in on_configure().
        auto param_desc = rcl_interfaces::msg::ParameterDescriptor{};

        param_desc.description = "URI of the QArm Research RT model's stream.";
        this->declare_parameter("driver_uri", "tcpip://localhost:18888", param_desc);

        param_desc.description = "Control mode the RT model runs in.";
        this->declare_parameter("operating_mode", 0.0, param_desc);

        param_desc.description = "Per-joint speed ceiling the RT model applies, rad/s.";
        this->declare_parameter("joint_speed_limit", 3.1415, param_desc);

        param_desc.description = "Per-joint acceleration ceiling the RT model applies, rad/s^2.";
        this->declare_parameter("joint_acceleration_limit", 12.566, param_desc);

        param_desc.description = "Tool offset the RT model adds along the end-effector axis, m.";
        this->declare_parameter("end_effector_offset", 0.0, param_desc);

        param_desc.description = "Half-extents [x, y, z] of the keep-out box around the base, m.";
        this->declare_parameter("base_collision_box", std::vector<double>{2.0, 2.0, 0.025}, param_desc);

    }

    ~qarm_research_hardware_interface() override
    {
        // Safety net in case the node is destroyed without a proper
        // deactivate/cleanup transition.
        close_stream();
    }

    // ------------------------- Lifecycle callbacks -------------------------

    CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
    {
        RCLCPP_INFO(this->get_logger(), "Configuring...");

        // Read parameters
        driver_uri               = this->get_parameter("driver_uri").as_string();
        RCLCPP_INFO(this->get_logger(), "driver uri set to: %s", driver_uri.c_str());

        operating_mode           = this->get_parameter("operating_mode").as_double();
        RCLCPP_INFO(this->get_logger(), "operating mode set to: %f", operating_mode);

        joint_speed_limit        = this->get_parameter("joint_speed_limit").as_double();
        RCLCPP_INFO(this->get_logger(), "joint speed limit set to: %f", joint_speed_limit);

        joint_acceleration_limit = this->get_parameter("joint_acceleration_limit").as_double();
        RCLCPP_INFO(this->get_logger(), "joint acceleration limit set to: %f", joint_acceleration_limit);

        end_effector_offset      = this->get_parameter("end_effector_offset").as_double();
        RCLCPP_INFO(this->get_logger(), "end effector offset set to: %f", end_effector_offset);

        const std::vector<double> box = this->get_parameter("base_collision_box").as_double_array();
        if (box.size() != 3)
        {
            RCLCPP_ERROR(this->get_logger(), "base_collision_box must have 3 values, got %zu.", box.size());
            return CallbackReturn::FAILURE;
        }
        for (size_t i = 0; i < 3; ++i)
        {
            base_collision_box[i] = box[i];
        }
        RCLCPP_INFO(this->get_logger(), "base collision box set to: [%f, %f, %f]",
            base_collision_box[0], base_collision_box[1], base_collision_box[2]);

        // Parameter set callback
        try
        {
            parameter_cb = this->add_on_set_parameters_callback(std::bind(&qarm_research_hardware_interface::set_parameters_callback, this, _1));
        }
        catch (const std::bad_alloc & e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error setting up parameters callback. %s", e.what());
            return CallbackReturn::FAILURE;
        }

        // Callback groups for timer and topic options
        topic_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        timer_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions sub_options;
        sub_options.callback_group = topic_callback_group_;

        rclcpp::PublisherOptions pub_options;
        pub_options.callback_group = topic_callback_group_;

        // Struct for using pstream
        t_pstream_options stream_options{};

        stream_options.receive_thread_attributes = NULL;
        stream_options.send_thread_attributes    = NULL;

        stream_options.num_receive_units = NUM_RECEIVE_UNITS;
        stream_options.num_send_units = NUM_SEND_UNITS;

        stream_options.receive_unit_size = sizeof(t_double);
        stream_options.send_unit_size = sizeof(t_double);

        stream_options.receive_buffer_size = 1000;
        stream_options.send_buffer_size = 1000;

        stream_options.send_fifo_size = 1000;
        stream_options.receive_fifo_size = 1000;

        stream_options.flags = 0;
        stream_options.flags |= PSTREAM_FLAG_MULTITHREADED;
        stream_options.flags |= PSTREAM_FLAG_MINIMIZE_LATENCY | PSTREAM_FLAG_SEND_MOST_RECENT | PSTREAM_FLAG_RECEIVE_MOST_RECENT;

        stream_options.size = sizeof(stream_options);

        // Setup the stream to get data from the driver model
        int result = pstream_connect(driver_uri.c_str(), &stream_options, &client);
        if (result != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize persistent stream with result: %d", result);
            return CallbackReturn::FAILURE;
        }
        stream_connected = true;

        //  Create lifecycle subscriptions
        joint_trajectory_subscription_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            "qarm_research/joint_trajectory_command", 10, std::bind(&qarm_research_hardware_interface::joint_trajectory_callback, this, _1), sub_options);

        end_effector_command_subscription_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "qarm_research/end_effector_command", 10, std::bind(&qarm_research_hardware_interface::end_effector_command_callback, this, _1), sub_options);
        joint_speed_command_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "qarm_research/joint_speed_command", 10, std::bind(&qarm_research_hardware_interface::joint_speed_command_callback, this, _1), sub_options);
        end_effector_speed_command_subscription_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "qarm_research/end_effector_speed_command", 10, std::bind(&qarm_research_hardware_interface::end_effector_speed_command_callback, this, _1), sub_options);
        gripper_position_command_subscription_ = this->create_subscription<std_msgs::msg::Float64>(
            "qarm_research/gripper_position_command", 10, std::bind(&qarm_research_hardware_interface::gripper_position_command_callback, this, _1), sub_options);
        gripper_current_command_subscription_ = this->create_subscription<std_msgs::msg::Float64>(
            "qarm_research/gripper_current_command", 10, std::bind(&qarm_research_hardware_interface::gripper_current_command_callback, this, _1), sub_options);
        estop_command_subscription_ = this->create_subscription<std_msgs::msg::Float64>(
            "qarm_research/estop_command", 10, std::bind(&qarm_research_hardware_interface::estop_command_callback, this, _1), sub_options);

        // Create lifecycle publishers (inactive until on_activate)
        joint_state_publisher_           = this->create_publisher<sensor_msgs::msg::JointState>("qarm_research/joint_states", 10, pub_options);
        gripper_state_publisher_         = this->create_publisher<std_msgs::msg::Float64MultiArray>("qarm_research/gripper_state", 10, pub_options);
        gripper_current_publisher_       = this->create_publisher<std_msgs::msg::Float64>("qarm_research/gripper_current", 10, pub_options);
        collision_state_publisher_       = this->create_publisher<std_msgs::msg::Bool>("qarm_research/collision_state", 10, pub_options);
        end_effector_pose_publisher_     = this->create_publisher<geometry_msgs::msg::PoseStamped>("qarm_research/end_effector_pose", 10, pub_options);
        end_effector_velocity_publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("qarm_research/end_effector_velocity", 10, pub_options);
        driver_loopback_publisher_       = this->create_publisher<std_msgs::msg::Float64MultiArray>("qarm_research/driver_loopback", 10, pub_options);


        // Create the timer but keep it stopped until on_activate
        timer_ = this->create_wall_timer(application_comm_sample_time, std::bind(&qarm_research_hardware_interface::timer_callback, this), timer_callback_group_);
        timer_->cancel();

        RCLCPP_INFO(this->get_logger(), "Configured successfully. Stream connected to %s", driver_uri.c_str());
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override
    {
        RCLCPP_INFO(this->get_logger(), "Activating...");

        // Activates all managed entities (lifecycle publishers)
        LifecycleNode::on_activate(state);

        // Start the communication timer
        timer_->reset();

        RCLCPP_INFO(this->get_logger(), "qarm_research_hardware_interface is active!");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override
    {
        RCLCPP_INFO(this->get_logger(), "Deactivating...");

        // Stop the communication timer
        timer_->cancel();

        // Deactivates all managed entities (lifecycle publishers)
        LifecycleNode::on_deactivate(state);

        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override
    {
        RCLCPP_INFO(this->get_logger(), "Cleaning up...");
        release_resources();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down from state %s...", state.label().c_str());

        if (timer_)
        {
            timer_->cancel();
        }
        release_resources();

        RCLCPP_INFO(this->get_logger(), "qarm_research_hardware_interface node has been shutdown successfully!");
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn on_error(const rclcpp_lifecycle::State &) override
    {
        RCLCPP_ERROR(this->get_logger(), "Error occurred, releasing resources.");
        release_resources();
        return CallbackReturn::SUCCESS;
    }

private:
    void close_stream()
    {
        if (stream_connected)
        {
            int result = pstream_close(client);
            RCLCPP_INFO(this->get_logger(), "Stream closed with result: %d", result);
            stream_connected = false;
        }
    }

    // Callback for joint trajectory commands
    void joint_trajectory_callback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
    {
        if (msg->points.size() > 0 && msg->points[0].positions.size() >= 6)
        {
            for (size_t i = 0; i < 6; ++i)
            {
                joint_command[i] = msg->points[0].positions[i];
            }
        }
    }

    // Callback for end effector commands
    void end_effector_command_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        end_effector_command[0] = msg->twist.linear.x;
        end_effector_command[1] = msg->twist.linear.y;
        end_effector_command[2] = msg->twist.linear.z;
        end_effector_command[3] = msg->twist.angular.x;
        end_effector_command[4] = msg->twist.angular.y;
        end_effector_command[5] = msg->twist.angular.z;
    }

    // Callback for joint speed commands
    void joint_speed_command_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (msg->velocity.size() >= 6)
        {
            for (size_t i = 0; i < 6; ++i)
            {
                joint_speed_command[i] = msg->velocity[i];
            }
        }
    }

    // Callback for end effector speed commands
    void end_effector_speed_command_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        end_effector_speed_command[0] = msg->twist.linear.x;
        end_effector_speed_command[1] = msg->twist.linear.y;
        end_effector_speed_command[2] = msg->twist.linear.z;
        end_effector_speed_command[3] = msg->twist.angular.x;
        end_effector_speed_command[4] = msg->twist.angular.y;
        end_effector_speed_command[5] = msg->twist.angular.z;
    }

    // Callback for gripper position command
    void gripper_position_command_callback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        gripper_position_command = msg->data;
    }

    // Callback for gripper current command
    void gripper_current_command_callback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        gripper_current_command = msg->data;
    }

    void estop_command_callback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        estop_command = msg->data;
    }

    void release_resources()
    {
        close_stream();
        timer_.reset();

        // The timer goes before the publishers it uses; the other order can crash.
        joint_state_publisher_.reset();
        gripper_state_publisher_.reset();
        gripper_current_publisher_.reset();
        collision_state_publisher_.reset();
        end_effector_pose_publisher_.reset();
        end_effector_velocity_publisher_.reset();
        driver_loopback_publisher_.reset();
        parameter_cb.reset();

        // Reset warning flags so a re-configure starts fresh
        receive_warning_sent = false;
        receive_recovered_sent = true;
        send_warning_sent = false;
        send_recovered_sent = true;
    }

    void timer_callback()
    {
        // Only communicate while the node is in the active state
        if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
        {
            return;
        }

        int result = pstream_receive(client, receive_buffer);
        if (result >= 0)
        {
            rclcpp::Time receive_time = this->get_clock()->now();

            if (!receive_recovered_sent)
            {
                RCLCPP_INFO(this->get_logger(), "Data received from the QArm_Research_Stream_Interface RT model.");
                receive_recovered_sent = true;
                receive_warning_sent = false;
            }

            // Publisher for joint state information
            auto joint_state_msg = sensor_msgs::msg::JointState();
            joint_state_msg.header.stamp = receive_time;
            joint_state_msg.name = {"joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"};
            joint_state_msg.position = {receive_buffer[6], receive_buffer[7], receive_buffer[8], receive_buffer[9], receive_buffer[10], receive_buffer[11]};
            joint_state_msg.velocity = {receive_buffer[12], receive_buffer[13], receive_buffer[14], receive_buffer[15], receive_buffer[16], receive_buffer[17]};
            joint_state_publisher_->publish(joint_state_msg);

            auto end_effector_pose_msg = geometry_msgs::msg::PoseStamped();
            end_effector_pose_msg.header.stamp = receive_time;
            end_effector_pose_msg.header.frame_id = "base_link";
            end_effector_pose_msg.pose.position.x = receive_buffer[18];
            end_effector_pose_msg.pose.position.y = receive_buffer[19];
            end_effector_pose_msg.pose.position.z = receive_buffer[20];
            end_effector_pose_msg.pose.orientation.x = receive_buffer[21];
            end_effector_pose_msg.pose.orientation.y = receive_buffer[22];
            end_effector_pose_msg.pose.orientation.z = receive_buffer[23];
            end_effector_pose_msg.pose.orientation.w = receive_buffer[24];
            end_effector_pose_publisher_->publish(end_effector_pose_msg);

            auto end_effector_velocity_msg = geometry_msgs::msg::TwistStamped();
            end_effector_velocity_msg.header.stamp = receive_time;
            end_effector_velocity_msg.header.frame_id = "base_link";
            end_effector_velocity_msg.twist.linear.x = receive_buffer[25];
            end_effector_velocity_msg.twist.linear.y = receive_buffer[26];
            end_effector_velocity_msg.twist.linear.z = receive_buffer[27];
            end_effector_velocity_msg.twist.angular.x = receive_buffer[28];
            end_effector_velocity_msg.twist.angular.y = receive_buffer[29];
            end_effector_velocity_msg.twist.angular.z = receive_buffer[30];
            end_effector_velocity_publisher_->publish(end_effector_velocity_msg);

            // Publisher for gripper state information
            std_msgs::msg::Float64MultiArray gripper_msg;
            gripper_msg.data = {receive_buffer[33]};
            gripper_state_publisher_->publish(gripper_msg);

            // Publisher for gripper current information
            std_msgs::msg::Float64 gripper_current_msg;
            gripper_current_msg.data = receive_buffer[32];
            gripper_current_publisher_->publish(gripper_current_msg);


            // Publisher for collision state information
            std_msgs::msg::Bool collision_msg;
            collision_msg.data = receive_buffer[34];
            collision_state_publisher_->publish(collision_msg);

            // Publisher for driver loopback information
            std_msgs::msg::Float64MultiArray loopback_msg;
            loopback_msg.data = {receive_buffer[35]};
            driver_loopback_publisher_->publish(loopback_msg);

            // Echo the timestamp from the most recent valid receive packet.
            send_buffer[46] = static_cast<double>(receive_buffer[36]);

        }
        else if (result != -QERR_WOULD_BLOCK)
        {
            if (!receive_warning_sent)
            {
                RCLCPP_WARN(this->get_logger(), "Error (%d) receiving data from the QArm_Research_Stream_Interface RT model. Make sure the RT model is running.", result);
                receive_warning_sent = true;
                receive_recovered_sent = false;
            }
        }

        // Parameters to return to QArm Research for data and mode modifications

        // Mode
        send_buffer[0] = static_cast<double>(operating_mode); // Operating mode

        // Joint angle command
        send_buffer[1] = static_cast<double>(joint_command[0]); // Joint 1 command
        send_buffer[2] = static_cast<double>(joint_command[1]); // Joint 2 command
        send_buffer[3] = static_cast<double>(joint_command[2]); // Joint 3 command
        send_buffer[4] = static_cast<double>(joint_command[3]); // Joint 4 command
        send_buffer[5] = static_cast<double>(joint_command[4]); // Joint 5 command
        send_buffer[6] = static_cast<double>(joint_command[5]); // Joint 6 command

        // End effector command
        send_buffer[7] = static_cast<double>(end_effector_command[0]); // End effector x command
        send_buffer[8] = static_cast<double>(end_effector_command[1]); // End effector y command
        send_buffer[9] = static_cast<double>(end_effector_command[2]); // End effector z command
        send_buffer[10] = static_cast<double>(end_effector_command[3]); // End effector roll command
        send_buffer[11] = static_cast<double>(end_effector_command[4]); // End effector pitch command
        send_buffer[12] = static_cast<double>(end_effector_command[5]); // End effector yaw command

        // Joint speed command
        send_buffer[13] = static_cast<double>(joint_speed_command[0]); // Joint 1 speed command
        send_buffer[14] = static_cast<double>(joint_speed_command[1]); // Joint 2 speed command
        send_buffer[15] = static_cast<double>(joint_speed_command[2]); // Joint 3 speed command
        send_buffer[16] = static_cast<double>(joint_speed_command[3]); // Joint 4 speed command
        send_buffer[17] = static_cast<double>(joint_speed_command[4]); // Joint 5 speed command
        send_buffer[18] = static_cast<double>(joint_speed_command[5]); // Joint 6 speed command

        // End effector speed command
        send_buffer[19] = static_cast<double>(end_effector_speed_command[0]); // End effector x speed command
        send_buffer[20] = static_cast<double>(end_effector_speed_command[1]); // End effector y speed command
        send_buffer[21] = static_cast<double>(end_effector_speed_command[2]); // End effector z speed command
        send_buffer[22] = static_cast<double>(end_effector_speed_command[3]); // End effector roll speed command
        send_buffer[23] = static_cast<double>(end_effector_speed_command[4]); // End effector pitch speed command
        send_buffer[24] = static_cast<double>(end_effector_speed_command[5]); // End effector yaw speed command

        // Joint Current command
        send_buffer[25] = static_cast<double>(joint_current_command[0]); // Joint 1 current command
        send_buffer[26] = static_cast<double>(joint_current_command[1]); // Joint 2 current command
        send_buffer[27] = static_cast<double>(joint_current_command[2]); // Joint 3 current command
        send_buffer[28] = static_cast<double>(joint_current_command[3]); // Joint 4 current command
        send_buffer[29] = static_cast<double>(joint_current_command[4]); // Joint 5 current command
        send_buffer[30] = static_cast<double>(joint_current_command[5]); // Joint 6 current command

        // Gripper position command
        send_buffer[31] = static_cast<double>(gripper_position_command); // Gripper position command

        // Gripper current command
        send_buffer[32] = static_cast<double>(gripper_current_command); // Gripper current command

        // E-Stop
        send_buffer[33] = static_cast<double>(estop_command); // E-Stop command

        // Gripper LEDs
        send_buffer[34] = static_cast<double>(gripper_leds_command_red); // Gripper LEDs command
        send_buffer[35] = static_cast<double>(gripper_leds_command_green); // Gripper LEDs command
        send_buffer[36] = static_cast<double>(gripper_leds_command_blue); // Gripper LEDs command

        // Base LEDs
        send_buffer[37] = static_cast<double>(base_leds_command_red); // Base LEDs command
        send_buffer[38] = static_cast<double>(base_leds_command_green); // Base LEDs command
        send_buffer[39] = static_cast<double>(base_leds_command_blue); // Base LEDs command

        // Joint Speed limit
        send_buffer[40] = static_cast<double>(joint_speed_limit); // Joint Speed limit

        // Joint Acceleration limit
        send_buffer[41] = static_cast<double>(joint_acceleration_limit); // Joint Acceleration limit

        // End Effector offset
        send_buffer[42] = static_cast<double>(end_effector_offset); // End Effector offset

        // Base collision box parameters
        send_buffer[43] = static_cast<double>(base_collision_box[0]); // Base collision box x
        send_buffer[44] = static_cast<double>(base_collision_box[1]); // Base collision box y
        send_buffer[45] = static_cast<double>(base_collision_box[2]); // Base collision box z

        result = pstream_send(client, send_buffer);
        if (result < 0 && result != -QERR_WOULD_BLOCK)
        {
            if (!send_warning_sent)
            {
                RCLCPP_WARN(this->get_logger(), "Error (%d) sending data to the QArm_Research_Stream_Interface RT model. Make sure the RT model is running.", result);
                send_warning_sent = true;
                send_recovered_sent = false;
            }
        }
        else
        {
            if (!send_recovered_sent)
            {
                RCLCPP_INFO(this->get_logger(), "Data sent to the QArm_Research_Stream_Interface RT model.");
                send_recovered_sent = true;
                send_warning_sent = false;
            }
        }
    }

    // Parameter set callbacks
    rcl_interfaces::msg::SetParametersResult set_parameters_callback(const std::vector<rclcpp::Parameter> & parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;


        for (const auto & parameter : parameters)
        {
            if (parameter.get_name() == "driver_uri")
            {
                if (stream_connected)
                {
                    result.successful = false;
                    result.reason = "driver_uri cannot be changed while the stream is connected. Cleanup, set driver_uri, then configure again.";
                    return result;
                }

                driver_uri = parameter.as_string();
                result.successful = true;
            }
            else if (parameter.get_name() == "operating_mode")
            {
                operating_mode = parameter.as_double();
                result.successful = true;
            }
            else if (parameter.get_name() == "joint_speed_limit")
            {
                joint_speed_limit = parameter.as_double();
                result.successful = true;
            }
            else if (parameter.get_name() == "joint_acceleration_limit")
            {
                joint_acceleration_limit = parameter.as_double();
                result.successful = true;
            }
            else if (parameter.get_name() == "end_effector_offset")
            {
                end_effector_offset = parameter.as_double();
                result.successful = true;
            }
            else if (parameter.get_name() == "base_collision_box")
            {
                std::vector<double> box = parameter.as_double_array();
                if (box.size() == 3)
                {
                    base_collision_box[0] = box[0];
                    base_collision_box[1] = box[1];
                    base_collision_box[2] = box[2];
                    result.successful = true;
                }
                else
                {
                    result.successful = false;
                    result.reason = "base_collision_box must be a vector of size 3.";
                }
            }
            else
            {
                result.successful = false;
                result.reason = "The parameter is invalid.";
            }
        }
        return result;
    }

    // private variables

    // QUARC Stream
    t_pstream client;
    bool stream_connected = false;
    bool receive_warning_sent = false;
    bool receive_recovered_sent = true;
    bool send_warning_sent = false;
    bool send_recovered_sent = true;
    static constexpr int NUM_RECEIVE_UNITS = 37;
    static constexpr int NUM_SEND_UNITS = 47;
    t_double send_buffer[NUM_SEND_UNITS] = {0.0};
    t_double joint_command[6] = {0.0, M_PI/2, -M_PI/2, M_PI, M_PI/2, 0.0};
    t_double end_effector_command[6] = {0.0};
    t_double joint_speed_command[6] = {0.0};
    t_double end_effector_speed_command[6] = {0.0};
    t_double joint_current_command[6] = {0.0};
    t_double gripper_position_command = 0.0;
    t_double gripper_current_command = 0.0;
    t_double estop_command = 0.0;
    t_double gripper_leds_command_red = 0.0;
    t_double gripper_leds_command_green = 1.0;
    t_double gripper_leds_command_blue = 0.0;
    t_double base_leds_command_red = 0.0;
    t_double base_leds_command_green = 1.0;
    t_double base_leds_command_blue = 0.0;
    t_double operating_mode = 0.0;
    t_double joint_speed_limit = 0.0;
    t_double joint_acceleration_limit = 0.0;
    t_double end_effector_offset = 0.0;
    t_double base_collision_box[3] = {2.0, 2.0, 0.025};
    t_double receive_buffer[NUM_RECEIVE_UNITS] = {0.0};

    // ROS2 parameters
    rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr parameter_cb;
    rclcpp::CallbackGroup::SharedPtr topic_callback_group_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;
    rclcpp::TimerBase::SharedPtr timer_;

    //lifecycle subscriptions
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr end_effector_command_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_speed_command_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr end_effector_speed_command_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_position_command_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_current_command_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr estop_command_subscription_;

    // Lifecycle publishers (only publish while the node is active)
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64MultiArray>::SharedPtr gripper_state_publisher_;
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64>::SharedPtr gripper_current_publisher_;
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>::SharedPtr collision_state_publisher_;
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::PoseStamped>::SharedPtr end_effector_pose_publisher_;
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::TwistStamped>::SharedPtr end_effector_velocity_publisher_;
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64MultiArray>::SharedPtr driver_loopback_publisher_;

    // Application variables
    std::string driver_uri = "tcpip://localhost:18888";
    std::chrono::milliseconds application_comm_sample_time = 10ms;
};

int main(int argc, char * argv[])
{
    // Initialize ROS 2 environment
    rclcpp::init(argc, argv);

    // Initialize ROS 2 QArm Research lifecycle node
    auto node = std::make_shared<qarm_research_hardware_interface>();

    // Get multi-threaded executor
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());

    RCLCPP_INFO(node->get_logger(), "qarm_research_hardware_interface lifecycle node started (unconfigured).");
    executor.spin();

    // Shutdown ROS 2 environment
    rclcpp::shutdown();
    return 0;
}
