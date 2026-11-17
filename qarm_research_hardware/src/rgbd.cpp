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
//
// RGB-D camera node for the QArm Research arm. Publishes colour and 16-bit
// depth through image_transport, with each stream's camera info.
//
// Image messages are filled directly rather than through cv_bridge, which can
// link a different OpenCV version from this node.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "image_transport/image_transport.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "quanser/quanser_messages.h"
#include "quanser/quanser_video3d.h"

namespace
{
constexpr double kMillimetresPerMetre = 1000.0;
}  // namespace

class RGBD : public rclcpp::Node
{
public:
  RGBD()
  : Node("rgbd")
  {
    declare_parameters();
    read_parameters();

    if (!open_camera()) {
      return;
    }

    report_depth_scale();
    if (!read_intrinsics(rgb_stream_, "colour", color_frame_id_, frame_width_rgb_,
      frame_height_rgb_, color_info_) ||
      !read_intrinsics(depth_stream_, "depth", depth_frame_id_, frame_width_depth_,
      frame_height_depth_, depth_info_))
    {
      return;
    }

    rgb_pub_ = image_transport::create_publisher(this, "camera/color_image", rmw_qos_profile_sensor_data);
    depth_pub_ = image_transport::create_publisher(this, "camera/depth_image", rmw_qos_profile_sensor_data);

    const auto info_qos = rclcpp::SensorDataQoS();
    color_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("camera/color_info", info_qos);
    depth_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("camera/depth_info", info_qos);

    rgb_buffer_.resize(static_cast<size_t>(frame_width_rgb_) * frame_height_rgb_ * 3);
    depth_buffer_.resize(static_cast<size_t>(frame_width_depth_) * frame_height_depth_);

    streaming_ = true;
    RCLCPP_INFO(
      get_logger(),
      "rgbd ready. colour %ux%u, depth %ux%u, %.1f Hz, device '%s'.",
      frame_width_rgb_, frame_height_rgb_, frame_width_depth_, frame_height_depth_,
      frame_rate_, camera_identifier_.c_str());
  }

  ~RGBD() override
  {
    if (opened_) {
      const t_error result = video3d_close(capture_);
      if (result < 0) {
        RCLCPP_ERROR(get_logger(), "Closing the camera failed: %d", result);
      }
    }
    RCLCPP_INFO(get_logger(), "rgbd exit");
  }

  bool streaming() const {return streaming_;}
  double frame_rate() const {return frame_rate_;}

  // One stamp for both images, so the vision node's synchroniser matches them exactly.
  void publish_images()
  {
    const rclcpp::Time stamp = get_clock()->now();
    publish_color(stamp);
    publish_depth(stamp);
  }

private:
  void declare_parameters()
  {
    declare_parameter("device_type", std::string("physical"));
    declare_parameter("camera_num", std::string("0"));
    declare_parameter("device_num", std::string(""));
    declare_parameter("frame_width_rgb", 640);
    declare_parameter("frame_height_rgb", 480);
    declare_parameter("frame_width_depth", 640);
    declare_parameter("frame_height_depth", 480);
    declare_parameter("frame_rate", 30.0);
    declare_parameter("color_frame_id", std::string("color_image"));
    declare_parameter("depth_frame_id", std::string("depth_image"));
  }

  void read_parameters()
  {
    device_type_ = get_parameter("device_type").as_string();
    camera_num_ = get_parameter("camera_num").as_string();
    device_num_ = get_parameter("device_num").as_string();
    frame_width_rgb_ = static_cast<t_uint32>(get_parameter("frame_width_rgb").as_int());
    frame_height_rgb_ = static_cast<t_uint32>(get_parameter("frame_height_rgb").as_int());
    frame_width_depth_ = static_cast<t_uint32>(get_parameter("frame_width_depth").as_int());
    frame_height_depth_ = static_cast<t_uint32>(get_parameter("frame_height_depth").as_int());
    frame_rate_ = get_parameter("frame_rate").as_double();
    color_frame_id_ = get_parameter("color_frame_id").as_string();
    depth_frame_id_ = get_parameter("depth_frame_id").as_string();
  }

  bool open_camera()
  {
    if (device_type_ == "physical") {
      camera_identifier_ = "0";
    } else if (device_type_ == "virtual") {
      camera_identifier_ = "0@tcpip://localhost:18965";
    } else if (device_type_ == "custom") {
      camera_identifier_ = camera_num_ + device_num_;
    } else {
      RCLCPP_ERROR(
        get_logger(),
        "device_type '%s' is not one of physical, virtual or custom.", device_type_.c_str());
      return false;
    }

    if (frame_rate_ <= 0.0) {
      RCLCPP_ERROR(get_logger(), "frame_rate must be > 0, not %.3f.", frame_rate_);
      return false;
    }

    // The vision node indexes a colour mask straight into the depth image.
    if (frame_width_rgb_ != frame_width_depth_ || frame_height_rgb_ != frame_height_depth_) {
      RCLCPP_ERROR(
        get_logger(),
        "Colour %ux%u and depth %ux%u must be the same size: the vision node "
        "indexes a colour mask into the depth image.",
        frame_width_rgb_, frame_height_rgb_, frame_width_depth_, frame_height_depth_);
      return false;
    }

    t_error result = video3d_open(camera_identifier_.c_str(), &capture_);
    if (result < 0) {
      RCLCPP_ERROR(get_logger(), "video3d_open failed: %s", error_text(result).c_str());
      return false;
    }
    opened_ = true;

    result = video3d_stream_open(
      capture_, VIDEO3D_STREAM_COLOR, 0, frame_rate_, frame_width_rgb_, frame_height_rgb_,
      IMAGE_FORMAT_ROW_MAJOR_INTERLEAVED_BGR, IMAGE_DATA_TYPE_UINT8, &rgb_stream_);
    if (result < 0) {
      RCLCPP_ERROR(get_logger(), "Opening the colour stream failed: %s", error_text(result).c_str());
      return false;
    }

    result = video3d_stream_open(
      capture_, VIDEO3D_STREAM_DEPTH, 0, frame_rate_, frame_width_depth_, frame_height_depth_,
      IMAGE_FORMAT_ROW_MAJOR_GRAYSCALE, IMAGE_DATA_TYPE_UINT16, &depth_stream_);
    if (result < 0) {
      RCLCPP_ERROR(get_logger(), "Opening the depth stream failed: %s", error_text(result).c_str());
      return false;
    }

    result = video3d_start_streaming(capture_);
    if (result < 0) {
      RCLCPP_ERROR(get_logger(), "video3d_start_streaming failed: %s", error_text(result).c_str());
      return false;
    }
    return true;
  }

  // Depth is published in millimetres whatever the device's raw unit.
  void report_depth_scale()
  {
    t_single depth_scale_m = 0.0f;
    const t_error result =
      video3d_stream_get_depth_scale(depth_stream_, IMAGE_DATA_TYPE_UINT16, &depth_scale_m);
    if (result < 0 || depth_scale_m <= 0.0f) {
      RCLCPP_WARN(
        get_logger(),
        "Could not read the depth scale (%s); assuming 1 mm per raw unit. "
        "If the true scale differs, published depth is off by that factor.",
        result < 0 ? error_text(result).c_str() : "the camera reported a non-positive scale");
      depth_mm_per_unit_ = 1.0;
      return;
    }

    depth_mm_per_unit_ = static_cast<double>(depth_scale_m) * kMillimetresPerMetre;
    if (std::abs(depth_mm_per_unit_ - 1.0) < 1e-6) {
      RCLCPP_INFO(get_logger(), "Depth scale is 1 mm per unit; published as captured.");
      depth_needs_scaling_ = false;
    } else {
      RCLCPP_INFO(
        get_logger(),
        "Depth scale is %.6f mm per unit; converting to millimetres on publish.",
        depth_mm_per_unit_);
      depth_needs_scaling_ = true;
    }
  }

  // Returns false, and the node does not start, if the camera reports none.
  bool read_intrinsics(
    t_video3d_stream stream, const char * name, const std::string & frame_id,
    t_uint32 width, t_uint32 height, sensor_msgs::msg::CameraInfo & info)
  {
    t_single intrinsics[3][3] = {};
    t_video3d_distortion_model model = VIDEO3D_DISTORTION_NONE;
    t_single coefficients[5] = {};
    const t_error result =
      video3d_stream_get_camera_intrinsics(stream, intrinsics, &model, coefficients);
    if (result < 0) {
      RCLCPP_ERROR(
        get_logger(), "The camera did not report its %s intrinsics: %s",
        name, error_text(result).c_str());
      return false;
    }

    // Column-major: the principal point is in the last row of the C array.
    const double fx = intrinsics[0][0];
    const double fy = intrinsics[1][1];
    const double cx = intrinsics[2][0];
    const double cy = intrinsics[2][1];

    info.header.frame_id = frame_id;
    info.width = width;
    info.height = height;
    info.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
    info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info.p = {fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0};
    fill_distortion(name, model, coefficients, info);

    RCLCPP_INFO(
      get_logger(), "%s intrinsics from the camera: fx=%.2f fy=%.2f cx=%.2f cy=%.2f, %s.",
      name, fx, fy, cx, cy, info.distortion_model.c_str());
    return true;
  }

  // ROS has no modified or inverse Brown-Conrady, so those are plumb_bob only
  // when their coefficients are all zero.
  void fill_distortion(
    const char * name, t_video3d_distortion_model model, const t_single coefficients[5],
    sensor_msgs::msg::CameraInfo & info)
  {
    const bool all_zero = std::all_of(
      coefficients, coefficients + 5, [](t_single c) {return c == 0.0f;});

    if (model == VIDEO3D_DISTORTION_KANNALA_BRANDT4) {
      info.distortion_model = "equidistant";
      info.d.assign(coefficients, coefficients + 4);
      return;
    }

    info.distortion_model = "plumb_bob";
    if (model == VIDEO3D_DISTORTION_NONE || model == VIDEO3D_DISTORTION_BROWN_CONRADY ||
      all_zero)
    {
      info.d.assign(coefficients, coefficients + 5);
      return;
    }

    RCLCPP_WARN(
      get_logger(),
      "The %s stream reports distortion model %d, which has no ROS equivalent; "
      "publishing zero distortion.", name, static_cast<int>(model));
    info.d.assign(5, 0.0);
  }

  void publish_color(const rclcpp::Time & stamp)
  {
    t_video3d_frame frame;
    t_error result = video3d_stream_get_frame(rgb_stream_, &frame);
    if (result < 0) {
      // WOULD_BLOCK: the next frame has not arrived yet.
      if (result != -QERR_WOULD_BLOCK) {
        RCLCPP_ERROR(
          get_logger(), "Getting a colour frame failed: %s", error_text(result).c_str());
      }
      return;
    }

    result = video3d_frame_get_data(frame, rgb_buffer_.data());
    video3d_frame_release(frame);
    if (result < 0) {
      RCLCPP_ERROR(get_logger(), "Reading colour frame data failed: %s", error_text(result).c_str());
      return;
    }

    sensor_msgs::msg::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = color_frame_id_;
    msg.height = frame_height_rgb_;
    msg.width = frame_width_rgb_;
    msg.encoding = "bgr8";
    msg.is_bigendian = 0;
    msg.step = frame_width_rgb_ * 3;
    msg.data.assign(rgb_buffer_.begin(), rgb_buffer_.end());
    rgb_pub_.publish(msg);

    color_info_.header = msg.header;
    color_info_pub_->publish(color_info_);
  }

  void publish_depth(const rclcpp::Time & stamp)
  {
    t_video3d_frame frame;
    t_error result = video3d_stream_get_frame(depth_stream_, &frame);
    if (result < 0) {
      if (result != -QERR_WOULD_BLOCK) {
        RCLCPP_ERROR(get_logger(), "Getting a depth frame failed: %s", error_text(result).c_str());
      }
      return;
    }

    result = video3d_frame_get_data(frame, depth_buffer_.data());
    video3d_frame_release(frame);
    if (result < 0) {
      RCLCPP_ERROR(get_logger(), "Reading depth frame data failed: %s", error_text(result).c_str());
      return;
    }

    if (depth_needs_scaling_) {
      for (auto & value : depth_buffer_) {
        const double millimetres = static_cast<double>(value) * depth_mm_per_unit_;
        value = static_cast<t_uint16>(std::min(millimetres, 65535.0) + 0.5);
      }
    }

    sensor_msgs::msg::Image msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = depth_frame_id_;
    msg.height = frame_height_depth_;
    msg.width = frame_width_depth_;
    // compressedDepth accepts 16UC1 only.
    msg.encoding = "16UC1";
    msg.is_bigendian = 0;
    msg.step = frame_width_depth_ * 2;
    const auto * depth_bytes = reinterpret_cast<const uint8_t *>(depth_buffer_.data());
    msg.data.assign(depth_bytes, depth_bytes + depth_buffer_.size() * sizeof(t_uint16));
    depth_pub_.publish(msg);

    depth_info_.header = msg.header;
    depth_info_pub_->publish(depth_info_);
  }

  std::string error_text(t_error result) const
  {
    char message[512];
    msg_get_error_messageA(nullptr, result, message, sizeof(message));
    return std::string(message);
  }

  std::string device_type_;
  std::string camera_num_;
  std::string device_num_;
  std::string camera_identifier_;
  std::string color_frame_id_;
  std::string depth_frame_id_;

  t_uint32 frame_width_rgb_ = 640;
  t_uint32 frame_height_rgb_ = 480;
  t_uint32 frame_width_depth_ = 640;
  t_uint32 frame_height_depth_ = 480;
  double frame_rate_ = 30.0;

  double depth_mm_per_unit_ = 1.0;
  bool depth_needs_scaling_ = false;

  bool opened_ = false;
  bool streaming_ = false;

  t_video3d capture_ = nullptr;
  t_video3d_stream rgb_stream_ = nullptr;
  t_video3d_stream depth_stream_ = nullptr;

  std::vector<t_uint8> rgb_buffer_;
  std::vector<t_uint16> depth_buffer_;

  image_transport::Publisher rgb_pub_;
  image_transport::Publisher depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_;
  sensor_msgs::msg::CameraInfo color_info_;
  sensor_msgs::msg::CameraInfo depth_info_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RGBD>();

  if (!node->streaming()) {
    RCLCPP_ERROR(node->get_logger(), "The camera did not start; exiting.");
    rclcpp::shutdown();
    return 1;
  }

  // One thread owns the camera handle; spin_some services everything else.
  RCLCPP_INFO(node->get_logger(), "Starting rgbd camera loop...");
  rclcpp::Rate rate(node->frame_rate());
  while (rclcpp::ok()) {
    node->publish_images();
    rclcpp::spin_some(node);
    rate.sleep();
  }

  RCLCPP_INFO(node->get_logger(), "rgbd camera ended.");
  rclcpp::shutdown();
  return 0;
}
