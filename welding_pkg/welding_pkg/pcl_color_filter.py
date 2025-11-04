#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def rgb_float_to_uint32(rgb_float_arr: np.ndarray) -> np.ndarray:
    # Reinterpret packed float32 to uint32 (endianness follows host)
    return rgb_float_arr.view(np.uint32)

def uint32_to_rgb(rgb_uint32_arr: np.ndarray) -> np.ndarray:
    r = (rgb_uint32_arr >> 16) & 0xFF
    g = (rgb_uint32_arr >> 8) & 0xFF
    b = rgb_uint32_arr & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def rgb_to_hsv_np(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB(0..255) -> HSV with H in [0,1), S,V in [0,1].
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-12

    # Hue
    h = np.empty_like(maxc)
    # where maxc == r
    mask_r = (maxc == r)
    mask_g = (maxc == g)
    mask_b = (maxc == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    h = (h / 6.0) % 1.0  # normalize to [0,1)

    # Saturation
    s = np.where(maxc > 0.0, delta / maxc, 0.0)

    # Value
    v = maxc

    return np.stack([h, s, v], axis=-1)

def rgb_tuple_to_float(rgb_arr: np.ndarray) -> np.ndarray:
    packed = (rgb_arr[:, 0].astype(np.uint32) << 16) | \
             (rgb_arr[:, 1].astype(np.uint32) << 8) | \
              rgb_arr[:, 2].astype(np.uint32)
    return packed.view(np.float32)

class PointCloudColorFilterHSV(Node):
    def __init__(self):
        super().__init__('pointcloud_color_filter_hsv')
        self.sub = self.create_subscription(PointCloud2, '/zed/zed_node/point_cloud/cloud_registered', self.cb, 10)
        self.pub = self.create_publisher(PointCloud2, '/filtered_cloud', 10)
        self.get_logger().info('✅ PointCloud HSV color filter node running.')

        # Thresholds (tune as needed)
        self.h_low = 15.0 / 360.0     # lower red bound (wrap)
        self.h_high = 340.0 / 360.0   # upper red bound (wrap) -> expressed via > h_high
        self.s_min = 0.4              # avoid greys
        self.v_min = 0.2              # avoid dark noise

    def cb(self, msg: PointCloud2):
        # Read points to numpy array [N,4]: x,y,z,rgb_float
        # pts = np.array(list(point_cloud2.read_points(
        #     msg, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True
        # )), dtype=np.float32)

        pts = point_cloud2.read_points_numpy(
            msg, field_names=('x','y','z','rgb'), skip_nans=False
        )  # structured array
        xyz = pts[:,:-1]
        rgb_float = pts[:,-1].astype(np.float32)
        # Decode packed RGB
        rgb_uint32 = rgb_float_to_uint32(rgb_float)
        rgb = uint32_to_rgb(rgb_uint32)  # Nx3 uint8

        # Convert to HSV
        hsv = rgb_to_hsv_np(rgb)  # Nx3 float32 (h,s,v) in [0..1]
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        # Red mask with hue wraparound + saturation/value gates
        # Red if h in [0,h_low) U (h_high,1]
        red_hue = (h < self.h_low) | (h > self.h_high)
        red_mask = red_hue & (s >= self.s_min) & (v >= self.v_min)

        # Make non-red points white
        rgb_out = rgb.copy()
        rgb_out[~red_mask] = np.array([255, 255, 255], dtype=np.uint8)
        rgb_out[red_mask] = np.array([255, 0, 0], dtype=np.uint8)


        # Repack to float
        rgb_new_float = rgb_tuple_to_float(rgb_out)

        # Compose output points
        pts_out = np.column_stack((xyz, rgb_new_float)).astype(np.float32)

        # Publish (reuse original layout/fields so RViz decodes RGB correctly)
        out_msg = point_cloud2.create_cloud(msg.header, msg.fields, pts_out)
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudColorFilterHSV()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
