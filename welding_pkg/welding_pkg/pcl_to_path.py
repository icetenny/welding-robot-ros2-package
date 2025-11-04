#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker


def quaternion_from_two_vectors(a, b):
    """Return quaternion (x,y,z,w) rotating unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        # 180°: pick any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return (v[0], v[1], v[2], 0.0)
    s = math.sqrt((1.0 + c) * 2.0)
    invs = 1.0 / s
    qx, qy, qz = v * invs
    qw = 0.5 * s
    return (qx, qy, qz, qw)


def compute_principal_axis_path(
    points_xyz: np.ndarray,
    interval_m: float = 0.01,
    slab_half_thickness_m: float = 0.02,
    min_pts_per_slab: int = 3,
):
    """
    points_xyz: (N,3) float32/64 in meters.
    Returns: (avg_points: (M,3), direction: (3,), mu: (3,))
    """
    if points_xyz.shape[0] < 2:
        return np.empty((0, 3)), np.array([1.0, 0.0, 0.0]), np.zeros(3)

    mu = points_xyz.mean(axis=0)
    X = points_xyz - mu
    # SVD (robust to covariance scaling)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    d = Vt[0]  # principal direction (unit)

    # Project onto principal axis
    t = X @ d  # scalars along d

    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        return np.empty((0, 3)), d, mu

    sample_ts = np.arange(t_min, t_max + 1e-9, interval_m)

    avg_points = []
    for ts in sample_ts:
        mask = np.abs(t - ts) <= slab_half_thickness_m
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        if idx.size < min_pts_per_slab:
            continue
        p_avg = points_xyz[idx].mean(axis=0)
        avg_points.append(p_avg)

    return (np.vstack(avg_points) if len(avg_points) else np.empty((0, 3))), d, mu


import numpy as np
from scipy.spatial import cKDTree


def local_pca_direction(points):
    # points: (M,3)
    if points.shape[0] < 3:
        return None
    C = np.cov(points.T)
    vals, vecs = np.linalg.eigh(C)
    d = vecs[:, np.argmax(vals)]
    d = d / (np.linalg.norm(d) + 1e-12)
    return d


def arc_length_resample(polyline, ds=0.01):
    # polyline: (K,3)
    if len(polyline) < 2:
        return polyline
    segs = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segs)])
    L = s[-1]
    if L < ds:
        return polyline[[0, -1]]
    samples = np.arange(0.0, L + 1e-9, ds)
    out = np.empty((len(samples), 3), dtype=float)
    j = 0
    for i, si in enumerate(samples):
        while j + 1 < len(s) and s[j + 1] < si:
            j += 1
        if j + 1 >= len(s):
            out[i] = polyline[-1]
            continue
        t = (si - s[j]) / (s[j + 1] - s[j] + 1e-12)
        out[i] = (1 - t) * polyline[j] + t * polyline[j + 1]
    return out


def estimate_curved_path(
    xyz,
    r_nbr=0.08,  # neighborhood radius (m)
    step=0.05,  # step size along tangent (m)
    ahead_dot_min=0.2,  # require forward progress
    max_hops=5000,
    avg_nbr=0.01,  # average neighbors within ±1 cm at each hop to denoise
):
    """
    xyz: (N,3) point cloud in meters.
    Returns: resampled_path: (M,3), raw_polyline: (K,3)
    """
    N = xyz.shape[0]
    if N < 10:
        return np.empty((0, 3)), np.empty((0, 3))

    # global PCA to get rough endpoints
    mu = xyz.mean(axis=0)
    X = xyz - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    d_global = Vt[0]
    t = X @ d_global
    i0, i1 = np.argmin(t), np.argmax(t)
    start_idx = i0  # choose one end; you can also run from both ends and merge

    kdt = cKDTree(xyz)
    path = []
    visited = np.zeros(N, dtype=bool)

    cur_idx = start_idx
    cur_pt = xyz[cur_idx].copy()
    prev_dir = (
        d_global
        if d_global.dot(local_pca_direction(xyz[kdt.query_ball_point(cur_pt, r_nbr)]))
        >= 0
        else -d_global
    )

    for hop in range(max_hops):
        path.append(cur_pt.copy())

        # local neighbors for PCA direction
        nbr_idx = kdt.query_ball_point(cur_pt, r_nbr)
        if len(nbr_idx) < 5:
            break
        P = xyz[nbr_idx]
        d_loc = local_pca_direction(P)
        if d_loc is None:
            break
        # Make tangent consistent (no flipping)
        if np.dot(d_loc, prev_dir) < 0:
            d_loc = -d_loc

        # propose next point ahead
        prop = cur_pt + step * d_loc

        # snap to nearest neighbor ahead
        dist, nn_idx = kdt.query(prop, k=10, distance_upper_bound=2 * r_nbr)
        if np.isscalar(nn_idx):
            nn_idx = [nn_idx]
            dist = [dist]
        # filter invalid (when distance_upper_bound returns inf)
        cands = [idx for idx, di in zip(nn_idx, dist) if np.isfinite(di) and idx < N]
        if not cands:
            break

        # require forward progress relative to local direction
        best_idx, best_score = None, -1e9
        for ci in cands:
            v = xyz[ci] - cur_pt
            dot = float(np.dot(v, d_loc))
            if dot > ahead_dot_min * step and dot > best_score:
                best_idx, best_score = ci, dot
        if best_idx is None:
            break

        # Average small neighborhood around the snapped point to denoise
        den_idx = kdt.query_ball_point(xyz[best_idx], avg_nbr)
        cur_pt = xyz[den_idx].mean(axis=0) if len(den_idx) >= 3 else xyz[best_idx]
        prev_dir = d_loc
        cur_idx = best_idx
        if visited[cur_idx]:
            # loop detected
            break
        visited[cur_idx] = True

        # early stop if near the other endpoint
        if np.linalg.norm(cur_pt - xyz[i1]) < 3 * step:
            path.append(xyz[i1].copy())
            break

    path = np.array(path)
    if len(path) < 2:
        return np.empty((0, 3)), path
    # 1 cm resample
    resampled = arc_length_resample(path, ds=0.05)
    return resampled, path


class PCAPathNode(Node):
    def __init__(self):
        super().__init__("pca_path_node")
        # Params
        self.declare_parameter("cloud_topic", "/random_cloud")
        self.declare_parameter("fixed_frame", "world")
        self.declare_parameter("interval_m", 0.03)  # 1 cm
        self.declare_parameter("slab_half_thickness_m", 0.05)  # ±3cm
        self.declare_parameter("min_pts_per_slab", 3)

        cloud_topic = (
            self.get_parameter("cloud_topic").get_parameter_value().string_value
        )
        self.fixed_frame = (
            self.get_parameter("fixed_frame").get_parameter_value().string_value
        )
        self.interval_m = float(self.get_parameter("interval_m").value)
        self.slab_half = float(self.get_parameter("slab_half_thickness_m").value)
        self.min_pts = int(self.get_parameter("min_pts_per_slab").value)

        self.sub = self.create_subscription(PointCloud2, cloud_topic, self.cloud_cb, 10)
        self.path_pub = self.create_publisher(Path, "pca_path", 10)
        self.marker_pub = self.create_publisher(Marker, "pca_path_points", 10)

        self.get_logger().info(
            f"Listening on {cloud_topic}; publishing path on /pca_path (frame {self.fixed_frame})"
        )

    def cloud_cb(self, msg: PointCloud2):
        pts = np.array(
            [
                [x, y, z]
                for x, y, z in point_cloud2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
            ],
            dtype=np.float32,
        )
        resampled, raw = estimate_curved_path(pts, r_nbr=0.08, step=0.05, avg_nbr=0.03)

        path = Path()
        path.header.frame_id = self.fixed_frame  # e.g., "map"
        path.header.stamp = msg.header.stamp
        for p in resampled:
            ps = PoseStamped()
            ps.header.frame_id = path.header.frame_id
            ps.header.stamp = path.header.stamp
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
                float(p[0]),
                float(p[1]),
                float(p[2]),
            )
            ps.pose.orientation.w = (
                1.0  # or compute tangent-based orientation if you need it
            )
            path.poses.append(ps)
        self.path_pub.publish(path)

        self.get_logger().info(f"Published path with {len(path.poses)} poses.")

    # def cloud_cb(self, msg: PointCloud2):
    #     # Read XYZ (ignore NaNs)
    #     pts = []
    #     for p in point_cloud2.read_points(
    #         msg, field_names=("x", "y", "z"), skip_nans=True
    #     ):
    #         pts.append([p[0], p[1], p[2]])
    #     if not pts:
    #         self.get_logger().warn("Empty cloud")
    #         return
    #     xyz = np.asarray(pts, dtype=np.float32)

    #     avg_pts, d, mu = compute_principal_axis_path(
    #         xyz,
    #         interval_m=self.interval_m,
    #         slab_half_thickness_m=self.slab_half,
    #         min_pts_per_slab=self.min_pts,
    #     )

    #     # Publish Path
    #     path = Path()
    #     path.header.frame_id = self.fixed_frame
    #     path.header.stamp = msg.header.stamp  # keep close to source time

    #     if avg_pts.shape[0] == 0:
    #         self.path_pub.publish(path)
    #         self.get_logger().warn("No slabs had enough points; path is empty.")
    #         return

    #     # Orientation: x-axis aligned with principal direction d
    #     qx, qy, qz, qw = quaternion_from_two_vectors(np.array([1.0, 0.0, 0.0]), d)

    #     for i, p in enumerate(avg_pts):
    #         ps = PoseStamped()
    #         ps.header.frame_id = self.fixed_frame
    #         ps.header.stamp = msg.header.stamp
    #         ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
    #             float(p[0]),
    #             float(p[1]),
    #             float(p[2]),
    #         )
    #         (
    #             ps.pose.orientation.x,
    #             ps.pose.orientation.y,
    #             ps.pose.orientation.z,
    #             ps.pose.orientation.w,
    #         ) = (qx, qy, qz, qw)
    #         path.poses.append(ps)

    #     self.path_pub.publish(path)

    #     # Also publish points as a Marker for quick visual sanity check
    #     m = Marker()
    #     m.header.frame_id = self.fixed_frame
    #     m.header.stamp = msg.header.stamp
    #     m.ns = "pca_path"
    #     m.id = 0
    #     m.type = Marker.POINTS
    #     m.action = Marker.ADD
    #     m.scale.x = 0.01  # point size (m)
    #     m.scale.y = 0.01
    #     m.color.a = 1.0
    #     m.color.r = 0.1
    #     m.color.g = 0.9
    #     m.color.b = 0.1
    #     for p in avg_pts:
    #         from geometry_msgs.msg import Point

    #         pt = Point()
    #         pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
    #         m.points.append(pt)
    #     # Keep marker alive a bit
    #     m.lifetime = Duration(sec=0, nanosec=300_000_000)  # 0.3s
    #     self.marker_pub.publish(m)

    #     self.get_logger().info(f"Published path with {len(path.poses)} poses; dir={d}")


def main():
    rclpy.init()
    node = PCAPathNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
