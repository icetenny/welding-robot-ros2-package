#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# Optional fast KDTree if SciPy is present; otherwise fallback later
try:
    from scipy.spatial import cKDTree as KDTree

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def orthonormal_from_x_and_camera(p0, x_dir, cam):
    x = x_dir / (np.linalg.norm(x_dir) + 1e-12)
    w = cam - p0
    # remove x component to get in-plane camera direction
    y = w - x * (w @ x)
    if np.linalg.norm(y) < 1e-9:
        # camera exactly on the tangent line; pick a stable fallback
        h = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(x @ h) > 0.95:
            h = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        y = h - x * (h @ x)
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    return x, y, z  # plane is spanned by (x,y), normal z


def pca_dir2d(Puv):
    # Puv: (M,2)
    Pc = Puv - Puv.mean(axis=0, keepdims=True)
    C = Pc.T @ Pc
    w, V = np.linalg.eigh(C)
    d = V[:, np.argmax(w)]
    d = d / (np.linalg.norm(d) + 1e-12)
    return d  # (2,)


def quat_from_rotm(R):
    """Convert 3x3 rotation matrix to geometry_msgs/Quaternion (x,y,z,w)."""
    # Robust branchless quaternion from matrix
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
                R[0, 0] + R[1, 1] + R[2, 2],
            ],
        ]
    )
    K = K / 3.0
    # Eigenvector of K with largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[:, np.argmax(w)]
    # (x,y,z,w) ordering
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def rgbfloat_to_uint32(rgb_f: np.ndarray) -> np.ndarray:
    return rgb_f.view(np.uint32)


def uint32_to_rgb(uint32_arr: np.ndarray) -> np.ndarray:
    r = (uint32_arr >> 16) & 0xFF
    g = (uint32_arr >> 8) & 0xFF
    b = uint32_arr & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def rgb_to_hsv_np(rgb_uint8: np.ndarray) -> np.ndarray:
    """Vectorized RGB(0..255) -> HSV (h in [0,1), s,v in [0,1])."""
    rgb = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-12

    h = np.empty_like(maxc)
    mask_r = maxc == r
    mask_g = maxc == g
    mask_b = maxc == b
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    h = (h / 6.0) % 1.0

    s = np.where(maxc > 0.0, delta / maxc, 0.0)
    v = maxc
    return np.stack([h, s, v], axis=-1)


def debug_nan_inf(msg):
    # ---- Read raw (with NaNs kept) ----
    pts_all = point_cloud2.read_points_numpy(
        msg, field_names=("x", "y", "z", "rgb"), skip_nans=False
    )
    xyz = pts_all[:, :3]
    rgb_float = pts_all[:, 3].astype(np.float32)

    rgb_float = pts_all[:, -1].astype(np.float32)
    # Decode packed RGB
    rgb_u32 = rgbfloat_to_uint32(rgb_float)
    rgb = uint32_to_rgb(rgb_u32)  # Nx3 uint8

    # NaN / Inf masks
    xyz_nan = np.isnan(xyz).any(axis=1)
    xyz_inf = np.isinf(xyz).any(axis=1)
    rgb_nan = np.isnan(rgb).any(axis=1)
    rgb_inf = np.isinf(rgb).any(axis=1)

    both_nan = xyz_nan | rgb_nan
    both_inf = xyz_inf | rgb_inf

    total = xyz.shape[0]

    print("---- skip_nans = False ----")
    print(f"Total points: {total}")
    print(f"xyz NaN: {xyz_nan.sum()} | xyz Inf: {xyz_inf.sum()}")
    print(f"rgb NaN: {rgb_nan.sum()} | rgb Inf: {rgb_inf.sum()}")
    print(f"Both NaN: {both_nan.sum()} | Both Inf: {both_inf.sum()}")
    print(f"Valid points: {total - (both_nan | both_inf).sum()}")

    # ---- Read again with skip_nans=True ----
    pts_valid = point_cloud2.read_points_numpy(
        msg, field_names=("x", "y", "z", "rgb"), skip_nans=True
    )
    print("---- skip_nans = True ----")
    print(f"Total valid points (ROS filtered): {pts_valid.shape[0]}")
    print(
        f"Removed: {total - pts_valid.shape[0]} ({(total - pts_valid.shape[0]) / total * 100:.2f}%)"
    )


def read_pointcloud(msg):
    # ---- Read raw (with NaNs kept) ----
    pts_all = point_cloud2.read_points_numpy(
        msg, field_names=("x", "y", "z", "rgb"), skip_nans=False
    )
    xyz = pts_all[:, :3]
    rgb_float = pts_all[:, 3].astype(np.float32)

    rgb_float = pts_all[:, -1].astype(np.float32)
    # Decode packed RGB
    rgb_u32 = rgbfloat_to_uint32(rgb_float)
    rgb = uint32_to_rgb(rgb_u32)  # Nx3 uint8

    # NaN / Inf masks
    xyz_nan = np.isnan(xyz).any(axis=1)
    xyz_inf = np.isinf(xyz).any(axis=1)
    rgb_nan = np.isnan(rgb).any(axis=1)
    rgb_inf = np.isinf(rgb).any(axis=1)

    both_nan = xyz_nan | rgb_nan
    both_inf = xyz_inf | rgb_inf

    valid_mask = ~(both_inf | both_nan)
    # print(valid_mask.shape)

    return xyz[valid_mask], rgb[valid_mask]


class WeldingPathNode(Node):
    def __init__(self):
        super().__init__("welding_path_node")

        # ---- Parameters you may tweak ----
        self.declare_parameter(
            "input_cloud_topic", "/zed/zed_node/point_cloud/cloud_registered"
        )
        self.declare_parameter("filtered_cloud_topic", "/filtered_cloud")
        self.declare_parameter("path_topic", "/red_path")
        self.declare_parameter("pose_array_topic", "/weld_poses")

        # HSV thresholds for RED (wraparound)
        self.declare_parameter("h_low_deg", 25.0)
        self.declare_parameter("h_high_deg", 335.0)
        self.declare_parameter("s_min", 0.2)
        self.declare_parameter("v_min", 0.2)

        # Clustering
        self.declare_parameter("cluster_radius", 0.02)  # 2 cm
        self.declare_parameter("cluster_min_pts", 50)

        # Path resampling step (meters)
        self.declare_parameter("path_step", 0.01)  # 1 cm

        # Read params
        in_topic = (
            self.get_parameter("input_cloud_topic").get_parameter_value().string_value
        )
        self.filtered_topic = (
            self.get_parameter("filtered_cloud_topic")
            .get_parameter_value()
            .string_value
        )
        self.path_topic = (
            self.get_parameter("path_topic").get_parameter_value().string_value
        )
        self.pose_array_topic = (
            self.get_parameter("pose_array_topic").get_parameter_value().string_value
        )

        self.h_low = (
            self.get_parameter("h_low_deg").get_parameter_value().double_value / 360.0
        )
        self.h_high = (
            self.get_parameter("h_high_deg").get_parameter_value().double_value / 360.0
        )
        self.s_min = self.get_parameter("s_min").get_parameter_value().double_value
        self.v_min = self.get_parameter("v_min").get_parameter_value().double_value

        self.cluster_radius = (
            self.get_parameter("cluster_radius").get_parameter_value().double_value
        )
        self.cluster_min_pts = int(
            self.get_parameter("cluster_min_pts").get_parameter_value().integer_value
        )
        self.path_step = (
            self.get_parameter("path_step").get_parameter_value().double_value
        )

        # IO
        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb_cloud, 5)
        self.pub_filtered = self.create_publisher(PointCloud2, self.filtered_topic, 5)
        self.pub_path = self.create_publisher(Path, self.path_topic, 5)
        self.pub_poses = self.create_publisher(PoseArray, self.pose_array_topic, 5)

        self.get_logger().info(
            f"✅ welding_path_node started. Sub: {in_topic} | "
            f"Pub: {self.filtered_topic}, {self.path_topic}, {self.pose_array_topic} | "
            f'{"SciPy KDTree ON" if SCIPY_OK else "SciPy not found (fallback clustering)"}'
        )

    def cb_cloud(self, msg: PointCloud2):
        # ---- Load cloud to structured numpy array ----
        # debug_nan_inf(msg=msg)
        xyz, rgb = read_pointcloud(msg)
        # print(xyz.shape, rgb.shape)
        # ---- HSV red segmentation (robust to brightness) ----
        hsv = rgb_to_hsv_np(rgb)  # (N,3) float32
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        red_hue = (h < self.h_low) | (h > self.h_high)
        red_mask = red_hue & (s >= self.s_min) & (v >= self.v_min)

        # ---- Publish filtered cloud: red as-is, non-red → white ----
        rgb_out = rgb.copy()
        rgb_out[~red_mask] = np.array([255, 255, 255], dtype=np.uint8)
        packed = (
            (rgb_out[:, 0].astype(np.uint32) << 16)
            | (rgb_out[:, 1].astype(np.uint32) << 8)
            | rgb_out[:, 2].astype(np.uint32)
        )
        rgb_new_float = packed.view(np.float32)
        cloud_out = np.column_stack((xyz, rgb_new_float)).astype(np.float32)
        self.pub_filtered.publish(
            point_cloud2.create_cloud(msg.header, msg.fields, cloud_out)
        )

        # ---- Extract red-only points for path ----
        red_xyz = xyz[red_mask]
        if red_xyz.shape[0] < self.cluster_min_pts:
            self.get_logger().warn("Not enough red points to form a path.")
            return

        # ---- Keep the largest spatial cluster (remove scattered noise) ----
        red_xyz = self._largest_cluster(
            red_xyz, self.cluster_radius, self.cluster_min_pts
        )
        if red_xyz is None:
            self.get_logger().warn("No cluster passed the min_pts threshold.")
            return

        # ---- PCA to get principal direction, then 1D resample at 1 cm ----
        mean = red_xyz.mean(axis=0)
        Xc = red_xyz - mean
        # SVD for principal direction
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        v_dir = Vt[0]  # principal axis (unit)

        # project points onto the axis -> sort by scalar projection
        s = Xc @ v_dir
        s_min, s_max = float(s.min()), float(s.max())
        if s_max - s_min < 1e-6:
            self.get_logger().warn("Degenerate PCA span for red path.")
            return
        step = max(self.path_step, 1e-3)
        s_grid = np.arange(s_min, s_max + step * 0.5, step)
        path_pts = mean + np.outer(s_grid, v_dir)  # (M,3)

        # ---- Publish path (nav_msgs/Path) ----
        path_msg = Path()
        path_msg.header = msg.header
        for p in path_pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
                float(p[0]),
                float(p[1]),
                float(p[2]),
            )
            # orientation not used in Path; leave identity
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)

        # ---- Precompute sums for "furthest z-axis" per point ----
        # We use ALL points (including non-red) as requested.
        N = xyz.shape[0]
        S1 = xyz.sum(axis=0)  # Σ P_i  (3,)
        S2 = xyz.T @ xyz  # Σ P_i P_i^T  (3x3)

        # ---- Build PoseArray: x = path tangent, z = smallest-eigvec(C(P0)) ----

        # Tunables (could be ROS params)
        slab_thickness = 0.02  # meters (e.g., 2 cm)
        slab_radius = 0.10  # meters (e.g., 10 cm)
        min_side_pts = 30  # minimum points per side for reliable angle

        angles_left = []
        angles_right = []
        cam_center = np.array([0, 0, 0])

        for i, p0 in enumerate(path_pts):
            # path tangent
            x_dir = (
                (path_pts[i + 1] - p0)
                if i < len(path_pts) - 1
                else (p0 - path_pts[i - 1])
            )
            x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)

            # Build plane from camera and path tangent
            x_hat, y_hat, z_hat = orthonormal_from_x_and_camera(
                p0, x_dir, cam_center
            )  # cam_center in same frame as xyz

            # Project neighbors to plane coords and select a thin slab
            D = xyz - p0  # (N,3) all points relative to p0
            d_n = D @ z_hat  # distance to plane (normal)
            u = D @ x_hat  # along-path in-plane coord
            v = D @ y_hat  # cross-path in-plane coord

            in_slab = (np.abs(d_n) <= 0.5 * slab_thickness) & (
                u * u + v * v <= slab_radius * slab_radius
            )

            # Split left/right by sign of v
            Puv = np.stack([u[in_slab], v[in_slab]], axis=1)
            vside = Puv[:, 1] if Puv.shape[0] else np.empty((0,))
            left_mask = vside > 0
            right_mask = vside < 0

            # Default “no angle”
            ang_left = np.nan
            ang_right = np.nan

            # Left side angle
            if np.count_nonzero(left_mask) >= min_side_pts:
                d = pca_dir2d(Puv[left_mask])  # (du, dv)
                # enforce pointing roughly forward along +u
                if d[0] < 0:
                    d = -d
                ang_left = float(
                    abs(np.arctan2(d[1], d[0]))
                )  # radians, [0, pi/2] typically

            # Right side angle
            if np.count_nonzero(right_mask) >= min_side_pts:
                d = pca_dir2d(Puv[right_mask])
                if d[0] < 0:
                    d = -d
                ang_right = float(abs(np.arctan2(d[1], d[0])))

            angles_left.append(ang_left)
            angles_right.append(ang_right)

            # (Optional) You can also orient your weld pose’s z-axis along the in-plane bisector
            # between the two side directions if both are valid, or follow your previous z rule.

        poses = PoseArray()
        poses.header = msg.header

        prev_z = None
        for i, p0 in enumerate(path_pts):
            # x-axis: path tangent
            if i < len(path_pts) - 1:
                x_dir = path_pts[i + 1] - p0
            else:
                x_dir = p0 - path_pts[i - 1]
            x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)

            # Build camera/path plane basis inline (no functions):
            # x_hat along path; y_hat = camera direction projected onto plane ⟂ x_hat
            w = cam_center - p0
            y_hat = w - x_dir * (w @ x_dir)
            if np.linalg.norm(y_hat) < 1e-9:
                h = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                if abs(x_dir @ h) > 0.95:
                    h = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                y_hat = h - x_dir * (h @ x_dir)
            y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-12)
            z_hat_cam = np.cross(x_dir, y_hat)
            z_hat_cam = z_hat_cam / (np.linalg.norm(z_hat_cam) + 1e-12)

            # Get left/right angles (radians) for this path point
            aL = angles_left[i]
            aR = angles_right[i]

            # z-axis: bisector between left and right in the camera/path plane
            if np.isnan(aL) or np.isnan(aR):
                z_dir = z_hat_cam  # fallback if one side missing
            else:
                # 2D directions in (u,v) plane: +u is x_dir, +v is y_hat
                dL = np.array([np.cos(aL), np.sin(aL)], dtype=np.float32)
                dR = np.array(
                    [np.cos(aR), -np.sin(aR)], dtype=np.float32
                )  # mirrored across centerline
                if dL[0] < 0:
                    dL = -dL
                if dR[0] < 0:
                    dR = -dR
                bis2d = dL + dR
                if np.linalg.norm(bis2d) < 1e-9:
                    bis2d = np.array(
                        [-dL[1], dL[0]], dtype=np.float32
                    )  # perpendicular fallback
                bis2d = bis2d / (np.linalg.norm(bis2d) + 1e-12)
                z_dir = bis2d[0] * x_dir + bis2d[1] * y_hat
                z_dir = z_dir / (np.linalg.norm(z_dir) + 1e-12)

            # Keep orientation continuous along path
            if prev_z is not None and (prev_z @ z_dir) > 0:
                z_dir = -z_dir
            prev_z = z_dir.copy()

            # Re-orthogonalize: make x ⟂ z, then y = z × x
            x_ortho = x_dir - z_dir * (x_dir @ z_dir)
            x_ortho = x_ortho / (np.linalg.norm(x_ortho) + 1e-12)
            y_dir = np.cross(z_dir, x_ortho)
            y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)

            # Rotation matrix columns = [x y z]
            R = np.column_stack([x_ortho, y_dir, z_dir])
            q = quat_from_rotm(R)  # assumes you already have this util

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (
                float(p0[0]),
                float(p0[1]),
                float(p0[2]),
            )
            pose.orientation = q
            poses.poses.append(pose)

        self.pub_poses.publish(poses)

    # ------------- helpers -------------
    def _largest_cluster(self, pts: np.ndarray, radius: float, min_pts: int):
        """
        Return largest cluster of pts using radius connectivity.
        If SciPy available -> fast KDTree; otherwise slow O(N^2) BFS.
        """
        N = pts.shape[0]
        if N == 0:
            return None

        if SCIPY_OK:
            tree = KDTree(pts)
            visited = np.zeros(N, dtype=bool)
            best_idx = None
            best_size = 0
            for i in range(N):
                if visited[i]:
                    continue
                # BFS over neighbor graph
                cluster = []
                stack = [i]
                visited[i] = True
                while stack:
                    j = stack.pop()
                    cluster.append(j)
                    idxs = tree.query_ball_point(pts[j], r=radius)
                    for k in idxs:
                        if not visited[k]:
                            visited[k] = True
                            stack.append(k)
                if len(cluster) >= min_pts and len(cluster) > best_size:
                    best_size = len(cluster)
                    best_idx = np.array(cluster, dtype=int)
            if best_idx is None:
                return None
            return pts[best_idx]

        # Fallback O(N^2) connectivity (small/medium N)
        visited = np.zeros(N, dtype=bool)
        best_idx = None
        best_size = 0
        r2 = radius * radius
        for i in range(N):
            if visited[i]:
                continue
            cluster = []
            stack = [i]
            visited[i] = True
            while stack:
                j = stack.pop()
                cluster.append(j)
                # naive neighbor search
                d2 = np.sum((pts - pts[j]) ** 2, axis=1)
                neigh = np.where((d2 <= r2) & (~visited))[0]
                for k in neigh:
                    visited[k] = True
                    stack.append(k)
            if len(cluster) >= min_pts and len(cluster) > best_size:
                best_size = len(cluster)
                best_idx = np.array(cluster, dtype=int)
        if best_idx is None:
            return None
        return pts[best_idx]


def main(args=None):
    rclpy.init(args=args)
    node = WeldingPathNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
