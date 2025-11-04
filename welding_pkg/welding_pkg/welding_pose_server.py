#!/usr/bin/env python3
import numpy as np
import rclpy
from custom_srv_pkg.srv import WeldPose
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
    y = w - x * (w @ x)
    if np.linalg.norm(y) < 1e-9:
        h = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(x @ h) > 0.95:
            h = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        y = h - x * (h @ x)
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    return x, y, z


def pca_dir2d(Puv):
    Pc = Puv - Puv.mean(axis=0, keepdims=True)
    C = Pc.T @ Pc
    w, V = np.linalg.eigh(C)
    d = V[:, np.argmax(w)]
    d = d / (np.linalg.norm(d) + 1e-12)
    return d


def quat_from_rotm(R):
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
        ]
    )
    K = K / 3.0
    w, V = np.linalg.eigh(K)
    q = V[:, np.argmax(w)]
    return Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def rgbfloat_to_uint32(rgb_f: np.ndarray) -> np.ndarray:
    return rgb_f.view(np.uint32)


def uint32_to_rgb(uint32_arr: np.ndarray) -> np.ndarray:
    r = (uint32_arr >> 16) & 0xFF
    g = (uint32_arr >> 8) & 0xFF
    b = uint32_arr & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def rgb_to_hsv_np(rgb_uint8: np.ndarray) -> np.ndarray:
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


def read_pointcloud(msg: PointCloud2):
    pts_all = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False)
    xyz = pts_all[:, :3]
    rgb_float = pts_all[:, -1].astype(np.float32)
    rgb_u32 = rgbfloat_to_uint32(rgb_float)
    rgb = uint32_to_rgb(rgb_u32)

    xyz_nan = np.isnan(xyz).any(axis=1)
    xyz_inf = np.isinf(xyz).any(axis=1)
    rgb_nan = np.isnan(rgb).any(axis=1)
    rgb_inf = np.isinf(rgb).any(axis=1)
    valid_mask = ~( (xyz_nan | rgb_nan) | (xyz_inf | rgb_inf) )
    return xyz[valid_mask], rgb[valid_mask]


class WeldingPathSrv(Node):
    def __init__(self):
        super().__init__("welding_path_srv")

        # ---- Parameters (same semantics as your node) ----
        self.declare_parameter("h_low_deg", 25.0)
        self.declare_parameter("h_high_deg", 335.0)
        self.declare_parameter("s_min", 0.2)
        self.declare_parameter("v_min", 0.2)
        self.declare_parameter("cluster_radius", 0.02)    # 2 cm
        self.declare_parameter("cluster_min_pts", 20)
        self.declare_parameter("path_step", 0.01)         # 1 cm
        self.declare_parameter("slab_thickness", 0.02)    # 2 cm
        self.declare_parameter("slab_radius", 0.10)       # 10 cm
        self.declare_parameter("min_side_pts", 5)
        self.declare_parameter("camera_center_xyz", [0.0, 0.0, 0.0])

        self.h_low = self.get_parameter("h_low_deg").value / 360.0
        self.h_high = self.get_parameter("h_high_deg").value / 360.0
        self.s_min = float(self.get_parameter("s_min").value)
        self.v_min = float(self.get_parameter("v_min").value)
        self.cluster_radius = float(self.get_parameter("cluster_radius").value)
        self.cluster_min_pts = int(self.get_parameter("cluster_min_pts").value)
        self.path_step = float(self.get_parameter("path_step").value)
        self.slab_thickness = float(self.get_parameter("slab_thickness").value)
        self.slab_radius = float(self.get_parameter("slab_radius").value)
        self.min_side_pts = int(self.get_parameter("min_side_pts").value)
        self.cam_center = np.array(self.get_parameter("camera_center_xyz").value, dtype=np.float32)

        self.srv = self.create_service(WeldPose, "compute_weld_poses", self.handle_request)

        self.get_logger().info(
            f"✅ welding_path_srv ready | SciPy KDTree: {'ON' if SCIPY_OK else 'OFF'}"
        )

    # ---------------- service callback ----------------
    def handle_request(self, request: WeldPose.Request, response: WeldPose.Response):
        msg = request.pointcloud

        self.get_logger().info("Receive MSG")
        # 1) Read cloud
        xyz, rgb = read_pointcloud(msg)
        if xyz.shape[0] == 0:
            self.get_logger().warn("Empty/invalid point cloud.")
            response.poses = PoseArray(header=msg.header, poses=[])
            return response

        # 2) RED segmentation (HSV, wrap-around hue)
        hsv = rgb_to_hsv_np(rgb)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        red_hue = (h < self.h_low) | (h > self.h_high)
        red_mask = red_hue & (s >= self.s_min) & (v >= self.v_min)
        red_xyz = xyz[red_mask]

        if red_xyz.shape[0] < self.cluster_min_pts:
            self.get_logger().warn("Not enough red points to form a path.")
            response.poses = PoseArray(header=msg.header, poses=[])
            return response

        # 3) Largest cluster
        red_xyz = self._largest_cluster(red_xyz, self.cluster_radius, self.cluster_min_pts)
        if red_xyz is None or red_xyz.shape[0] < self.cluster_min_pts:
            self.get_logger().warn("No cluster passed the min_pts threshold.")
            response.poses = PoseArray(header=msg.header, poses=[])
            return response

        # 4) PCA axis + 1 cm resample
        mean = red_xyz.mean(axis=0)
        Xc = red_xyz - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        v_dir = Vt[0]
        s = Xc @ v_dir
        s_min, s_max = float(s.min()), float(s.max())
        if s_max - s_min < 1e-6:
            self.get_logger().warn("Degenerate PCA span for red path.")
            response.poses = PoseArray(header=msg.header, poses=[])
            return response

        step = max(self.path_step, 1e-3)
        s_grid = np.arange(s_min, s_max + step * 0.5, step)
        path_pts = mean + np.outer(s_grid, v_dir)

        # 5) Precompute left/right side angles for each path point
        slab_th = self.slab_thickness
        slab_r = self.slab_radius
        min_side = self.min_side_pts
        cam_center = self.cam_center

        angles_left = []
        angles_right = []
        for i, p0 in enumerate(path_pts):
            # path tangent
            x_dir = (path_pts[i + 1] - p0) if i < len(path_pts) - 1 else (p0 - path_pts[i - 1])
            x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)

            # camera/path plane
            x_hat, y_hat, z_hat = orthonormal_from_x_and_camera(p0, x_dir, cam_center)

            # neighbors in thin slab around plane & radius
            D = xyz - p0
            d_n = D @ z_hat
            u = D @ x_hat
            v = D @ y_hat
            in_slab = (np.abs(d_n) <= 0.5 * slab_th) & (u * u + v * v <= slab_r * slab_r)

            Puv = np.stack([u[in_slab], v[in_slab]], axis=1) if np.count_nonzero(in_slab) > 0 else np.empty((0,2))
            if Puv.shape[0] == 0:
                angles_left.append(np.nan)
                angles_right.append(np.nan)
                continue

            vside = Puv[:, 1]
            left_mask = vside > 0
            right_mask = vside < 0

            # defaults
            ang_left = np.nan
            ang_right = np.nan

            if np.count_nonzero(left_mask) >= min_side:
                d = pca_dir2d(Puv[left_mask])
                if d[0] < 0: d = -d
                ang_left = float(abs(np.arctan2(d[1], d[0])))

            if np.count_nonzero(right_mask) >= min_side:
                d = pca_dir2d(Puv[right_mask])
                if d[0] < 0: d = -d
                ang_right = float(abs(np.arctan2(d[1], d[0])))

            angles_left.append(ang_left)
            angles_right.append(ang_right)

        print(angles_left, angles_right)

        # 6) Build PoseArray (x = path tangent, z = bisector in camera/path plane)
        poses = PoseArray()
        poses.header = msg.header
        prev_z = None

        for i, p0 in enumerate(path_pts):
            # x-axis (path tangent)
            x_dir = (path_pts[i + 1] - p0) if i < len(path_pts) - 1 else (p0 - path_pts[i - 1])
            x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)

            # camera/path plane basis (inline)
            w = cam_center - p0
            y_hat = w - x_dir * (w @ x_dir)
            if np.linalg.norm(y_hat) < 1e-9:
                h = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                if abs(x_dir @ h) > 0.95:
                    h = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                y_hat = h - x_dir * (h @ x_dir)
            y_hat = y_hat / (np.linalg.norm(y_hat) + 1e-12)
            z_hat_cam = - np.cross(x_dir, y_hat)
            z_hat_cam = z_hat_cam / (np.linalg.norm(z_hat_cam) + 1e-12)

            aL = angles_left[i]
            aR = angles_right[i]

            if np.isnan(aL) or np.isnan(aR):
                z_dir = z_hat_cam
            else:
                dL = np.array([np.cos(aL), np.sin(aL)], dtype=np.float32)
                dR = np.array([np.cos(aR), -np.sin(aR)], dtype=np.float32)
                if dL[0] < 0: dL = -dL
                if dR[0] < 0: dR = -dR
                bis2d = dL + dR
                if np.linalg.norm(bis2d) < 1e-9:
                    bis2d = np.array([-dL[1], dL[0]], dtype=np.float32)
                bis2d = bis2d / (np.linalg.norm(bis2d) + 1e-12)
                z_dir = bis2d[0] * x_dir + bis2d[1] * y_hat
                z_dir = z_dir / (np.linalg.norm(z_dir) + 1e-12)


            z_ortho = z_dir - x_dir * (z_dir @ x_dir)
            z_ortho = z_ortho / (np.linalg.norm(z_ortho) + 1e-12)

            if prev_z is None:
                if (z_ortho @ w) <= 0:
                    z_ortho = -z_ortho
            z_ortho = -z_ortho
            
            if prev_z is not None and (prev_z @ z_ortho) < 0:
                z_ortho = -z_ortho
            prev_z = z_ortho.copy()
            
            y_dir = np.cross(z_ortho, x_dir)
            y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)


            R = np.column_stack([x_dir, y_dir, z_ortho])
            q = quat_from_rotm(R)

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(p0[0]), float(p0[1]), float(p0[2])
            pose.orientation = q
            poses.poses.append(pose)

        response.poses = poses
        self.get_logger().info(f"Returning Weld Poses of length: {len(response.poses.poses)}")
        return response

    # ------------- helpers -------------
    def _largest_cluster(self, pts: np.ndarray, radius: float, min_pts: int):
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
    node = WeldingPathSrv()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
