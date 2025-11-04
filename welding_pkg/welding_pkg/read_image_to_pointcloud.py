#!/usr/bin/env python3
import cv2
import numpy as np
import open3d as o3d


def get_point_cloud_from_depth(depth, K, bbox=None):
    cam_fx, cam_fy, cam_cx, cam_cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    im_H, im_W = depth.shape
    xmap, ymap = np.meshgrid(np.arange(im_W), np.arange(im_H))

    if bbox is not None:
        rmin, rmax, cmin, cmax = bbox
        depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
        xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
        ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

    z = depth.astype(np.float32)
    x = (xmap - cam_cx) * z / cam_fx
    y = (ymap - cam_cy) * z / cam_fy
    cloud = np.stack([x, y, z], axis=-1)
    return cloud  # (H,W,3)


def create_open3d_pcd(rgb, depth, K, bbox=None):
    cloud = get_point_cloud_from_depth(depth, K, bbox)
    mask = (cloud[..., 2] > 0) & np.isfinite(cloud[..., 2])
    pts = cloud[mask]

    colors = (rgb.astype(np.float32) / 255.0)[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(pts.shape, cloud.shape, mask.shape)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


if __name__ == "__main__":
    # --- Load images ---
    rgb = cv2.cvtColor(
        cv2.imread(
            "/home/icetenny/senior-2/results/Run-2025-10-17-12-37-46-92/Cap-14-41-21-20/rgb.png"
        ),
        cv2.COLOR_BGR2RGB,
    )
    depth = cv2.imread(
        "/home/icetenny/senior-2/results/Run-2025-10-17-12-37-46-92/Cap-14-41-21-20/depth.png",
        cv2.IMREAD_UNCHANGED,
    ).astype(np.float32)
    # Depth units: if millimeters, convert to meters
    if depth.max() > 1000:
        depth /= 1000.0

    # --- Intrinsics (example; replace with your camera K) ---
    K = np.array(
        [[521.0, 0.0, 549.0], [0.0, 541.0, 312.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    # --- Create Open3D point cloud ---
    pcd = create_open3d_pcd(rgb, depth, K)

    # --- Visualize and/or save ---
    o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud("output_cloud.ply", pcd)
    # print(
    #     "✅ Saved to output_cloud.ply with", np.asarray(pcd.points).shape[0], "points"
    # )
