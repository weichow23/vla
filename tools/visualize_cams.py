import os
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import json
from data_preprocessing.correspondences2poses import euler_to_rotation_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

def calc_cam_cone_pts_3d(c2w, fov_deg, zoom=0.5):
    fov_rad = np.deg2rad(fov_deg)

    cam_x = c2w[0, -1]
    cam_y = c2w[1, -1]
    cam_z = c2w[2, -1]

    # With forward direction along +z.
    corn1 = [np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), 1.0]
    corn2 = [-np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0), 1.0]
    corn3 = [-np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), 1.0]
    corn4 = [np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0), 1.0]
    corn5 = [0, np.tan(fov_rad / 2.0), 1.0]

    corn1 = np.dot(c2w[:3, :3], corn1)
    corn2 = np.dot(c2w[:3, :3], corn2)
    corn3 = np.dot(c2w[:3, :3], corn3)
    corn4 = np.dot(c2w[:3, :3], corn4)
    corn5 = np.dot(c2w[:3, :3], corn5)

    # Normalize and scale
    corn1 = np.array(corn1) / np.linalg.norm(corn1) * zoom
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    
    corn2 = np.array(corn2) / np.linalg.norm(corn2) * zoom
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    
    corn3 = np.array(corn3) / np.linalg.norm(corn3) * zoom
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    
    corn4 = np.array(corn4) / np.linalg.norm(corn4) * zoom
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]
    
    corn5 = np.array(corn5) / np.linalg.norm(corn5) * zoom
    corn_x5 = cam_x + corn5[0]
    corn_y5 = cam_y + corn5[1]
    corn_z5 = cam_z + corn5[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4, corn_x5]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4, corn_y5]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4, corn_z5]

    return np.array([xs, ys, zs]).T


class CameraVisualizer:
    def __init__(self, poses, legends, colors, images=None, mesh_path=None, camera_x=1.0):
        self._fig = None
        self._camera_x = camera_x        
        self._poses = poses
        self._legends = legends
        self._colors = colors

        self._raw_images = None
        self._bit_images = None
        self._image_colorscale = None

        if images is not None:
            self._raw_images = images
            self._bit_images = []
            self._image_colorscale = []
            for img in images:
                if img is None:
                    self._bit_images.append(None)
                    self._image_colorscale.append(None)
                    continue
                bit_img, colorscale = self.encode_image(img)
                self._bit_images.append(bit_img)
                self._image_colorscale.append(colorscale)

        self._mesh = None
        if mesh_path is not None and os.path.exists(mesh_path):
            import trimesh
            self._mesh = trimesh.load(mesh_path, force='mesh')

    def encode_image(self, raw_image):
        # Convert raw image to a palette image and generate a colorscale.
        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
        bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
        return bit_image, colorscale

    def update_figure(
            self, scene_bounds, 
            base_radius=0.0, zoom_scale=1.0, fov_deg=50., 
            mesh_z_shift=0.0, mesh_scale=1.0, 
            show_background=False, show_grid=False, show_ticklabels=False, y_up=False,
            label_mode="both"):
        """
        label_mode options:
          - "near": show camera names as text labels near the cameras (and remove legend entries)
          - "legend": show camera names only in the legend (no near text)
          - "both": show camera names both as near labels and in the legend.
        """
        fig = go.Figure()

        if self._mesh is not None:
            fig.add_trace(
                go.Mesh3d(
                    x=self._mesh.vertices[:, 0] * mesh_scale,  
                    y=self._mesh.vertices[:, 2] * -mesh_scale,  
                    z=(self._mesh.vertices[:, 1] + mesh_z_shift) * mesh_scale,  
                    i=self._mesh.faces[:, 0],
                    j=self._mesh.faces[:, 1],
                    k=self._mesh.faces[:, 2],
                    color=None,
                    facecolor=None,
                    opacity=0.8,
                    lighting={'ambient': 1},
                )
            )

        for cam_idx in tqdm(range(len(self._poses))):
            pose = self._poses[cam_idx]
            clr = self._colors[cam_idx]
            legend = self._legends[cam_idx]

            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (0, 5)]
            cone = calc_cam_cone_pts_3d(pose, fov_deg)
            
            # Draw the camera cone edges.
            for i, edge in enumerate(edges):
                (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                # Only include legend entries if label_mode is "legend" or "both"
                show_legend = (i == 0 and label_mode in ["legend", "both"])
                fig.add_trace(go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                    line=dict(color=clr, width=3),
                    name=legend, showlegend=show_legend))
            
            # Optionally add text labels near the camera.
            if label_mode in ["near", "both"]:
                if cone[0, 2] < 0:
                    text_position = 'bottom center'
                    text_z = cone[0, 2] - 0.05
                else:
                    text_position = 'top center'
                    text_z = cone[0, 2] + 0.05
                fig.add_trace(go.Scatter3d(
                    x=[cone[0, 0]], y=[cone[0, 1]], z=[text_z],
                    mode='text', text=legend, textposition=text_position, showlegend=False))

            # Optionally, add an image on the camera plane if provided.
            if self._bit_images and self._bit_images[cam_idx]:
                raw_image = self._raw_images[cam_idx]
                bit_image = self._bit_images[cam_idx]
                colorscale = self._image_colorscale[cam_idx]
                (H, W, C) = raw_image.shape
                z_plane = np.zeros((H, W)) + base_radius
                (x_plane, y_plane) = np.meshgrid(
                    np.linspace(-self._camera_x, self._camera_x, W),
                    np.linspace(1.0, -1.0, H) * H / W)
                xyz = np.concatenate([x_plane[..., None], y_plane[..., None], z_plane[..., None]], axis=-1)
                rot_xyz = np.matmul(xyz, pose[:3, :3].T) + pose[:3, -1]
                x_img, y_img, z_img = rot_xyz[:, :, 0], rot_xyz[:, :, 1], rot_xyz[:, :, 2]
                fig.add_trace(go.Surface(
                    x=x_img, y=y_img, z=z_img,
                    surfacecolor=bit_image,
                    cmin=0,
                    cmax=255,
                    colorscale=colorscale,
                    showscale=False,
                    lighting_diffuse=1.0,
                    lighting_ambient=1.0,
                    lighting_fresnel=1.0,
                    lighting_roughness=1.0,
                    lighting_specular=0.3))
            
        # Configure scene and layout.
        fig.update_layout(
            height=720,
            autosize=True,
            hovermode=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(yanchor='bottom', y=0.01, xanchor='right', x=0.99),
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0),
                            center=dict(x=0.0, y=0.0, z=0.0),
                            up=dict(x=0.0, y=0.0, z=1.0)),
                xaxis_title='X',
                yaxis_title='Z' if not y_up else 'Y',
                zaxis_title='Y' if not y_up else 'Z',
                xaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                yaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                zaxis=dict(
                    range=[-scene_bounds, scene_bounds],
                    showticklabels=show_ticklabels,
                    showgrid=show_grid,
                    zeroline=False,
                    showbackground=show_background,
                    showspikes=False,
                    showline=False,
                    ticks='')
            )
        )

        self._fig = fig
        return fig

def camera_from_dict(cam):
    """
    Given a dictionary with keys "r" (Euler angles) and "t" (translation),
    returns a 3x4 camera-to-world pose matrix.
    Assume the given pose is world-to-camera.
    """
    R_wc = euler_to_rotation_matrix(cam["r"])
    t_wc = np.array(cam["t"]).reshape((3, 1))
    
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc
    pose = np.concatenate([R_cw, t_cw], axis=1)
    return pose


# --- Example usage ---

if __name__ == '__main__':
    cam_metapath = 'data_preprocessing/meta_data/val/progress_dict.json'
    with open(cam_metapath, 'r') as f:
        cam_meta = json.load(f)

    cameras = []
    # subset = 100
    for idx, cam in enumerate(cam_meta.keys()):
        if "0" in cam_meta[cam] and cam_meta[cam]["0"] is not None:
            cameras.append(cam_meta[cam]["0"])
        # if len(cameras) >= subset:
        #     break
    
    # Convert each camera dictionary to a 3x4 pose matrix.
    poses = [camera_from_dict(cam) for cam in cameras]
    
    pose_errors = [cam["mean_proj_error"] for cam in cameras]
    scaled_pose_errors = (np.array(pose_errors) - np.min(pose_errors)) / (np.max(pose_errors) - np.min(pose_errors))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(val) for val in scaled_pose_errors]

    legends = [f"Camera {i}" for i in range(len(poses))]

    # Instantiate the visualizer.
    visualizer = CameraVisualizer(poses, legends, colors)
    
    # Update and display the figure.
    # Set label_mode to "legend", "near", or "both" depending on your preference.
    fig = visualizer.update_figure(scene_bounds=5.0, 
                                   base_radius=1,
                                   show_grid=True,
                                   show_ticklabels=True,
                                   show_background=True,
                                   fov_deg=50.,
                                   label_mode="none")
    fig.write_html('camera_viewer.html')
