import torch
from PIL import Image
import numpy as np
import ujson as json
import torch.nn.functional as F
import math

def get_coord3d(image, depth_path, cam_params_path):
    width, height = image.size

    try:
        depth_image = Image.open(depth_path)
        depth_image = np.array(depth_image, dtype=np.float32)
        depth_image = depth_image / 1000.0
        depth_image[depth_image == 0] = np.nan
    except Exception as exn:
        print(f"Warning: Failed to open depth image {depth_path}. Exception: {exn}")
        depth_image = np.full((height, width), np.nan, dtype=np.float32)

    try:
        with open(cam_params_path, "r") as f:
            cam_params = json.load(f)
    except Exception as exn:
        print(f"Warning: Failed to open camera parameters {cam_params_path}. Exception: {exn}")
        fx = fy = max(width, height)
        cx = width / 2.0
        cy = height / 2.0
        cam_params = {
            "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0]
        }

    K = np.array(cam_params["intrinsics"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) * depth_image / fx
    y = (v - cy) * depth_image / fy
    z = depth_image
    coord3d = np.stack((x, y, z), axis=2)

    R = np.array(cam_params["rotation"])
    T = np.array(cam_params["translation"])
    coord3d_flat = coord3d.reshape(-1, 3).T
    coord3d_world = R @ coord3d_flat + T[:, np.newaxis]
    coord3d = coord3d_world.T.reshape(height, width, 3)
    coord3d[:, :, -1] *= -1 
    return torch.tensor(coord3d, dtype=torch.float32)

def resize_coord3d_resize(coord3d, new_size, mode='bicubic'):
    coord3d_perm = coord3d.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    coord3d_interp = F.interpolate(coord3d_perm, size=new_size, mode=mode, align_corners=True)
    coord3d_resized = coord3d_interp.squeeze(0).permute(1, 2, 0)
    return coord3d_resized

def get_coord3d_info(image_path, depth_path, cam_params_path, image, interp_mode='bicubic'):
    new_width, new_height = image.size
    coord3d = get_coord3d(Image.open(image_path), depth_path, cam_params_path)
    coord3d_resized = resize_coord3d_resize(coord3d, new_size=(new_height, new_width), mode=interp_mode)
    return coord3d_resized

def coord3d_to_flat_patches(coord3d_resized, patch_size, merge_size, temporal_patch_size=1, is_return_grid_size=False):
    resized_height, resized_width, _ = coord3d_resized.shape

    patches = np.expand_dims(coord3d_resized.numpy(), axis=0)  # shape: (1, H, W, 3)
    patches = patches.transpose(0, 3, 1, 2)

    T = patches.shape[0]
    remainder = T % temporal_patch_size
    if remainder != 0:
        repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - remainder, axis=0)
        patches = np.concatenate([patches, repeats], axis=0)

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    grid_size = (grid_t, grid_h, grid_w)

    if is_return_grid_size:
        return flatten_patches, grid_size
    else:
        return flatten_patches

def generate_3D_positional_encoding(coordinates, channels, grid_dims, scale=20.0):
    device = coordinates.device
    dtype = coordinates.dtype
    pt = 2
    pc = 3
    ps = 14
    vec_len = pt * pc * ps * ps
    if coordinates.shape[1] != vec_len:
        raise ValueError(f"Expected coordinate vector length {vec_len}, got {coordinates.shape[1]}.")
    L = channels // 6
    if L * 6 != channels:
        raise ValueError("num_channels must be divisible by 6.")
    freq = torch.linspace(1.0, scale, steps=L, device=device, dtype=torch.float32).view(L, 1, 1, 1)
    out_encodings = []
    out_nan_masks = []
    counts = []
    sizes = []
    for i in range(grid_dims.shape[0]):
        t_dim, w_dim, h_dim = grid_dims[i].tolist()
        cnt = int(t_dim * w_dim * h_dim * ps * ps)
        counts.append(cnt)
        sizes.append((int(h_dim * ps), int(w_dim * ps)))
    coordinates = coordinates.view(-1, pt * pc)
    start = 0
    for cnt, size in zip(counts, sizes):
        end = start + cnt
        grid_coords = coordinates[start:end]
        grid_coords = grid_coords.view(cnt * pt, pc)
        grid_coords = grid_coords.view(-1, pc, *size)
        mask = torch.isnan(grid_coords)
        nan_mask = mask.any(dim=1)
        grid_coords = grid_coords.masked_fill(mask, 0)
        coord_scaled = grid_coords.unsqueeze(1).float() * freq * 2 * math.pi
        sin_vals = torch.sin(coord_scaled)
        cos_vals = torch.cos(coord_scaled)
        sincos = torch.stack((sin_vals, cos_vals), dim=2)
        sincos = sincos.permute(0, 3, 2, 1, 4, 5)
        pe = sincos.reshape(pt, channels, *size).to(dtype=dtype)
        nan_mask_exp = nan_mask.unsqueeze(1).expand_as(pe)
        pe = pe.masked_fill(nan_mask_exp, 0)
        nan_mask = nan_mask.to(dtype=dtype)
        pe = pe.view(-1, pt * ps * ps, channels)
        nan_mask = nan_mask.view(-1, pt * ps * ps)
        out_encodings.append(pe)
        out_nan_masks.append(nan_mask)
        start = end
    return torch.cat(out_encodings, dim=0), torch.cat(out_nan_masks, dim=0)

