import os
import re
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
import torch

def get_log_time_path(
    log_path: str, run_dir:str = ".*"
) -> str:

    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [
            os.path.join(log_path, run.name) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        # sort matched runs by alphabetical order (latest run should be last)
        runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        run_path = runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    return run_path

def get_checkpoint_path(log_path: str, checkpoint: str) -> str:
    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(log_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{log_path}' match '{checkpoint}'.")
    if checkpoint == "model_.*.zip":
        model_checkpoints.sort(key=lambda m: int(re.search(r"\d+", m).group()))
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(log_path, checkpoint_file)

def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
         "frame": sim_utils.UsdFileCfg(
             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
             scale=(0.5, 0.5, 0.5),
         ),
        }
    )
    return VisualizationMarkers(marker_cfg)

def transform_from_w2y(num_envs:int, root_ori_quat_w:torch.Tensor, target_w:torch.Tensor) -> torch.Tensor:
    # transformation
    root_ori_euler_w = math_utils.euler_xyz_from_quat(root_ori_quat_w)
    root_ori_euler_w_stack = math_utils.wrap_to_pi(torch.stack(root_ori_euler_w, dim=1))
    root_ori_euler_w_stack = torch.stack(root_ori_euler_w, dim=1)

    root_euler_rot_z = torch.zeros(num_envs, 3).to(root_ori_euler_w_stack.device)
    root_euler_rot_z[:, 2] = root_ori_euler_w_stack[:, 2]
    rot_mat_z = math_utils.matrix_from_euler(root_euler_rot_z, "XYZ")

    #batch matrix multiplication
    R = rot_mat_z.view(num_envs, 3, 3)
    R = R.transpose(1,2)
    target_w = target_w.unsqueeze(-1)
    # root_ori_euler_w_vec = root_ori_euler_w_stack.unsqueeze(-1)
    root_lin_vel_yaw_vec = torch.bmm(R, target_w)
    root_lin_vel_yaw= root_lin_vel_yaw_vec.squeeze(-1)

    return root_lin_vel_yaw