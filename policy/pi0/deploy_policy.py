import numpy as np
import torch
import dill
import os, sys
from typing import List, Optional
from PIL import Image
import matplotlib.pyplot as plt
import time

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, os.path.join(parent_directory, "src"))
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """Return ([front,right,left] rgb), state from env observation."""
    input_rgb_arr: list[np.ndarray] = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state: np.ndarray = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args: dict) -> PI0:
    """Create PI0 wrapper from config dict."""
    train_config_name, model_name, checkpoint_id, pi0_step = (
        usr_args["train_config_name"], usr_args["model_name"], usr_args["checkpoint_id"], usr_args["pi0_step"],
    )
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)

last_ep_num = 0
step = 0 
def eval(TASK_ENV, model: PI0, observation: dict) -> None:
    global step, last_ep_num
    step += 1
    if TASK_ENV.ep_num != last_ep_num:
        step = 0
        last_ep_num = TASK_ENV.ep_num

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)
        # Enable attention recording for this episode.
        model.enable_attention(True)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]
    last_action = actions[0]

    for action in actions:
        diff = action - last_action
        # print(np.abs(diff).max(), np.linalg.norm(diff))
        last_action = action
        TASK_ENV.now_obs['observation']['attention_mask'] = {'rgb': generate_attention_visualization(model, TASK_ENV)}
        TASK_ENV.take_action(action)
        model.get_attention(True)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()


def generate_attention_visualization(model: PI0, TASK_ENV):
    # return []
    attn_values= model.get_attention(False)
    if len(attn_values) == 0 or TASK_ENV.eval_video_path is None:
        return []
    # turn the attn_values to a image, each row is for one layer
    first_set = attn_values[:18] # (1, 1, 8, 816, 816)
    first_set = [a[0,0] for a in first_set]
    imgs = np.concatenate(first_set, axis=0)
    base_name = '0_clean_300m'
    instruction_path = f"/home/gerigkja/RoboTwin/eval_result/random/instruction_{base_name}_{TASK_ENV.ep_num}_{TASK_ENV.seed}.txt"
    if not os.path.exists(instruction_path):
        with open(instruction_path, "w") as f:
            f.write(TASK_ENV.get_instruction())
    path = f"/home/gerigkja/RoboTwin/eval_result/random/{base_name}_{TASK_ENV.ep_num}_{step}_{TASK_ENV.seed}.npy"
    if not os.path.exists(path):
        np.save(path, imgs)
    diffusion_path = f"/home/gerigkja/RoboTwin/eval_result/random/diffusion_{base_name}_{TASK_ENV.ep_num}_{step}_{TASK_ENV.seed}.npy"
    if not os.path.exists(diffusion_path):
        np.save(diffusion_path, np.concatenate(attn_values[18:])[:,0].reshape(10, 18, 8, 51, 867))
    return []
    imgs[:, :, :768] = 1
    imgs = (imgs * 255).astype(np.uint8)
    imgs = np.repeat(np.expand_dims(imgs, -1), 3, -1)
    # calculate how much each head gives attention to the images
    attention_entropy = np.zeros((18, 8))
    log_argmax_unique = np.zeros((18, 8))
    img_argmax_fraction = np.zeros((18, 8))
    img_attention = np.zeros((18, 8))
    img_0_attention = np.zeros((18, 8))
    img_1_attention = np.zeros((18, 8))
    img_2_attention = np.zeros((18, 8))
    for i, q_layer in enumerate(first_set):
        for j, head in enumerate(q_layer):
            distr = head.sum(-2) / head.sum()
            focus_tokens = np.unique(head.argmax(-2))
            log_argmax_unique[i, j] = np.log10(focus_tokens.shape[0])
            img_argmax_fraction[i, j] = focus_tokens[focus_tokens < 768].shape[0] / focus_tokens.shape[0]
            attention_entropy[i, j] = -np.sum(distr * np.log(distr))
            img_attention[i, j] = head[:,:768].sum() / head.sum()
            img_0_attention[i, j] = head[:,:256].sum() / head.sum()
            img_1_attention[i, j] = head[:,256:512].sum() / head.sum()
            img_2_attention[i, j] = head[:,512:768].sum() / head.sum()
    indiv_images = np.hstack([img_0_attention, img_1_attention, img_2_attention])
    names = ["img_attention", "indiv_images", "attention_entropy", "log_argmax_unique", "img_argmax_fraction"]
    max_val = {"img_attention": 1, "indiv_images": 1, 
               "attention_entropy": None, "log_argmax_unique": None, "img_argmax_fraction": None}
    for name, img in zip(names, [img_attention, indiv_images, attention_entropy, log_argmax_unique, img_argmax_fraction]):
        plt.imshow(img, vmin=0, vmax=max_val[name])
        plt.colorbar()
        plt.title(f"Episode {TASK_ENV.ep_num}, Frame {step}")
        plt.ylabel("Layer")
        plt.yticks(range(18), [f"Layer {i}" for i in range(18)])
        plt.tight_layout()
        plt.savefig(str(TASK_ENV.eval_video_path) + f"/{name}_{TASK_ENV.ep_num}_{step}.png")
        plt.close()
    return list(imgs)

def to_overlay_rgb(weight_map: np.ndarray, target_hw: tuple[int, int], vmin=None, vmax=None) -> np.ndarray:
    w = weight_map
    vmin = float(w.min()) if vmin is None else vmin
    vmax = float(w.max()) if vmax is None else vmax
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    w_norm = (w - vmin) / denom
    h, w_ = target_hw
    up = Image.fromarray((w_norm * 255).astype(np.uint8), mode="L").resize((w_, h), resample=Image.BILINEAR)
    up_np = np.asarray(up, dtype=np.float32) / 255.0
    r = up_np
    g = (up_np * 0.6)
    b = np.zeros_like(up_np)
    overlay = np.stack([r, g, b], axis=-1)
    return (overlay * 255.0).astype(np.uint8)

COUNTER = 0

def overlay_attention_on_frames(attn_records: List[np.ndarray], frames_by_name: dict) -> dict:
    """Return new frames dict with attention heatmap overlay per camera.

    Inputs:
    - attn_records: list of attention arrays (B,K,G,T,S); uses the last entry.
    - frames_by_name: mapping camera name -> RGB ndarray (H,W,3), uint8 or float [0,1].

    Output:
    - dict with the same keys for modified frames (uint8 RGB).
    """
    global COUNTER
    if not attn_records or not isinstance(frames_by_name, dict):
        return frames_by_name

    arr = attn_records[-(198 - COUNTER)]
    COUNTER += 1
    COUNTER %= 198
    if arr.ndim == 4:
        arr = arr[:, None, None, ...]
    if arr.ndim != 5:
        return frames_by_name

    attn_s = arr.mean(axis=(0, 1, 2, 3))
    grid = 16
    per_cam_tokens = grid * grid
    total_img_tokens = 3 * per_cam_tokens
    if attn_s.shape[0] < total_img_tokens:
        return frames_by_name

    # Token order in prefix: [front, left, right]
    # Map to camera keys: front -> head_camera, left -> left_camera, right -> right_camera
    token_to_cam = ["head_camera", "left_camera", "right_camera"]
    maps = []
    start = 0
    for _ in range(3):
        end = start + per_cam_tokens
        maps.append(attn_s[start:end].reshape(grid, grid))
        start = end

    alpha = 0.8
    out_frames: dict = {}
    for attn_map, cam_key in zip(maps, token_to_cam, strict=True):
        frame = frames_by_name.get(cam_key)
        if frame is None:
            continue
        if frame.dtype != np.uint8:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        h, w = frame.shape[:2]
        overlay = to_overlay_rgb(attn_map, (h, w), attn_s.min(), attn_s.max()).astype(np.float32)
        blended = np.clip((1.0 - alpha) * frame.astype(np.float32) + alpha * overlay, 0, 255)
        out_frames[cam_key] = blended.astype(np.uint8)

    # Keep other cameras unchanged
    for k, v in frames_by_name.items():
        if k not in out_frames:
            if type(v) == list:
                out_frames[k] = v
            elif v.dtype != np.uint8:
                out_frames[k] = (v * 255.0).clip(0, 255).astype(np.uint8)
            else:
                out_frames[k] = v

    return out_frames
