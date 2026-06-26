import torch
import numpy as np
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new_ref_pose_bind_ablation import WanVideoPipeline, ModelConfig

import os
import glob
from tqdm import tqdm
from diffsynth.models.utils import load_state_dict
import gc
import re
from diffsynth.trainers.unified_dataset import *

# ================= 配置区域 =================
MOTION_ROOT = 'data/joint_vecs'
STD_PATH = 'data/Std.npy'
MEAN_PATH = 'data/Mean.npy'

ref_pose_path = 'data/ref_pose'
VIDEO_ROOT = 'data/raw_data'
POSE_ROOT = 'data/raw_data_pkl'
OUTPUT_DIR = 'output_results_clips'
full_path = "models/ckpt/model.safetensors"

CLIP_LENGTH = 49
HEIGHT = 1280
WIDTH = 720
MAX_FRAMES = 50  # 新增：最大推理帧数限制
VIDEO_IDS = None  # 如果为 None，则自动从 VIDEO_ROOT 搜索
# ===========================================
std = np.load(STD_PATH)
mean = np.load(MEAN_PATH)
def load_model():
    print("Loading Model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors", model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    if os.path.exists(full_path):
        print(f"Loading full model from {full_path}...")
        state_dict = load_state_dict(full_path)
        pipe.dit.load_state_dict(state_dict, strict=False)
        pipe.convert.load_state_dict(state_dict, strict=False)
        print(state_dict.keys())
    # if os.path.exists(lora_path):
    #     pipe.load_lora(pipe.dit, lora_path, alpha=1)
    #     pipe.load_trainable(pipe.convert, lora_path)
    pipe.enable_vram_management()
    return pipe


def draw_pose_adaptive(pose_data, H, W, idx=0):
    """
    根据 bodies 的维度自动选择绘制逻辑：
    - (f, n, 2): 三维张量，执行序列绘制逻辑
    - (n, 2): 二维张量，执行单帧绘制逻辑
    """
    bodies = pose_data['bodies']
    
    # 获取 bodies 的维度深度
    dims = np.array(bodies).ndim

    if dims == 3:
        # --- 执行原 draw_pose 逻辑 (处理序列中的第 idx 帧) ---
        # print(f"检测到序列数据 {bodies.shape}，提取第 {idx} 帧")
        curr_bodies = pose_data['bodies'][idx]
        hands = pose_data['hands'][idx]
        subset = pose_data['body_indices'][idx:idx+1]
        
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        candidate = np.nan_to_num(curr_bodies, nan=-1)
        
        # 这里的函数调用保持你原有的三维逻辑写法
        canvas = draw_body_and_foot(canvas, candidate, subset)
        canvas = draw_handpose(canvas, hands)
        return canvas

    elif dims == 2:
        # --- 执行原 draw_pose_canvas 逻辑 (处理单帧数据) ---
        # print(f"检测到单帧数据 {bodies.shape}")
        candidate = pose_data['bodies']
        subset = pose_data['body_indices']
        hands = pose_data['hands']
        
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 这里的函数调用保持你原有的二维逻辑写法（带 util. 前缀）
        canvas = draw_body_and_foot(canvas, candidate, subset[np.newaxis, :])
        canvas = draw_handpose(canvas, hands)
        return canvas

    else:
        raise ValueError(f"未知的 Pose 数据维度: {dims}. 预期为 2 (n,2) 或 3 (f,n,2)")
    
def find_video_path(video_id):
    """
    在 VIDEO_ROOT 及其子文件夹下搜索名为 processed_{video_id}.mp4 或 .MP4 的文件
    """
    patterns = [
        os.path.join(VIDEO_ROOT, "**", f"processed_{video_id}.mp4"),
        os.path.join(VIDEO_ROOT, "**", f"processed_{video_id}.MP4")
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    return None


def process_single_video(pipe, video_id):
    """使用 VideoData 统一处理并保存 1x4 对比视频和纯生成视频"""
    save_dir = os.path.join(OUTPUT_DIR, video_id)
    pure_output_filename = os.path.join(save_dir, f"{video_id}.mp4")
    combined_filename = os.path.join(save_dir, f"{video_id}_full_compare.mp4")
    # import pudb; pudb.set_trace()
    # if video_id!='298225546508894':
    #     return
    if os.path.exists(save_dir):
        print(f"Skip: {video_id} already exists.")
        return
    try:
        motion_path = os.path.join(MOTION_ROOT, f"{video_id}_processed_input.npy")
    except Exception as e:
        print(f"!!! Failed to find motion for {video_id}: {e}")
        return
    # 3. 加载完整 Motion 并预处理
    motion = np.load(motion_path)
    motion = (motion - mean) / std
    motion = motion[np.newaxis, :, :] # shape: [1, Channels, Total_Frames]

    # ref_img_path = os.path.join(ref_path, f"{video_id}_input.png")
    # cross_id_image = Image.open(ref_img_path).convert('RGB')
    # cross_id_image = cross_id_image.resize((WIDTH, HEIGHT), Image.LANCZOS)
    # ref_img = VideoData(image_folder=ref_img_path, height=HEIGHT, width=WIDTH)
    ref_pose_file = os.path.join(ref_pose_path, f"{video_id}_processed_{video_id}.pkl")
    all_pose = load_keypoints_pkl(ref_pose_file)


    video_path = find_video_path(video_id)
    pose_path = os.path.join(POSE_ROOT, f"{video_id}_data.pkl")
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载并初始化所有 VideoData 对象
    gt_video_data = VideoData(video_path, height=HEIGHT, width=WIDTH)
    all_pose_data_raw = load_keypoints_pkl(pose_path)
    
    # 提取 Pose 数据
    ref_pose_data = all_pose_data_raw.get('image_data') 
    video_pose_data = all_pose_data_raw.get('video_data')
    H_pose, W_pose = video_pose_data['size']

    # 计算处理帧数
    process_len = min(len(gt_video_data), MAX_FRAMES)
    num_clips = process_len // CLIP_LENGTH
    
    # 2. 绘制姿态并封装进 VideoData 以便自动处理尺寸
    ref_pose_img = draw_pose_adaptive(ref_pose_data, H_pose, W_pose, idx=0)
    ref_pose_video = VideoData(array_list=[ref_pose_img], height=HEIGHT, width=WIDTH)
    
    full_pose_frames = []
    for idx in range(process_len):
        pose_frame = draw_pose_adaptive(video_pose_data, H_pose, W_pose, idx=idx)
        full_pose_frames.append(pose_frame)
    full_pose_video = VideoData(array_list=full_pose_frames, height=HEIGHT, width=WIDTH)

    print(f"Processing Video: {video_id} | Total Clips: {num_clips}")

    all_pred_frames = []
    all_gt_frames = []

    for clip_idx in range(num_clips):
        start_idx = clip_idx * CLIP_LENGTH
        end_idx = start_idx + CLIP_LENGTH
        H_pose, W_pose, _ = all_pose['size']
        ref_pose = draw_pose(all_pose, H_pose, W_pose, idx=start_idx)
        ref_pose = VideoData(array_list=[ref_pose], height=HEIGHT, width=WIDTH)
        # 准备模型输入
        clip_reference_image = gt_video_data[start_idx]
        clip_pose_driven = [full_pose_video[i] for i in range(start_idx, end_idx)]
        # 模型需要列表格式的参考姿态 [[H, W, 3]]
        input_pose_bundle = [clip_pose_driven, [ref_pose[0]]]

        # input_pose_bundle = [clip_pose_driven, [ref_pose_video[0]]]
        clip_gt_frames = [gt_video_data[i] for i in range(start_idx, end_idx)]
        
        # Motion (注意维度 [1, C, T])
        clip_motion = motion[:, start_idx:end_idx]
        if clip_motion.shape[1] < CLIP_LENGTH:
             # 如果出现尾部不足的情况（一般由外层循环控制避免，但为了鲁棒性加上padding）
            pad_len = CLIP_LENGTH - clip_motion.shape[1]
            last_val = clip_motion[:, -1:, :]
            clip_motion = np.concatenate([clip_motion, np.tile(last_val, (1, pad_len, 1))], axis=1)

        clip_motion = np.concatenate([np.tile(clip_motion[:, :1], (1, 3, 1)), clip_motion[:, :CLIP_LENGTH]], axis=1)
        
        video_result = pipe(
            prompt="A person is dancing, like a model",
            driven_pose=input_pose_bundle, 
            motion_sequence=clip_motion, 
            input_video_clip=clip_gt_frames,
            vace_reference_image=clip_reference_image, 
            num_frames=CLIP_LENGTH,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            seed=1, 
            tiled=True
        )

        all_pred_frames.extend(video_result)
        all_gt_frames.extend(clip_gt_frames)
        
        torch.cuda.empty_cache()
        gc.collect()

    # 3. 保存与拼接
    if all_pred_frames:
        # 保存纯生成结果
        # save_video(all_pred_frames, pure_output_filename, fps=15)
        
        # 封装生成的视频到 VideoData 确保尺寸一致性
        # pred_video_data = VideoData(array_list=all_pred_frames, height=HEIGHT, width=WIDTH)
        
        combined_frames = []
        for i in range(len(all_pred_frames)):
            # 1x4 横向布局
            canvas = Image.new('RGB', (WIDTH * 5, HEIGHT))
            
            # 使用 VideoData 的索引访问，保证都是统一尺寸的 PIL Image
            canvas.paste(gt_video_data[i], (0, 0))              # GT
            canvas.paste(all_pred_frames[i], (WIDTH, 0)) 
            canvas.paste(clip_reference_image, (WIDTH * 2, 0))# Pred
            canvas.paste(ref_pose[0], (WIDTH * 3, 0))     # Ref Pose (固定)
            canvas.paste(full_pose_video[i], (WIDTH * 4, 0))    # Full Pose
            
            combined_frames.append(canvas)
        
        save_video(combined_frames, combined_filename, fps=15)
        print(f"Successfully processed {video_id}")
        
def main():
    pipe = load_model()

    global VIDEO_IDS
    if VIDEO_IDS is None:
        print(f"Scanning {VIDEO_ROOT} for 'processed_*.mp4'...")
        search_pattern = os.path.join(VIDEO_ROOT, "**", "processed_*.[mM][pP]4")
        all_video_files = glob.glob(search_pattern, recursive=True)
        
        extracted_ids = []
        for f in all_video_files:
            file_name = os.path.basename(f)
            if 'input' in file_name:
                continue
            vid = re.sub(r'^processed_', '', os.path.splitext(file_name)[0])
            extracted_ids.append(vid)
        
        VIDEO_IDS = sorted(list(set(extracted_ids)))
        print(f"Found {len(VIDEO_IDS)} video IDs.")
    
    for vid in tqdm(VIDEO_IDS, desc="Overall Progress"):
        try:
            process_single_video(pipe, vid)
        except Exception as e:
            print(f"!!! Failed {vid}: {e}")
            import traceback
            traceback.print_exc()
        
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()