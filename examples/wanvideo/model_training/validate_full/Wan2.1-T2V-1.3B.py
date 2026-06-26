# import torch
# from PIL import Image
# from diffsynth import save_video, VideoData, load_state_dict
# from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


# pipe = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
#         ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
#         ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
#     ],
# )
# state_dict = load_state_dict("models/train/Wan2.1-T2V-1.3B_full/epoch-1.safetensors")
# pipe.dit.load_state_dict(state_dict)
# pipe.enable_vram_management()

# video = pipe(
#     prompt="from sunset to night, a small town, light, house, river",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     seed=1, tiled=True
# )
# save_video(video, "video_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)


import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffsynth import save_video, VideoData
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.unified_dataset import *
import os
import glob
from tqdm import tqdm

# ================= 配置区域 =================
# 视频 ID 列表 (如果不指定，稍后会扫描目录)
# VIDEO_IDS = ['00336'] 
VIDEO_IDS = ['00335','00336','00337','00338','00339','00340'] # 设置为 None 则自动扫描 VIDEO_ROOT 下的所有 .mp4

# 数据根目录
VIDEO_ROOT = 'data/training_data'
POSE_ROOT = 'data/training_pose'
MOTION_ROOT = 'data/vace_step_2/new_joint_vecs'

# Motion 标准化参数
STD_PATH = 'data/vace_step_2/Std.npy'
MEAN_PATH = 'data/vace_step_2/Mean.npy'

# 输出设置
OUTPUT_DIR = 'output_results_clips'
CLIP_LENGTH = 41   # 每个 clip 的帧数
HEIGHT = 832
WIDTH = 480
# ===========================================

def load_model():
    """加载模型"""
    print("Loading Model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    # pipe.load_lora(pipe.dit, "models/train/rope_first_bind_full_epi_ref_xinit/epoch-19.safetensors", alpha=1)
    # pipe.load_trainable(pipe.convert, "models/train/rope_first_bind_full_epi_ref_xinit/epoch-19.safetensors")
    # pipe.enable_vram_management()
    # return pipe

    state_dict = load_state_dict("models/train/Wan2.1-T2V-1.3B_full/epoch-16.safetensors")
    pipe.dit.load_state_dict(state_dict, strict=False)
    pipe.convert.load_state_dict(state_dict, strict=False)
    # pipe.load_trainable(pipe.convert, "models/train/Wan2.1-T2V-1.3B_full/epoch-16.safetensors")
    pipe.enable_vram_management()
    return pipe

def process_single_video(pipe, video_id, std_norm, mean_norm):
    """处理单个视频的所有 Clips"""
    
    # 1. 构造文件路径
    video_path = os.path.join(VIDEO_ROOT, f"{video_id}.mp4")
    motion_path = os.path.join(MOTION_ROOT, f"{video_id}.npy")
    pose_path = os.path.join(POSE_ROOT, f"{video_id}.pkl")
    
    # 检查文件是否存在
    if not (os.path.exists(video_path) and os.path.exists(motion_path) and os.path.exists(pose_path)):
        print(f"Skipping {video_id}: Missing files.")
        return

    print(f"Processing Video: {video_id}")
    
    # 为当前视频创建保存目录
    save_dir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(save_dir, exist_ok=True)

    # 2. 加载 Reference Image 
    # (修改点：逻辑移至加载 GT 视频后，提取全局第0帧)

    # 3. 加载完整 GT 视频
    gt_video_data = VideoData(video_path, height=HEIGHT, width=WIDTH)
    total_frames = len(gt_video_data)
    
    # 【修改点】提取整个视频的第一帧作为全局参考图
    # 确保所有 Clip 都使用这一张图片
    global_reference_image = gt_video_data[0]
    
    # 4. 加载完整 Motion 并预处理
    motion = np.load(motion_path)
    motion = (motion - mean_norm) / std_norm
    motion = motion[np.newaxis, :, :] # shape: [1, Channels, Total_Frames]
    # 5. 加载完整 Pose 并绘制
    print("Drawing full pose sequence...")
    all_pose_data = load_keypoints_pkl(pose_path)
    H_pose, W_pose, _ = all_pose_data['size']
    
    # 预先处理好所有的 Pose 帧
    # 注意：如果视频非常长，这里可能会耗费内存。如果OOM，可以将 pose 绘制移到 clip 循环内
    full_pose_frames = []
    # 确保只处理到视频实际长度，防止越界
    process_len = min(total_frames, motion.shape[2]) 
    
    for idx in range(process_len):
        scale_dwpose = draw_pose(all_pose_data, H_pose, W_pose, idx=idx)
        full_pose_frames.append(scale_dwpose)
    
    full_pose_video = VideoData(array_list=full_pose_frames, height=HEIGHT, width=WIDTH)
    
    # 计算 Clip 数量
    num_clips = process_len // CLIP_LENGTH
    print(f"Total frames: {process_len}, Clip length: {CLIP_LENGTH}, Num clips: {num_clips}")

    # 用于收集所有帧的列表 (合并长视频用)
    all_pred_frames = []
    all_gt_frames = []

    # ================= Clip 循环 =================
    for clip_idx in range(num_clips):
        start_idx = clip_idx * CLIP_LENGTH
        end_idx = start_idx + CLIP_LENGTH
        
        print(f"  > Processing Clip {clip_idx+1}/{num_clips} (Frames {start_idx}-{end_idx})")

        # A. 切片数据
        # Reference Image: 使用全局第一帧 (修改处)
        clip_reference_image = gt_video_data[start_idx] ##global_reference_image

        # Pose
        clip_pose_frames = [full_pose_video[i] for i in range(start_idx, end_idx)]
        
        # Motion (注意维度 [1, C, T])
        clip_motion = motion[:, start_idx:end_idx]
        clip_motion = np.concatenate([np.tile(clip_motion[:, :1], (1, 3, 1)),clip_motion[:, :CLIP_LENGTH]], axis=1)

        # GT Frames (用于保存对比)
        clip_gt_frames = [gt_video_data[i] for i in range(start_idx, end_idx)]

        # B. 推理
        seed_val = 1  # 可选：每个 clip 使用不同种子，或者固定
        
        video_result = pipe(
            prompt="A person is dancing, like a model",
            driven_pose=clip_pose_frames, 
            motion_sequence=clip_motion, 
            vace_reference_image=clip_reference_image, # 此时这里是全局第一帧
            num_frames=CLIP_LENGTH,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            seed=seed_val, 
            tiled=True
        )

        # C. 收集结果 (暂不保存 Clip 文件)
        all_pred_frames.extend(video_result)
        all_gt_frames.extend(clip_gt_frames)

        # 清理显存 (防止长视频累积)
        torch.cuda.empty_cache()

    # ================= 循环结束，保存合并视频 =================
    if all_pred_frames:
        print(f"Saving merged videos for {video_id}...")
        
        # 1. 保存完整 GT 视频
        gt_filename = os.path.join(save_dir, f"{video_id}_full_gt.mp4")
        save_video(all_gt_frames, gt_filename, fps=15, quality=5)

        # 2. 保存完整 Pred 视频
        pred_filename = os.path.join(save_dir, f"{video_id}_full_pred.mp4")
        save_video(all_pred_frames, pred_filename, fps=15, quality=5)
        
        # 3. 保存合并对比视频 (左右拼接)
        combined_frames = []
        for i in range(len(all_pred_frames)):
            img_gt = all_gt_frames[i].copy()
            img_pred = all_pred_frames[i].copy()
            
            # 简单的左右拼接
            canvas = Image.new('RGB', (WIDTH * 2, HEIGHT))
            canvas.paste(img_gt, (0, 0))
            canvas.paste(img_pred, (WIDTH, 0))
            combined_frames.append(canvas)
            
        combined_filename = os.path.join(save_dir, f"{video_id}_full_compare.mp4")
        save_video(combined_frames, combined_filename, fps=15, quality=5)
        print(f"Saved: {combined_filename}")

def main():
    # 初始化
    pipe = load_model()
    std = np.load(STD_PATH)
    mean = np.load(MEAN_PATH)

    # 获取视频列表
    global VIDEO_IDS
    if VIDEO_IDS is None:
        # 扫描 VIDEO_ROOT 下所有的 .mp4 文件名（不含扩展名）
        files = glob.glob(os.path.join(VIDEO_ROOT, "*.mp4"))
        VIDEO_IDS = [os.path.splitext(os.path.basename(f))[0] for f in files]
        VIDEO_IDS.sort()
    
    print(f"Found {len(VIDEO_IDS)} videos to process.")

    # 主循环
    for vid in tqdm(VIDEO_IDS, desc="Total Progress"):
        try:
            process_single_video(pipe, vid, std, mean)
        except Exception as e:
            print(f"Error processing video {vid}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
