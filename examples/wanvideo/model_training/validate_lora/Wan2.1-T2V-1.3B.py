import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import os
from diffsynth.trainers.unified_dataset import *

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/merge_allxinit_1xadd_Wan2.1-T2V-1.3B_lora/epoch-4.safetensors", alpha=1)
pipe.load_trainable(pipe.convert, "models/train/merge_allxinit_1xadd_Wan2.1-T2V-1.3B_lora/epoch-4.safetensors")
pipe.enable_vram_management()


frame_processor=ImageCropAndResize(832, 480, 832*480, 16, 16)

pose_data = ''
pose = []
pose_frames = sorted([os.path.join(pose_data, p) for p in os.listdir(pose_data)])
for img in pose_frames[:41]:
    frame = Image.open(img)
    frame = frame.convert("RGB")
    frame = frame_processor(frame)
    pose.append(frame)

video = pipe(
    prompt="A woman is showcasing her elegant outfit, posing like a model",
    driven_pose=pose,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=1, tiled=True
)
save_video(video, "video_Wan2.1-T2V-1.3B.mp4", fps=15, quality=5)
