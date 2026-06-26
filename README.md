# RASA

<p align="center">
  <img src="assets/framework.png" alt="RASA framework" width="90%">
</p>

<p align="center">
  <a href="https://hidream-ai.github.io/RASA/">Project Page</a> |
  <a href="https://hidream-ai.github.io/RASA/">Paper</a> |
  <a href="https://huggingface.co/HiDream-ai/RASA">Model</a>
</p>

## Video Demo

<p align="center">
  <video src="https://github.com/user-attachments/assets/1b456d7b-4df5-4182-850f-9ae3264ca91d" width="45%" controls autoplay muted loop playsinline></video>
  <video src="https://github.com/user-attachments/assets/50b8f82f-7bbc-4879-8557-79027e03c062" width="45%" controls autoplay muted loop playsinline></video>
</p>




## Installation

### Requirements

- Linux GPU environment with CUDA 12.1
- Python 3.11
- PyTorch 2.4.1
- DiffSynth-Studio 1.1.8

### Environment Setup

```bash
git clone https://github.com/HiDream-ai/RASA_code.git
cd RASA

conda create -n rasa python=3.11 -y
conda activate rasa

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

The scripts are designed for GPU inference and training. Adjust the PyTorch/CUDA installation if your cluster requires a specific CUDA build.

## Model Files

### Wan2.1-T2V-1.3B Pretrained Model

Download the Wan2.1-T2V-1.3B pretrained model from ModelScope:

- [Wan-AI/Wan2.1-T2V-1.3B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)

Place the downloaded files under:

```text
models/Wan-AI/Wan2.1-T2V-1.3B/
- diffusion_pytorch_model.safetensors
- models_t5_umt5-xxl-enc-bf16.pth
- Wan2.1_VAE.pth
- google/
```

### RASA Pretrained Model

Download the RASA pretrained model from ModelScope:

```bash
modelscope download --model RASA --local_dir models/ckpt
```

Place the downloaded files under:

```text
models/ckpt/
- model.safetensors    # RASA 720p weights
- net_last.pth         # Motion encoder
```

## Data Layout

The custom validation pipeline uses the following data structure:

```text
data/
- joint_vecs/       # Motion numpy files, e.g. *_processed_input.npy
- raw_data/         # Source videos, searched recursively as processed_<video_id>.mp4
- raw_data_pkl/     # Per-video pose pickle files, e.g. <video_id>_data.pkl
- ref_pose/         # Reference pose pickle files
- Mean.npy          # Motion normalization mean
- Std.npy           # Motion normalization std
```

Keep video IDs consistent across the video, pose, reference pose, and motion files. The validation script uses those IDs to match all inputs.

## Training

Launch training with `accelerate`:

```bash
accelerate launch --config_file accelerate_config_T2V.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/raw_data \
  --dataset_metadata_path xxxx.csv \
  --data_file_keys video,pose_pkl,vace_reference_image,motion_sequence \
  --height 832 \
  --width 480 \
  --dataset_repeat 1 \
  --model_paths '["models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors","models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth","models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"]' \
  --learning_rate 1e-5 \
  --num_epochs 20 \
  --remove_prefix_in_ckpt "pipe.dit.,pipe.convert." \
  --output_path ./models/train/kvq_retarget \
  --trainable_models "dit,convert" \
  --extra_inputs vace_reference_image
```

Key dataset arguments:

- `--dataset_base_path`: Root directory of the source videos and related training files.
- `--dataset_metadata_path`: Path to the metadata CSV file.
- `--data_file_keys`: Metadata columns used by the dataloader. For RASA training, use `video,pose_pkl,vace_reference_image,motion_sequence`.

Example metadata CSV:

```csv
video,prompt,pose_pkl,vace_reference_image,motion_sequence
00001.mp4,"A person is dancing, posing like a model.",training_pose/00001.pkl,"[37, 39, 24, 25, 23]",joint_vecs/00001.npy
```

`vace_reference_image` stores the frame indices used as reference images. You can provide any valid frame indices from the corresponding source video.
## Inference

Run the RASA evaluation script:

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_custom_eval_ref_CIM.py
```

Main configuration values are defined near the top of the script:

```python
MOTION_ROOT = "data/joint_vecs"
STD_PATH = "data/Std.npy"
MEAN_PATH = "data/Mean.npy"
ref_pose_path = "data/ref_pose"
VIDEO_ROOT = "data/raw_data"
POSE_ROOT = "data/raw_data_pkl"
OUTPUT_DIR = "output_results_clips"
full_path = "models/ckpt/model.safetensors"
```

Generated clips and comparison videos are written to `output_results_clips/`.

## Evaluation

We follow the evaluation protocol used by DisCo. Please refer to the official DisCo repository for metric details and evaluation setup: [Wangt-CN/DisCo](https://github.com/Wangt-CN/DisCo)

After preparing the generated videos and evaluation data, run:

```bash
sh gen_eval.sh
```

## Notes
- For multi-GPU training, check `num_processes` in `accelerate_config_T2V.yaml` before launching.










