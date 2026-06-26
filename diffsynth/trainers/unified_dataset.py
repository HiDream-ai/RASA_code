import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as F
from .aug_pose import load_keypoints_pkl, mp_main
from .draw_util import *
def make_combined_video(video, ref_img, pose, reference,
                        frame_height=832, frame_width=480,
                        out_path="combined.mp4",
                        fps=25,
                        to_gif=False):
    """
    video: list of 41 PIL.Images
    pose:  list of 41 PIL.Images
    reference: PIL.Image
    """
    # import pudb; pudb.set_trace()
    # 统一尺寸
    video = [F.resize(v, (frame_height, frame_width)) for v in video]
    pose  = [F.resize(p, (frame_height, frame_width)) for p in pose]
    reference = F.resize(reference, (frame_height, frame_width))
    ref_img = F.resize(ref_img, (frame_height, frame_width))
    frames = []
    for v, p in zip(video, pose):
        v_np = np.array(v)
        p_np = np.array(p)
        ref_np = np.array(reference)
        ref_img_np = np.array(ref_img)
        # 拼成一帧： ref | video | pose
        frame = np.concatenate([ref_np, ref_img_np, v_np, p_np], axis=1)
        frames.append(frame)

    # 输出视频或 GIF
    if to_gif:
        imageio.mimsave(out_path, frames, fps=fps)
    else:
        writer = imageio.get_writer(out_path, fps=fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        
def draw_pose(pose, H, W, idx):
    bodies = pose['bodies'][idx]
    faces = pose['faces'][idx]
    hands = pose['hands'][idx]
    candidate = bodies
    subset = pose['body_indices'][idx:idx+1]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    candidate = np.nan_to_num(candidate, nan=-1)
    canvas = draw_body_and_foot(canvas, candidate, subset)

    canvas = draw_handpose(canvas, hands)


    return canvas

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators

    def __call__(self, data, key, random_index=None, ref_id=None, **kwargs):
        for operator in self.operators:
            data = operator(data, key, random_index, ref_id, **kwargs)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadFolder(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, convert_RGB=True):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.convert_RGB = convert_RGB
        
    def get_num_frames(self, num_available_frames):
        num_frames = self.num_frames
        if num_available_frames < num_frames:
            num_frames = num_available_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, key, random_index=None,**kwargs):
        num_frames = self.num_frames
        num_frames = self.get_num_frames(len(os.listdir(data)))
        
        pose = []
        pose_frames = sorted([os.path.join(data, p) for p in os.listdir(data)])
        for img in pose_frames[random_index:random_index + num_frames]:
            frame = Image.open(img)
            if self.convert_RGB: frame = frame.convert("RGB")
            frame = self.frame_processor(frame)
            pose.append(frame)
        return pose

class LoadPkl_ref(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, convert_RGB=True):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.convert_RGB = convert_RGB
        self.std = np.load('data/Std.npy')
        self.mean = np.load('data/Mean.npy')
    def get_num_frames(self, num_available_frames):
        num_frames = self.num_frames
        if num_available_frames < num_frames:
            num_frames = num_available_frames
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, key, random_index=None, ref_img=None, **kwargs):
        if data.endswith('npy'):
            motion = np.load(data)
            motion = (motion - self.mean) / self.std
            motion = motion[np.newaxis,:,:]
            motion = motion[:,random_index:random_index+self.num_frames]
            motion = np.concatenate([np.tile(motion[:, :1], (1, 3, 1)),motion[:, :self.num_frames]], axis=1)
            return motion
        else:
            
            num_frames = self.num_frames
            all_pose = load_keypoints_pkl(data)
            num_frames = self.get_num_frames(len(all_pose['bodies']))
            H,W,_ = all_pose['size']
            pose = []

            ref_pose = draw_pose(all_pose,H,W,idx = random_index)
            ref_frame = Image.fromarray(ref_pose)
            if self.convert_RGB: ref_frame = ref_frame.convert("RGB")
            ref_frame = self.frame_processor(ref_frame)
            scale_pose = mp_main(all_pose,start=random_index,end=random_index + num_frames, mode='train')
            for idx in range(random_index, random_index+num_frames):
                scale_dwpose_woface = draw_pose(scale_pose,H,W,idx = idx)
                frame = Image.fromarray(scale_dwpose_woface)
                if self.convert_RGB: frame = frame.convert("RGB")
                frame = self.frame_processor(frame)
                pose.append(frame)
                
            # import pudb; pudb.set_trace()
            return [pose, [ref_frame]]

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, data: str, key, random_index, ref_id, **kwargs):
        reader = imageio.get_reader(data)
        a = int(reader.count_frames())
        num_frames = self.get_num_frames(reader)
        ref_images = []
        # for ref_id in range(a):
        #     image = reader.get_data(ref_id)
        #     image = Image.fromarray(image)
        #     image = self.frame_processor(image)
        #     ref_images.append(image)
        if a - num_frames-5>0:
            random_index = random.randrange(0, a - num_frames-5)
        else:
            random_index = 0
        # frames = ref_images[random_index: random_index + num_frames]
        ref_img = reader.get_data(ref_id)
        ref_img = Image.fromarray(ref_img)
        ref_img = self.frame_processor(ref_img)
        
        frames = []
        for frame_id in range(random_index, random_index + num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return [frames, random_index, ref_img]

        # return [frames, ref_images, random_index]



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str, key, random_index, ref_id, **kwargs):
        if os.path.isdir(data):
            file_ext_name = "folder"
        else:
            file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data, key, random_index, ref_id, **kwargs)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data, key, random_index=None, ref_id=None, **kwargs):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data, key, random_index, ref_id, **kwargs)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data, key, random_index=None, ref_id=None, **kwargs):
        if key=='video':
            return os.path.join(self.base_path, data)
        else:
            return data  # pose is already an absolute path



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        onetomany=1,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
        self.onetomany = onetomany
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("pkl","npy"), LoadPkl_ref(num_frames, time_division_factor, time_division_remainder, 
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            random_index = 0
            for key in self.data_file_keys:
                if key in data:
                    if key =="vace_reference_image":
                        data[key] = [data['video'][0][0]]

                        data['video'] = data['video'][0]
                        continue
                    if key =="pose_pkl":
                        random_index = data['video'][1]
                    if key =="video":
                        ref_id = random.choice(eval(data["vace_reference_image"]))
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key], key, random_index if key=="pose_pkl" or key=="motion_sequence" else None, ref_id if key=='video' else None,
                                                            onetomany=self.onetomany)

        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
