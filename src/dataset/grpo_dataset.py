import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import SYSTEM_MESSAGE

import re

def replace_image_tokens(input_string, is_video=False):
    pattern = r'\n?' + re.escape("<image>") + r'\n?' if not is_video else r'\n?' + re.escape("<video>") + r'\n?'

    return re.sub(pattern, '', input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def get_image_content(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    return content

def get_video_content(video_path, min_pixels, max_pixels, width, height, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    

    return content

class GRPODataset(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(GRPODataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        contents = []

        if "image" in sources:

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(get_image_content(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                contents.append(get_video_content(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps))

        conversations = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        user_input = conversations[0]
        gpt_response = conversations[1]

        text_content = {"type": "text", "text": user_input['content']}

        contents.append(text_content)

        user_prompt = [{"role": "user", "content": contents}]

        if len(SYSTEM_MESSAGE) > 0:
            system_message = {"role": "system", "content": SYSTEM_MESSAGE}
            user_prompt.insert(0, system_message)
        
        data_dict = dict(
            prompt=user_prompt,
            assistant=gpt_response,
        )

        return data_dict
    
def make_grpo_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    grpo_dataset = GRPODataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )

    return dict(train_dataset=grpo_dataset,
                eval_dataset=None)