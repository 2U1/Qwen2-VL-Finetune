# Fine-tuning Phi3.x-Vision Series

This repository contains a script for training [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) without only using huggingface.

## Table of Contents

- [Installation](#installation)
  - [Using `environment.yaml`](#using-environmentyaml)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Full Finetuning](#full-finetuning)
  - [Finetune with LoRA](#finetune-with-lora)
    - [Merge LoRA Weights](#merge-lora-weights)

## Supported Features

- Deepspeed
- LoRA, QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image training and inference

## Installation

Install the required packages using `environment.yml`.

The code of Qwen2-VL has been in the latest Hugging face transformers and we advise you to build from source.

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate qwen2
pip install flash-attn --no-build-isolation
pip install qwen-vl-utils
pip install git+https://github.com/huggingface/transformers accelerate
```

**Note:** You should install the `flash-attn` after running other libraries `environment.yaml`.

#### Model loading issue

Showing the warning <br>
`` Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'mrope_section'} ``

Don't know what causes this issue. Looks like env issue.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**

<details>
<summary>Example Dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

## Training

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Phi3-vision model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for `vision_tower` and `merger`.
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

#### Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [ ] Support for video data

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)
- [model loading issue](#model-loading-issue)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Qwen2-VL-Finetuning,
  author = {Yuwon Lee},
  title = {Qwen2-VL-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Qwen2-VL-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA](https://github.com/haotian-liu/LLaVA): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct): Awesome pretrained MLLM based on Qwen2.