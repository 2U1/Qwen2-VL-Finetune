conda create -n qwen2 python==3.10
conda activate qwen2

# CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

看你的版本，我用的118

```
pip install qwen-vl-utils
pip install packaging
pip install ninja
pip install flash-attn==2.7.0.post2 --no-build-isolation
pip install liger-kernel --no-build-isolation
pip install deepspeed
pip install tensorboardx
pip install ujson
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers accelerate
```

