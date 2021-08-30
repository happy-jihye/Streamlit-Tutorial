# 🖼️ VQGAN-CLIP

This repo was heavily based on the implementation of tnwei.
- [`/vqgan-clip-app`](https://github.com/tnwei/vqgan-clip-app)

<p align='center'><b> ✨ Inference using StreamLIT </b></p> 
<p align='center'><img src='../asset/vqgan.gif?raw=1' width = '1100' ></p>

```
# install python packages
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer

# clone other repositories
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'

# download checkpoints
mkdir checkpoints
curl -L -o checkpoints/vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o checkpoints/vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
```

