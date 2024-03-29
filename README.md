# Streamlit Tutorials

#### Install
```
pip install streamlit
```

#### Run
```
cd [directory]
streamlit run app.py --server.address 0.0.0.0 --server.port [your port]
# http://0.0.0.0:[your port]
```
## Main App
```
# run
streamlit run app.py --server.address 0.0.0.0 --server.port [your port]
```

<p align="center">
    <img src='asset/main.gif?raw=1' width = '900' >
</p>

## 💸 Stock Price Dashboard ✨

```
# install
pip install yfinance fbprophet plotly

# run
cd stock-price-dashboard
streamlit run app.py --server.address 0.0.0.0 --server.port [your port]
```

<p align="center">
    <img src='asset/finance.gif?raw=1' width = '900' >
</p>

## 🙃 Cartoon StyleGAN ✨

- [`happy-jihye/Cartoon-StyleGAN`](https://github.com/happy-jihye/Cartoon-StyleGAN)

```
# install
pip install bokeh ftfy regex tqdm gdown
# for styleclip
pip install git+https://github.com/openai/CLIP.git

# run
cd cartoon-stylegan
streamlit run app.py --server.address 0.0.0.0 --server.port [your port]
```

<p align="center">
    <img src='asset/cartoon-stylegan-1.gif?raw=1' width = '700' >
</p>


## 🖼️ VQGAN-CLIP ✨

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

# run
cd VQGAN-CLIP
streamlit run app.py --server.address 0.0.0.0 --server.port [your port]
```


<p align='center'><img src='asset/vqgan.gif?raw=1' width = '1100' ></p>

