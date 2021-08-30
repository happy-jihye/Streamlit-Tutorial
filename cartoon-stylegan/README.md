# Cartoon-StyleGAN

<p align='center'><b> ✨ Inference using StreamLIT </b></p> 
<p align='center'><img src='../asset/cartoon-stylegan-1.gif?raw=1' width = '900' ></p>

- [Original Repo](https://github.com/happy-jihye/Cartoon-StyleGAN)

## Application
### 1. Generative Videos

<p align='center'><img src='../asset/cartoon-stylegan-2.gif?raw=1' width = '900' ></p>

### 2. Style Mixing

<p align='center'><img src='../asset/cartoon-stylegan-3.gif?raw=1' width = '900' ></p>

### 3. Closed-Form Factorization

<p align='center'><img src='../asset/cartoon-stylegan-4.gif?raw=1' width = '900' ></p>

### 4. StyleCLIP

<p align='center'><img src='../asset/cartoon-stylegan-5.gif?raw=1' width = '900' ></p>


## Install

```
pip install bokeh ftfy regex tqdm gdown

# for styleclip
pip install git+https://github.com/openai/CLIP.git
```

## Run
```
streamlit run app.py --server.address 0.0.0.0 --server.port [your port]
```