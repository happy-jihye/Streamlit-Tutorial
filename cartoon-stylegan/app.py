import os
import sys
import torch

import utils

import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

from contextlib import contextmanager
from threading import current_thread
from io import StringIO


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    "this will show the prints"
    st_redirect(sys.stdout, dst)
    #yield

@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield


if __name__ == '__main__':
    
    st.title("Cartoon StyleGAN 🙃")
    st.markdown("")

    # https://shields.io/
    # st.markdown("<br>", unsafe_allow_html=True)
    # """
    # [![Star](https://img.shields.io/github/stars/happy-jihye/Cartoon-StyleGAN.svg?logo=github&style=social)](https://github.com/happy-jihye/Cartoon-StyleGAN)
    # """
        
    st.info('''
    Recent studies have shown remarkable success in the unsupervised image to image (I2I) translation. However, due to the imbalance in the data, learning joint distribution for various domains is still very challenging. Although existing models can generate realistic target images, it’s difficult to maintain the structure of the source image. In addition, training a generative model on large data in multiple domains requires a lot of time and computer resources. To address these limitations, I propose a novel image-to-image translation method that generates images of the target domain by finetuning a stylegan2 pretrained model. The stylegan2 model is suitable for unsupervised I2I translation on unbalanced datasets; it is highly stable, produces realistic images, and even learns properly from limited data when applied with simple fine-tuning techniques. Thus, in this project, I propose new methods to preserve the structure of the source images and generate realistic images in the target domain.
    ''')

    col1, col2, col3 = st.columns([3,2,2])
    open_colab = col1.button("🚀 Open in Colab")  # logic handled further down
    open_github = col2.button("💻 Github")  # logic handled further down
    open_arxiv = col3.button("📒 Arxiv")  # logic handled further down
    
    if open_colab:
        utils.open_link('https://colab.research.google.com/github/happy-jihye/Cartoon-StyleGan2/blob/main/Cartoon_StyleGAN2.ipynb')
    if open_github:
        utils.open_link('https://github.com/happy-jihye/Cartoon-StyleGAN')
    if open_arxiv:
        utils.open_link('https://arxiv.org/abs/2106.12445')


    st.header('1. Downloading pretrained model...')

    network = st.selectbox("Select a model", 
                ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG', 'Romance101', 'TrueBeauty'])
    
    col1, col2 = st.columns(2)
    with col1:
        button = st.button("Download")
    with col2:
        print_log = st.checkbox("Print log")
            
    if button:
        if print_log:
            with st_stderr("code"):
                utils.download_pretrained_model(False, file=f"{network}.pt", path='./networks')
        else:
            utils.download_pretrained_model(False, file=f"{network}.pt", path='./networks')
            st.success(f"{network} Checkpoint Download Complete! ")


    st.markdown('---')
    st.header('2. Generate Images using Pretrained model !🙃')
    col1, col2 = st.columns(2)
    with col1:
        network1 = st.selectbox("Select a network1", 
                ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=0)
    with col2:
        network2 = st.selectbox("Select a network2", 
                ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=2)

    col1, col2 = st.columns(2)
    with col1:
        network3 = st.selectbox("Select a network3", 
                ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=5)
    with col2:
        network4 = st.selectbox("Select a network4", 
                ['ffhq256', 'NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=7)


    col1, col2, col3 = st.columns([3,2,2])
    button2 =col2.button("Generate")


    if button2:
        device='cuda'
        n_sample=5
        truncation = 0.7

        # Load Generator
        g1 = utils.load_stylegan_generator(f'./networks/{network1}.pt')
        g2 = utils.load_stylegan_generator(f'./networks/{network2}.pt')
        g3 = utils.load_stylegan_generator(f'./networks/{network3}.pt')
        g4 = utils.load_stylegan_generator(f'./networks/{network4}.pt')
        st.success(f'Load All Generator !')

        st.session_state.g1 = g1 
        st.session_state.g2 = g2
        st.session_state.g3 = g3 
        st.session_state.g4 = g4 



    # Save Images
    number_of_img = st.slider('Number of Images', 1, 30, step=1, value=5)
    number_of_step = st.slider('Number of Steps', 1, 10, step=1, value=7)
    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown('')
        st.markdown('')
        swap = st.checkbox('Swap')
    with col2:
        swap_layer_num = st.slider('number of layer to swap', 1, 6, step=1, value=2)


    col1, col2, col3 = st.columns(3)
    button3 = col2.button("Make Interpolation Video")
    if button3:
        from utils import make_interpolation_video
        make_interpolation_video(st.session_state.g1, st.session_state.g2, st.session_state.g3, st.session_state.g4, number_of_img, number_of_step, swap, swap_layer_num)

    if os.path.isfile(f'./asset/result.mp4'):
            
        st.markdown('''
        ---
        ### Result
        I trained the model with a naver webtoon dataset that didn't align. Models trained with naver webtoon dataset may have worse results than disney or metface dataset.
        ''')
        st.video(f'./asset/result.mp4')

    # --------------------------------
    st.markdown('---')
    st.header('3. Style Mixing')
    device='cuda'
    
    col1, col2 = st.columns(2)
    with col1:
        network1 = st.selectbox("Select a network1", 
                ['ffhq256', 'NaverWebtoon', 'Romance101', 'TrueBeauty', 'Disney', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=0)
    with col2:
        network2 = st.selectbox("Select a network2", 
                ['ffhq256', 'NaverWebtoon', 'Romance101', 'TrueBeauty', 'Disney', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=2)

    st.markdown('### 3.1 Interpolation')

    col1, col2 = st.columns(2)
    with col1:
        seed1 = st.slider('seed1', 0, 1000000, value=627356, step=1)
        torch.manual_seed(seed1)
        latent1 = torch.randn(1, 14, 512, device=device)
    with col2:
        seed2 = st.slider('seed2', 0, 1000000, value=159972, step=1)
        torch.manual_seed(seed2)
        latent2 = torch.randn(1, 14, 512, device=device)


    col1, col2, col3 = st.columns([3,2,2])
    button4 = col2.button('Ok 😊')

    if button4:
        # Load Generator
        g1 = utils.load_stylegan_generator(f'./networks/{network1}.pt')    
        g2 = utils.load_stylegan_generator(f'./networks/{network2}.pt')    

        utils.interpolation(g1, g2, latent1, latent2, input_latent=True, out_name1='stylemixing1.png', out_name2='stylemixing2.png',)

        st.session_state.g1 = g1
        st.session_state.g2 = g2


    if os.path.isfile('./asset/stylemixing1.png'):
        st.markdown('**result**')
        st.image('./asset/stylemixing1.png')
        st.image('./asset/stylemixing2.png')

    # Ex2 
    st.markdown('### 3.2 Layer mixing')

    col1, col2 = st.columns(2)
    with col1:
        latent_mixing1 = st.slider('mixing layer 1', 0, 15, value=4, step=1)

    with col2:
        latent_mixing2 = st.slider('mixing layer 2', 0, 15, value=9, step=1)

    col1, col2, col3 = st.columns([3,2,2])
    button5 = col2.button('Ok 😉')

    if button5:
        utils.stylemixing(st.session_state.g1,st.session_state.g2, latent1, latent2, latent_mixing1, latent_mixing2)
    
    if os.path.isfile('./asset/stylemixing3.png'):
        st.markdown('**result**')
        st.image('./asset/stylemixing3.png')


    # Ex3
    st.markdown('### 3.3 Style mixing')

    network = st.selectbox("Select a network", 
            ['ffhq256', 'NaverWebtoon', 'Romance101', 'TrueBeauty', 'Disney', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
            index=1)

    col1, col2 = st.columns(2)
    with col1:
        n_sample = st.slider('number of sample', 0, 10, value=6, step=1)

    with col2:
        latent_mixing1 = st.slider('mixing layer', 0, 14, value=6, step=1)

    col1, col2, col3 = st.columns([3,2,2])
    button5 = col2.button('Ok ☺')

    if button5:
        g = utils.load_stylegan_generator(f'./networks/{network}.pt')    

        r_latent = torch.randn(n_sample, 14, 512, device=device)
        utils.stylemixing2(g, r_latent, n_sample=n_sample, latent_mixing1= latent_mixing1)
    
    if os.path.isfile('./asset/stylemixing4.png'):
        st.markdown('**result**')
        st.image('./asset/stylemixing4.png')


    # --------------------------------
    st.markdown('---')
    st.header('4. Closed-Form Factorization')
    st.markdown(
        '- [CVPR 2021] Closed-Form Factorization of Latent Semantics in GANs : [`Project Page`](https://genforce.github.io/sefa/)'
    )
    st.markdown(
        'The following animations are created by manipulating the versatile semantics unsupervisedly found by SeFa from GAN models trained on various datasets.'
    )
    col1, col2 = st.columns(2)
    with col1:
        network1 = st.selectbox("Select a network 1", 
                ['ffhq256', 'NaverWebtoon', 'Romance101', 'TrueBeauty', 'Disney', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=0)
    with col2:
        network2 = st.selectbox("Select a network 2", 
                ['ffhq256', 'NaverWebtoon', 'Romance101', 'TrueBeauty', 'Disney', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG'],
                index=2)

    seed = st.slider('seed', 0, 1000000, value=123177 ,step=1)
    col1, col2 = st.columns(2)
    with col1:
        index = st.slider('index', 0, 512, value=7, step=1)

    with col2:
        degree = st.slider('degree', 0, 30, value=14, step=1)
    
    col1, col2, col3 = st.columns([3,2,2])
    button5 = col2.button('Ok 🤗')
    if button5:
        if not os.path.isfile('./netowrks/factor.py'):
            os.system('python3 closed_form_factorization.py --factor_name ./networks/factor.pt /workspace/streamlit/cartoon-stylegan/networks/ffhq256.pt')
        os.system(f"python3 apply_factor.py --index=7 --degree=14 --seed=123177 --n_sample=5 \
                        --ckpt='./networks/{network1}.pt' --ckpt2='./networks/{network2}.pt' \
                        --factor='./networks/factor.pt' --video") 
    
    
    if os.path.isfile('./asset/sefa_video.mp4'):
        st.markdown('**result**')
        st.video('./asset/sefa_video.mp4')

    # --------------------------------
    st.markdown('---')
    st.header('5. StyleCLIP')
    st.markdown(
        '- [ICCV 2021] StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery : [`Paper`](https://arxiv.org/abs/2103.17249), [`Github`](https://github.com/orpatashnik/StyleCLIP)'
    )
    col1, col2 = st.columns(2)
    with col1:
        description = st.text_input('Write driving text', value='a really angry face')
    with col2:
        network = st.selectbox("Select a model ", 
                ['NaverWebtoon', 'NaverWebtoon_StructureLoss', 'NaverWebtoon_FreezeSG', 'Disney', 'Disney_StructureLoss', 'Disney_FreezeSG', 'Metface_StructureLoss', 'Metface_FreezeSG', 'Romance101'])
    
    optimization_steps = st.slider('optimization steps', 0, 1000, value=300 ,step=10)
    seed = st.slider('seed ', 0, 1000000, value=356289 ,step=1)
    l2_lambda = 0.004
    create_video=False
    result_dir = './asset/styleclip'
    os.makedirs(result_dir, exist_ok=True)

    args = {
        "seed" : seed,
        "description": description,
        "ckpt": "./networks/ffhq256.pt",
        "ckpt2": f"./networks/{network}.pt",
        "stylegan_size": 256,
        "latent_dim" : 14,
        "lr_rampup": 0.05,
        "lr": 0.1,
        "step": optimization_steps,
        "l2_lambda": l2_lambda,
        "latent_path": None,
        "truncation": 0.7,
        "save_intermediate_image_every": 1 if create_video else 20,
        "device" : "cuda",
        "results_dir": result_dir,
    }

    col1, col2, col3 = st.columns([3,2,2])
    button6 = col2.button('Ok 🥰')


    if button6:
        from styleclip_optimization import main
        from argparse import Namespace

        final_result, latent_init, latent_fin = main(Namespace(**args))
        st.success(f"Finish StyleCLIP Optimization !")

        st.session_state.latent_init = latent_init
        st.session_state.latent_fin = latent_fin

    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown('')
        st.markdown('')
        swap = st.checkbox('Swap ')
    with col2:
        swap_layer_num = st.slider('number of layer to swap ', 1, 6, step=1, value=2)
    strength =st.slider('number of layer to swap ', 1.0, 3.0, step=0.25, value=2.0)
    col1, col2, col3 = st.columns([3,2,2])
    button7 = col2.button('Ok 🥰 ')   


    if button7:
        # Load Generator
        g1 = utils.load_stylegan_generator(f'./networks/ffhq256.pt')    
        g2 = utils.load_stylegan_generator(f'./networks/{network}.pt')    

        utils.interpolation(g1, g2, st.session_state.latent_init, st.session_state.latent_fin, input_latent=False, strength=strength, swap=swap, swap_layer_num = swap_layer_num, out_name1='styleclip1.png', out_name2='styleclip2.png')

    if os.path.isfile('./asset/styleclip1.png'):
        st.markdown('**result**')
        st.image('./asset/styleclip1.png')
        st.image('./asset/styleclip2.png')