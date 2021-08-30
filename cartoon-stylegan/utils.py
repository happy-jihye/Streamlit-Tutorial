import os
import torch
from torchvision.utils import save_image, make_grid
import PIL.Image as pilimg
import imageio
from skimage import img_as_ubyte
from bokeh.models.widgets import Div
import streamlit as st


def open_link(url, new_tab=True):
    """Dirty hack to open a new web page with a streamlit button."""
    # From: https://discuss.streamlit.io/t/how-to-link-a-button-to-a-webpage/1661/3
    if new_tab:
        js = f"window.open('{url}')"  # New tab or window
    else:
        js = f"window.location.href = '{url}'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

# ---------------
# for styleclip

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )
        
# ---------------------------------

google_drive_paths = {

    "ffhq256.pt" : "https://drive.google.com/uc?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO",
    "encoder_ffhq.pt" : "https://drive.google.com/uc?id=1QQuZGtHgD24Dn5E21Z2Ik25EPng58MoU",

    # Naver Webtoon
    "NaverWebtoon.pt" : "https://drive.google.com/uc?id=1yIn_gM3Fk3RrRphTPNBPgJ3c-PuzCjOB",
    "NaverWebtoon_FreezeSG.pt" : "https://drive.google.com/uc?id=1OysFtj7QTy7rPnxV9TXeEgBfmtgr8575",
    "NaverWebtoon_StructureLoss.pt" : "https://drive.google.com/uc?id=1Oylfl5j-XGoG_pFHtQwHd2G7yNSHx2Rm",

    "Romance101.pt" : "https://drive.google.com/uc?id=1wWt4dPC9TJfJ6cF3mwg7kQvpuVwPhSN7",

    "TrueBeauty.pt" : "https://drive.google.com/uc?id=1yEky49SnkBqPhdWvSAwgK5Sbrc3ctz1y",

    "Disney.pt" : "https://drive.google.com/uc?id=1z51gxECweWXqSYQxZJaHOJ4TtjUDGLxA",
    "Disney_FreezeSG.pt" : "https://drive.google.com/uc?id=1PJaNozfJYyQ1ChfZiU2RwJqGlOurgKl7",
    "Disney_StructureLoss.pt" : "https://drive.google.com/uc?id=1PILW-H4Q0W8S22TO4auln1Wgz8cyroH6",
    
    "Metface_FreezeSG.pt" : "https://drive.google.com/uc?id=1P5T6DL3Cl8T74HqYE0rCBQxcq15cipuw",
    "Metface_StructureLoss.pt" : "https://drive.google.com/uc?id=1P65UldIHd2QfBu88dYdo1SbGjcDaq1YL",
}

def download_pretrained_model(download_all=True, file='', path = 'networks'):

    os.makedirs(path, exist_ok=True)

    from gdown import download as drive_download
    
    if download_all==True:
        for nn in google_drive_paths:
            url = google_drive_paths[nn]
            networkfile = os.path.join(path, nn)

            drive_download(url, networkfile, quiet=False)

    else:
        url = google_drive_paths[file]
        networkfile = os.path.join(path, file)

        drive_download(url, networkfile, quiet=False)

def load_stylegan_generator(network, device='cuda'):

    image_size=256
    latent_dim = 512

    from model import Generator

    network = torch.load(network, map_location=device)

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(network["g_ema"], strict=False)

    return generator

def make_interpolation_video(g1, g2, g3, g4, number_of_img, number_of_step, swap, swap_layer_num, device='cuda'):

    trunc1 = g1.mean_latent(4096)
    trunc2= g2.mean_latent(4096)
    trunc3= g3.mean_latent(4096)
    trunc4= g4.mean_latent(4096)

    image = []

    with torch.no_grad():

        latent1 = torch.randn(1, 14, 512, device=device)
        latent1 = g1.get_latent(latent1)
        latent_interp = torch.zeros(1, latent1.shape[1], latent1.shape[2]).to(device)

        for _ in range(number_of_img):
            # latent1

            latent2 = torch.randn(1, 14, 512, device=device)
            latent2 = g1.get_latent(latent2)


            for j in range(number_of_step):

                latent_interp = latent1 + (latent2-latent1) * float(j/(number_of_step-1))

                imgs_gen1, save_swap_layer = g1([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc1,
                                        swap=swap, swap_layer_num=swap_layer_num,
                                        randomize_noise=False)
                imgs_gen2, _ = g2([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc2,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        )
                imgs_gen3, _ = g3([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc3,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        )
                imgs_gen4, _ = g4([latent_interp],
                                        input_is_latent=True,                                     
                                        truncation=0.7,
                                        truncation_latent=trunc4,
                                        swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                        )

                                
                grid = make_grid(torch.cat([imgs_gen1, imgs_gen2, imgs_gen3, imgs_gen4], 0),
                                    nrow=4,
                                    normalize=True,
                                    range=(-1,1),
                                    )
                
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = pilimg.fromarray(ndarr)
                image.append(im)

            latent1 = latent2

    imageio.mimsave(f'./asset/result.mp4', \
                    [img_as_ubyte(image[i]) \
                    for i in range(len(image))])

def interpolation(generator1, generator2, latent1, latent2, number_of_step=6, out_name1='', out_name2='', input_latent=False, strength=1, device='cuda', swap=False, swap_layer_num=3):
    
    if input_latent:
        latent1 = generator1.get_latent(latent1)
        latent2 = generator2.get_latent(latent2)

    trunc1 = generator1.mean_latent(4096)
    trunc2 = generator2.mean_latent(4096)

    number_of_step = 6 #@param {type:"slider", min:0, max:10, step:1}
    latent_interp = torch.zeros(number_of_step, latent1.shape[1], latent1.shape[2]).to(device)

    with torch.no_grad():
        for j in range(number_of_step):

            latent_interp[j] = latent1 + strength * (latent2-latent1) * float(j/(number_of_step-1))

            imgs_gen1, save_swap_layer = generator1([latent_interp],
                                    input_is_latent=True,                                     
                                    truncation=0.7,
                                    truncation_latent=trunc1,
                                    swap=swap, swap_layer_num=swap_layer_num,
                                    )
                                    
            imgs_gen2, _ = generator2([latent_interp],
                                    input_is_latent=True,                                     
                                    truncation=0.6,
                                    truncation_latent=trunc2,
                                    swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
                                    )

    import imageio
    imageio.imsave(f'./asset/{out_name1}', tensor2image(torch.cat([img_gen for img_gen in imgs_gen1], dim=2)))
    imageio.imsave(f'./asset/{out_name2}', tensor2image(torch.cat([img_gen for img_gen in imgs_gen2], dim=2)))


def stylemixing(generator1, generator2, latent1, latent2, latent_mixing1, latent_mixing2, truncation=0.7, swap=False, swap_layer_num=3):
    
    latent1 = generator1.get_latent(latent1)
    trunc1 = generator1.mean_latent(4096)

    latent2 = generator2.get_latent(latent2)
    trunc2 = generator2.mean_latent(4096)

    latent3 = torch.cat([latent1[:,:latent_mixing1,:], latent2[:,latent_mixing1:latent_mixing2,:], latent1[:,latent_mixing2:,:]], dim = 1)
    latent = torch.cat([latent1, latent2, latent3], dim = 0)

    with torch.no_grad():
        img1, save_swap_layer = generator1(
            [latent],
            truncation=truncation,
            truncation_latent=trunc1,
            swap=swap, swap_layer_num=swap_layer_num,
            input_is_latent=True,
        )

        img2, _ = generator2(
            [latent],
            input_is_latent=True,
            truncation=truncation,
            truncation_latent=trunc2,
            swap=swap, swap_layer_num=swap_layer_num, swap_layer_tensor=save_swap_layer,
        )

    ffhq = torch.cat([img1[0], img1[1], img1[2]], dim=2)
    cartoon = torch.cat([img2[0], img2[1], img2[2]], dim=2)

    imageio.imsave('./asset/stylemixing3.png', tensor2image(torch.cat([ffhq, cartoon], dim = 1)))

def stylemixing2(generator, r_latent, n_sample, latent_mixing1, truncation=0.7):
    
    latent1 = generator.get_latent(r_latent)
    trunc = generator.mean_latent(4096)

    # generate image
    img1, _ = generator(
        [latent1],
        truncation=truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    images = [torch.cat([img for img in img1], 2)]

    for i in range(n_sample):
        latent = latent1[i].unsqueeze(0).repeat(n_sample, 1, 1)
        latent2 = torch.cat([latent[:,:latent_mixing1,:], latent1[:,latent_mixing1:,:]], dim = 1)
        
        # generate image
        img1, _ = generator(
            [latent2],
            truncation=truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        images.append(torch.cat([img for img in img1], 2))

    imageio.imsave('./asset/stylemixing4.png', tensor2image(torch.cat([img for img in images], 1)))
