import os
import streamlit as st
import argparse
from pathlib import Path
import sys
import datetime
import shutil
import json

sys.path.append("./taming-transformers")

from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from CLIP import clip

from utils import (
    load_vqgan_model,
    MakeCutouts,
    parse_prompt,
    resize_image,
    Prompt,
    synth,
    checkin,
    download_checkpoint,
)
from typing import Optional, List
from omegaconf import OmegaConf
import imageio
import numpy as np

defaults = OmegaConf.load("defaults.yaml")


def run(
    # Inputs
    text_input: str = "the first day of the waters",
    vqgan_ckpt: str = "vqgan_imagenet_f16_16384",
    num_steps: int = 300,
    image_x: int = 300,
    image_y: int = 300,
    image_prompts: List = [],
    continue_prev_run: bool = False,
    seed: Optional[int] = None,
    **kwargs,  # Use this to receive Streamlit objects
):
    # Leaving most of this untouched
    args = argparse.Namespace(
        prompts=[text_input],
        image_prompts=image_prompts,
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[int(image_x), int(image_y)],
        # init_image=None,
        init_weight=0.0,
        # clip.available_models()
        # ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']
        # Visual Transformer seems to be the smallest
        clip_model="ViT-B/32",
        vqgan_config=f"./checkpoints/{vqgan_ckpt}.yaml",
        vqgan_checkpoint=f"./checkpoints/{vqgan_ckpt}.ckpt",
        step_size=0.05,
        cutn=64,
        cut_pow=1.0,
        display_freq=50,
        seed=seed,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if continue_prev_run is True:
        # Streamlit tie-in -----------------------------------
        model = st.session_state["model"]
        perceptor = st.session_state["perceptor"]
        # End of Streamlit tie-in ----------------------------

    else:
        # Streamlit tie-in -----------------------------------
        # Remove the cache first! CUDA out of memory
        if "model" in st.session_state:
            del st.session_state["model"]

        if "perceptor" in st.session_state:
            del st.session_state["perceptor"]

        # debug_slot.write(st.session_state) # DEBUG

        model = st.session_state["model"] = load_vqgan_model(
            args.vqgan_config, args.vqgan_checkpoint
        ).to(device)
        perceptor = st.session_state["perceptor"] = (
            clip.load(args.clip_model, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        # End of Streamlit tie-in ----------------------------

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2 ** (model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # if args.init_image:
    #     pil_image = Image.open(fetch(args.init_image)).convert('RGB')
    if continue_prev_run:
        # Streamlit tie-in -----------------------------------
        pil_image = st.session_state["prev_im"]
        # End of Streamlit tie-in ----------------------------

        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=device), n_toks
        ).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    
    if len(args.image_prompts) != 0:
        for uploaded_image in args.image_prompts:
            # path, weight, stop = parse_prompt(prompt)
            # img = resize_image(Image.open(fetch(path)).convert("RGB"), (sideX, sideY))
            pil_image = uploaded_image.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)        


    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    pMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    # Streamlit tie-in -----------------------------------------------------------------
    for uploaded_image in args.image_prompts:
        # path, weight, stop = parse_prompt(prompt)
        # img = resize_image(Image.open(fetch(path)).convert("RGB"), (sideX, sideY))
        img = resize_image(uploaded_image.convert("RGB"), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))
    # End of Streamlit tie-in ----------------------------------------------------------

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    def ascend_txt():
        out = synth(model, z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        result = []

        if args.init_weight:
            result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))

        return result

    # Streamlit tie-in -----------------------------------

    step_counter = 0
    frames = []

    try:
        # Try block catches st.script_runner.StopExecution, no need of a dedicated stop button
        # Reason is st.form is meant to be self-contained either within sidebar, or in main body
        # The way the form is implemented in this app splits the form across both regions
        # This is intended to prevent the model settings from crowding the main body
        # However, touching any button resets the app state, making it impossible to
        # implement a stop button that can still dump output
        while True:
            # While loop to accomodate running predetermined steps or running indefinitely
            status_text.text(f"Running step {step_counter}")
            opt.zero_grad()
            lossAll = ascend_txt()
            im = checkin(step_counter, lossAll, model, z)

            if num_steps > 0:  # skip when num_steps = -1
                step_progress_bar.progress((step_counter + 1) / num_steps)
            else:
                step_progress_bar.progress(100)


            # At every step, display and save image
            if len(image_prompts) != 0:
                im_init_display_slot.image(image_prompts, caption="Input image")
                im_display_slot.image(im, caption="Output image", output_format="PNG")
            else:
                im_display_slot.image(im, caption="Output image", output_format="PNG")

            st.session_state["prev_im"] = im

            # ref: https://stackoverflow.com/a/33117447/13095028
            # im_byte_arr = io.BytesIO()
            # im.save(im_byte_arr, format="JPEG")
            # frames.append(im_byte_arr.getvalue()) # read()
            frames.append(np.asarray(im))

            # End of Stremalit tie-in ------------------------

            loss = sum(lossAll)
            loss.backward()
            opt.step()
            with torch.no_grad():
                z.copy_(z.maximum(z_min).minimum(z_max))

            step_counter += 1

            if (step_counter == num_steps) and num_steps > 0:
                break

        # Stitch into video using imageio
        writer = imageio.get_writer("temp.mp4", fps=24)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # Save to output folder if run completed
        runoutputdir = outputdir / datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        runoutputdir.mkdir()

        im.save(runoutputdir / "output.PNG", format="PNG")
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        with open(runoutputdir / "details.json", "w") as f:
            json.dump(
                {"num_steps": step_counter, "text_input": text_input}, f, indent=4
            )

        status_text.text("Done!")

        # End of Stremalit tie-in ----------------------------

        return im

    except st.script_runner.StopException as e:
        # Dump output to dashboard
        print(f"Received Streamlit StopException")
        status_text.text("Execution interruped, dumping outputs ...")
        writer = imageio.get_writer("temp.mp4", fps=24)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # Save to output folder if run completed
        runoutputdir = outputdir / datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        runoutputdir.mkdir()

        im.save(runoutputdir / "output.PNG", format="PNG")
        shutil.copy("temp.mp4", runoutputdir / "anim.mp4")

        with open(runoutputdir / "details.txt", "w") as f:
            json.dump(
                {"num_steps": step_counter, "text_input": text_input}, f, indent=4
            )

        status_text.text("Done!")


if __name__ == "__main__":

    outputdir = Path("output")

    if not outputdir.exists():
        outputdir.mkdir()

    st.set_page_config(page_title="VQGAN-CLIP")

    st.title('🖼️ VQGAN-CLIP ')
    st.markdown('')
    st.markdown('''
    - developped by [`@happy-jihye`](https://github.com/happy-jihye)
    ''')

    setup = st.expander('🔧 Set Up')
    with setup:
        st.markdown('1. Install python packages')
        st.code('''
pip install ftfy regex tqdm omegaconf pytorch-lightning 
pip install Python kornia imageio imageio-ffmpeg einops torch_optimizer
        ''')


        st.markdown('2. Clone other repositories')
        st.code('''
git clone 'https://github.com/openai/CLIP'
git clone 'https://github.com/CompVis/taming-transformers'
        ''')

        st.markdown('3. checkpoints')
        st.markdown('''
See [`CompVis/taming-transformers`](https://github.com/CompVis/taming-transformers#overview-of-pretrained-models) for more information about pre-trained models.
        ''')
        col1, col2 = st.columns([3,1])
        with col1:
            checkpoint = st.selectbox('Download Checkpoint',
                        ['imagenet_1024', 'imagenet_16384', 'coco', 'faceshq', 'sflckr']
            )
        with col2:
            st.markdown('')
            st.markdown('')
            checkpoint_down = st.button('Download 😊')   
        if checkpoint_down:
            download_checkpoint(checkpoint)



    st.markdown('---')

    # Determine what weights are available in `checkpoints/`
    available_weights = os.listdir('./checkpoints')
    available_weights = [file.strip('.ckpt') for file in available_weights if file.endswith(".ckpt")]

    # Set vqgan_imagenet_f16_1024 as default if possible
    if "vqgan_imagenet_f16_1024" in available_weights:
        default_weight_index = available_weights.index("vqgan_imagenet_f16_1024")
    else:
        default_weight_index = 0

    # Start of input form
    with st.form("form-inputs"):
        # Only element not in the sidebar, but in the form
        text_input = st.text_input(
            "Text prompt",
            value='A painting in the style of Picasso',
            help="VQGAN-CLIP will generate an image that best fits the prompt",
        )
        radio = st.sidebar.radio(
            "Model weights",
            available_weights,
            index=default_weight_index,
            help="Choose which weights to load, trained on different datasets. Make sure the weights and configs are downloaded to `assets/` as per the README!",
        )
        num_steps = st.sidebar.slider(
            "Num steps",
            value=defaults["num_steps"],
            min_value=0,
            max_value=1000,
            step=1,
            help="Specify -1 to run indefinitely. Use Streamlit's stop button in the top right corner to terminate execution. The exception is caught so the most recent output will be dumped to dashboard",
        )

        image_x = st.sidebar.number_input(
            "Xdim", value=defaults["Xdim"], help="Width of output image, in pixels"
        )
        image_y = st.sidebar.number_input(
            "ydim", value=defaults["ydim"], help="Height of output image, in pixels"
        )
        set_seed = st.sidebar.checkbox(
            "Set seed",
            value=defaults["set_seed"],
            help="Check to set random seed for reproducibility. Will add option to specify seed",
        )

        seed_widget = st.sidebar.empty()
        if set_seed is True:
            seed = seed_widget.number_input(
                "Seed", value=defaults["seed"], help="Random seed to use"
            )
        else:
            seed = None

        use_custom_starting_image = st.sidebar.checkbox(
            "Add image prompt(s)",
            value=defaults["use_image_prompts"],
            help="Check to add image prompt(s), conditions the network similar to the text prompt",
        )

        starting_image_widget = st.sidebar.empty()
        if use_custom_starting_image is True:
            image_prompts = starting_image_widget.file_uploader(
                "Upload image prompts(s)",
                type=["png", "jpeg", "jpg"],
                accept_multiple_files=True,
                help="Image prompt(s) for the network, will be resized to fit specified dimensions",
            )
            print(image_prompts)
            # Convert from UploadedFile object to PIL Image
            if len(image_prompts) != 0:
                image_prompts = [Image.open(i) for i in image_prompts]
        else:
            image_prompts = []

        continue_prev_run = st.sidebar.checkbox(
            "Continue previous run",
            value=defaults["continue_prev_run"],
            help="Use existing image and existing weights for the next run",
        )
        col1, col2, col3 = st.columns([3,2,2])
        submitted = col2.form_submit_button("Run!")
        # End of form

    status_text = st.empty()
    status_text.text("Pending input prompt")
    step_progress_bar = st.progress(0)

    col1, col2 = st.columns(2)
    with col1:
        im_init_display_slot = st.empty()
    with col2:
        im_display_slot = st.empty()
    vid_display_slot = st.empty()
    debug_slot = st.empty()

    if "prev_im" in st.session_state:
        if len(image_prompts) != 0:
            im_init_display_slot.image(image_prompts, caption="Input image")
            im_display_slot.image(st.session_state["prev_im"], caption="Output image", output_format="PNG")
        else:
            im_display_slot.image(st.session_state["prev_im"], caption="Output image", output_format="PNG")



    if submitted:
        # debug_slot.write(st.session_state) # DEBUG
        status_text.text("Loading weights ...")
        im = run(
            # Inputs
            text_input=text_input,
            vqgan_ckpt=radio,
            num_steps=num_steps,
            image_x=int(image_x),
            image_y=int(image_y),
            seed=int(seed) if set_seed is True else None,
            image_prompts=image_prompts,
            continue_prev_run=continue_prev_run,
            im_display_slot=im_display_slot,
            step_progress_bar=step_progress_bar,
            status_text=status_text,
        )
        vid_display_slot.video("temp.mp4")
        # debug_slot.write(st.session_state) # DEBUG