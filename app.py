import os

import streamlit as st

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


# st.set_page_config(layout="wide")

st.title('Play with Stable-Diffusion v1-4')

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
auth_token = os.environ.get("StableDiffusion") or True


with st.spinner(
	text='Loading...'
):
	pipe = StableDiffusionPipeline.from_pretrained(
		model_id,
		revision="fp16",
		torch_dtype=torch.float16,
		use_auth_token=auth_token
	)

	pipe = pipe.to(device)


def infer(prompt, samples=2, steps=30, scale=7.5, seed=25):
	generator = torch.Generator(device=device).manual_seed(seed)

	with autocast("cuda"):
		images_list = pipe(
			[prompt] * samples,
			num_inference_steps=steps,
			guidance_scale=scale,
			generator=generator
		)

	return images_list["sample"]


with st.form(key='new'):

	prompt = st.text_area(label='Enter prompt')

	col1, col2, col3 = st.columns(3)

	with st.expander(label='Expand parameters'):
		n_samples = col1.select_slider(
			label='Num images',
			options=range(1, 5),
			value=1
		)

		steps = col2.select_slider(
			label='Steps',
			options=range(1, 101),
			value=40
		)

		scale = col3.select_slider(
			label='Guidance Scale',
			options=range(1, 21),
			value=7
		)

	st.form_submit_button()

	if prompt:
		images = infer(
			prompt,
			samples=n_samples,
			steps=steps,
			scale=scale
		)

		for image in images:
			st.image(image)
		with torch.no_grad():
			torch.cuda.empty_cache()
			pipe.to('cpu')
			pipe = None
	else:
		st.warning('Enter prompt.')
