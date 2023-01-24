# modal-upscaling

Description: Upscale images using the StableDiffusionUpscalePipeline from the diffusers library.

![image](https://user-images.githubusercontent.com/24563696/213843797-6e058a99-a795-4657-a991-9e937fada878.png)

This example uses the A100 GPU to upscale images using `stabilityai/stable-diffusion-x4-upscaler` model that is loaded from the Hugging Face model hub.
The model is cached in a shared volume to avoid downloading it on every run.
The upscaled images are saved to a shared volume. The shared volume can be accessed through the modal CLI (see [modal docs](https://modal.com/docs/reference/cli/volume)).

To retrieve the upscaled images, run the following command: 

`modal volume get image_upscaling_vol /output/*.png [DESTINATION_DIR]`.
