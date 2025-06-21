# Heat Map Script

import gradio as gr
import numpy as np
import torch
from matplotlib import cm
from modules import scripts
from modules.processing import StableDiffusionProcessing
from PIL import Image

class SdBlockState:
    def __init__(self, h: torch.Tensor, hsp: torch.Tensor, block_info):
        self.h = h
        self.hsp = hsp
        self.block_nr = block_info[1]


def get_block_state(unet, block_states):
    def output_block_patch(
        h: torch.Tensor, hsp: torch.Tensor, transformer_options
    ):
        state = SdBlockState(h.cpu(), hsp.cpu(), transformer_options["block"])
        if state.block_nr not in block_states:
            block_states[state.block_nr] = []
        block_states[state.block_nr].append(state)
        return h, hsp

    unet_patched = unet.clone()
    unet_patched.set_model_output_block_patch(output_block_patch)
    return unet_patched

class Heatmap(scripts.Script):
    def title(self):
        return "Heatmaps"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Blocks() as heatmap_interface:
            with gr.Accordion(label="Heatmap", open=False):
                enabled = gr.Checkbox(label="Enable", value=False)

                block_nr_slider = gr.Slider(
                    minimum=0, maximum=8, step=1, label="Block Number", value=4
                )
                step_slider = gr.Slider(
                    minimum=1, maximum=100, step=1, label="Step", value=10
                )
                heatmap_gallery = gr.Gallery(
                    label="Generated Heatmaps", type="pil"
                )

            def get_output_gallery(c):
                c.component.change(
                    fn=self.update_heatmaps,
                    inputs=[enabled, block_nr_slider, step_slider],
                    outputs=[heatmap_gallery],
                )

            self.on_after_component(
                get_output_gallery, elem_id="txt2img_gallery"
            )
            block_nr_slider.change(
                    fn=self.update_heatmaps,
                    inputs=[enabled, block_nr_slider, step_slider],
                    outputs=[heatmap_gallery],
                )
            step_slider.change(
                fn=self.update_heatmaps,
                inputs=[enabled, block_nr_slider, step_slider],
                outputs=[heatmap_gallery],
            )
                        
        return [enabled, block_nr_slider, step_slider, heatmap_gallery]

    def setup(self, p: StableDiffusionProcessing, *args):
        enabled, block_nr_slider, step_slider, heatmap_gallery = args
        if enabled is not True:
            return
        self.block_states = {}
        self.renderer = FeatureRenderer()

    def update_heatmaps(self, enabled, block_nr_slider, step_slider):
        if enabled is not True:
            return
        
        if hasattr(self,'block_states') == False:
            return
        
        print(self.block_states)
        
        block_state = self.block_states[int(block_nr_slider)][int(step_slider)]
        print(f"Render Features for Block {int(block_nr_slider)} at step {int(step_slider)}")
        return self.renderer.grid(block_state.hsp)


class FeatureRenderer:
    cols: int = 32

    def grid(self, features, padding=1):
        rows = int(len(features) / self.cols) + 1
        heatmaps = [
            self.feature(feature.detach().numpy())
            for feature in features
        ]
        width, height = heatmaps[0].size
        width += padding
        height += padding

        grid_canvas = Image.new("RGB", (self.cols * width, rows * height))

        for i in range(rows):
            for j in range(self.cols):
                pos = (j * width, i * height)
                grid_canvas.paste(heatmaps[i * self.cols + j], pos)

        return [grid_canvas] + [heatmaps]

    def feature(self, array):
        # Normalize the array to 0-255 range for image.
        normalized_array = (array - np.min(array)) / (
            np.max(array) - np.min(array)
        )
        heatmap_image = (normalized_array * 255).astype(np.uint8)

        jet_map = cm.get_cmap("jet")
        heatmap_color = (
            jet_map(heatmap_image.reshape(-1))[:, :3] * 255
        ).astype(np.uint8)

        # Create PIL image from the array.
        return Image.fromarray(heatmap_color.reshape(*heatmap_image.shape, -1))