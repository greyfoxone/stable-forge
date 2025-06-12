# Heat Map Script
import io
from typing import Tuple

import gradio as gr
import matplotlib.pyplot as plt
import torch
from modules import scripts
from modules.processing import StableDiffusionProcessing
from PIL import Image


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
        return [enabled, block_nr_slider, step_slider, heatmap_gallery]

    def setup(self, p: StableDiffusionProcessing, *args):
        enabled, block_nr_slider, step_slider, heatmap_gallery = args
        if enabled is not True:
            return
        self.block_states = {}

    def update_heatmaps(self, enabled, block_nr_slider, step_slider):
        if enabled is not True:
            return

        if not self.block_states:
            return

        block_state = self.block_states[int(block_nr_slider)][int(step_slider)]

        caption = f"Block: {block_nr_slider}  Step {step_slider}"
        return self.cond_uncond_diffs(block_state, caption)

    # takes the diff of cond and uncond for each feature
    # and renders that diff
    def cond_uncond_diffs(self, block_state, caption):
        grid_size = 4
        grid_col = 2
        heatmaps = []
        features = block_state.backbone[1] - block_state.backbone[0]
        for i in range(0,len(features),grid_size): 
            grid = HeatmapGrid(size=grid_size, caption=caption + f"\nfeatures {i} - {i+grid_size}", cols=grid_col)

            for j, feature in enumerate(features[i:i+grid_size]):
                grid.axes.flat[j].set_title(f"Feature {i+j+1}")
                grid.axes.flat[j].imshow(feature.detach().numpy(), cmap="viridis")
                
    
            heatmaps.append(grid.render())
        return heatmaps

    # calculates the mean of each feature and renders the diff
    # of each feature from the mean
    def deviation_from_mean(self, block_state, caption):
        features = block_state.backbone.mean(dim=0)
        mean = torch.mean(features, dim=0)
        grid = HeatmapGrid(size=len(features), caption=caption, cols=32)

        print(f"Drawing Features for {caption}")
        for i, feature in enumerate(features):
            grid.axes.flat[i].imshow(feature - mean, cmap="viridis")
        print("Done")

        heatmaps = grid.render()
        return [heatmaps]

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ):
        enabled, block_nr_slider, step_slider, heatmap_gallery = args
        if enabled is not True:
            return

        unet = p.sd_model.forge_objects.unet
        unet_patched = heatmap_patch(unet, self.block_states)
        p.sd_model.forge_objects.unet = unet_patched
        return


# Tensor Dimension is (B=2,C,W,H)
# B=2 because it's the cond and uncond tensor each
class SdBlockState:
    def __init__(self, h: torch.Tensor, hsp: torch.Tensor, block_info):
        self.backbone = h
        self.skip = hsp
        self.block_nr = block_info[1]


def heatmap_patch(unet, block_states):
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


class HeatmapGrid:
    def __init__(self, size: int, caption: str = "", cols: int = 8):
        self.cols = cols
        self.size = size
        self.rows = int(size / cols) + int(size % cols != 0)
        self.caption = caption
        self.fig, self.axes = self.setup_grid()
        self.fig.suptitle(caption)

    def setup_grid(self) -> Tuple[plt.Figure, plt.Axes]:
        print(f"Setting up {self.cols} x {self.rows} for {self.caption}")
        fig, axes = plt.subplots(
            nrows=self.cols,
            ncols=self.rows,
            figsize=(self.rows * 3, self.cols * 3),
        )

        for ax in axes.flat:
            ax.axis("off")

        #        plt.tight_layout()
        print("Done")

        return fig, axes

    def render(self) -> Image:
        print(f"Rendering {self.caption}")
        buffer = io.BytesIO()
        self.fig.savefig(buffer, format="png")
        buffer.seek(0)
        image = Image.open(buffer)

        plt.close(self.fig)
        print("Done")
        return image
