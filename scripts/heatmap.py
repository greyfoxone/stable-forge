# Heat Map Script

import pickle

import gradio as gr
import numpy as np
import torch
from matplotlib import cm
from modules import scripts
from modules.processing import StableDiffusionProcessing
from PIL import Image


def get_block_state(unet, block_states):
    def output_block_patch(
        h: torch.Tensor, hsp: torch.Tensor, transformer_options
    ):
        block_nr = transformer_options["block"][1]
        if block_nr not in block_states:
            block_states[block_nr] = []
        block_states[block_nr].append(
            {"h": h.detach().cpu(), "hsp": hsp.detach().cpu()}
        )
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

        if hasattr(self, "block_states") is False:
            return

        block_state = self.block_states[int(block_nr_slider)][int(step_slider)]
        print(
            f"Render Features for Block {int(block_nr_slider)} at step {int(step_slider)}"
        )

        features = block_state["hsp"].mean(dim=0)
        return self.renderer.grid(features)

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ):
        if not args or not any(args):
            return
        enabled, block_nr_slider, step_slider, heatmap_gallery = args
        if enabled is not True:
            return

        unet = p.sd_model.forge_objects.unet
        unet_patched = get_block_state(unet, self.block_states)
        p.sd_model.forge_objects.unet = unet_patched
        return

    def postprocess(self, p, processed, *args):
        # save  self.block_states to disk with bickle
        with open("block_states.pkl", "wb") as f:
            pickle.dump(self.block_states, f)


class FeatureRenderer:
    cols: int = 32

    def grid(
        self, features: list[torch.Tensor], padding=1
    ) -> list[tuple[Image.Image, str]]:
        all_features_flat = np.concatenate(
            [f.detach().numpy().flatten() for f in features]
        )
        norm_min, norm_max = np.min(all_features_flat), np.max(
            all_features_flat
        )

        heatmaps = [
            self.feature(f.detach().numpy(), norm_min, norm_max)
            for f in features
        ]
        grid_canvas = Image.new("RGB", (self.cols * width, rows * height))
        print("Render grid")
        for i in range(rows):
            for j in range(self.cols):
                pos = (j * width, i * height)
                idx = i * self.cols + j
                # check if idx larger than heatmaps
                # add empty cells
                if idx < len(heatmaps):
                    grid_canvas.paste(heatmaps[idx], pos)
        heatmap_count = len(heatmaps)
        grid_title = f"Grid {heatmap_count}"
        feature_titles = [
            (h, f"Feature {n+1}/{heatmap_count}")
            for n, h in enumerate(heatmaps)
        ]

        return [(grid_canvas, grid_title)] + feature_titles

    def feature(
        self, array: np.ndarray, norm_min: float, norm_max: float
    ) -> Image.Image:
        normalized_array = (array - norm_min) / (norm_max - norm_min)
        heatmap_image = (normalized_array * 255).astype(np.uint8)

        jet_map = cm.get_cmap("jet")
        heatmap_color = (
            jet_map(heatmap_image.reshape(-1))[:, :3] * 255
        ).astype(np.uint8)

        return Image.fromarray(heatmap_color.reshape(*heatmap_image.shape, -1))