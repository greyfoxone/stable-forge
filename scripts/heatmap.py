# Heat Map Script

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import torch
from modules import scripts
from modules.processing import StableDiffusionProcessing
from modules.shared import class_debug_log
from modules.shared import debug_log
from modules.shared_state import log
from PIL import Image
from scripts.heatmaps.render import HiddenStateVisualizer


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


@class_debug_log
class Heatmap(scripts.Script):
    def title(self):
        return "Heatmaps"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @debug_log
    def ui(self, *args, **kwargs):
        with gr.Blocks() as heatmap_interface:
            with gr.Accordion(label="Heatmap", open=False):
                enabled = gr.Checkbox(label="Enable", value=False)

                block_nr_slider = gr.Slider(
                    minimum=0, maximum=8, step=1, label="Block Number", value=4
                )

                heatmap_gallery = gr.Gallery(
                    label="Generated Heatmaps", type="pil"
                )

        def get_output_gallery(c):
            c.component.change(
                fn=self.update_heatmaps,
                inputs=[enabled, block_nr_slider],
                outputs=[heatmap_gallery],
            )

        if self.is_img2img:
            self.on_after_component(
                get_output_gallery, elem_id="img2img_gallery"
            )
        else:
            self.on_after_component(
                get_output_gallery, elem_id="txt2img_gallery"
            )
            block_nr_slider.change(
                fn=self.update_heatmaps,
                inputs=[enabled, block_nr_slider],
                outputs=[heatmap_gallery],
            )

        return [enabled, block_nr_slider, heatmap_gallery]

    def setup(self, p: StableDiffusionProcessing, *args):
        enabled, block_nr_slider, heatmap_gallery = args
        if enabled is not True:
            return
        self.block_states = {}

    @debug_log
    def update_heatmaps(
        self,
        enabled: bool,
        block_nr_slider: int,
    ) -> list:
        if not enabled or not hasattr(self, "block_states"):
            return []

        block = block_nr_slider
        grids = []
        captions = []
        def compute_diff(step: int) -> None:
            current_state = self.block_states[block][step]
            prev_state = self.block_states[block][step - 1]
            diff = prev_state["hsp"] - current_state["hsp"]
            caption = f"block_{block + 1}_step_{step}-_{step + 1}"
            grids.append(HiddenStateVisualizer(diff).render_diff(caption))
            captions.append(caption)
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            list(
                executor.map(
                    compute_diff, range(1, len(self.block_states[block]))
                )
            )
        
        gif_path = "heatmap.gif"
#        os.remove(gif_path)
        grids[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=grids[1:],
            duration=500,
            loop=0,
        )
        # open the gif file into an PIL image and delete the file
        gif_img = Image.open(gif_path)
        
        return [(gif_path,f"block_{block + 1}")] + [(grid, caption) for grid, caption in zip(grids, captions)]


    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ):
        if not args or not any(args):
            return

        enabled, block_nr_slider, heatmap_gallery = args
        if enabled is not True:
            return

        unet = p.sd_model.forge_objects.unet
        unet_patched = get_block_state(unet, self.block_states)
        p.sd_model.forge_objects.unet = unet_patched
        return