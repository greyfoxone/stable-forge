import io

import gradio as gr
import numpy as np
import torch
from modules import scripts
from modules.helper import debug_log
from modules.script_callbacks import remove_current_script_callbacks
from modules.shared import state
from modules.ui_components import InputAccordion
from PIL import Image


def compression_ratio(array: np.ndarray) -> float:
    img = Image.fromarray((array * 255).astype(np.uint8))
    with io.BytesIO() as output:
        img.save(output, format="JPEG", compress_level=3)
        compressed_bytes = output.getvalue()
    ratio = len(compressed_bytes) / (array.size * array.itemsize)
    return ratio


def sort_by_compression(features: torch.Tensor) -> torch.Tensor:
    comp_ratios = torch.tensor(
        [compression_ratio(feat.cpu().numpy()) for feat in features]
    )
    sorted_indices = torch.argsort(comp_ratios, descending=True)
    print(f"{comp_ratios[sorted_indices][:3]} ... {comp_ratios[sorted_indices][-3:]}")
    return sorted_indices


def unet_patch(unet_patcher, *args):
    enabled, top_x, scale_factor = args

    if enabled is not True:
        return
    # Scale the top_x (sort_by_compression) hidden state features with
    # scale_factor in the original tensor

    def output_block_patch(h, hsp, transformer_options):
        print(f"{transformer_options['block']} Step: {state.sampling_step}")
        for cond in hsp:
            sorted_indices = sort_by_compression(cond)
#            if torch.all(cond[sorted_indices] == cond[sorted_indices][0]):
#                continue
#            cond[sorted_indices[:top_x]] *= scale_factor
            cond[sorted_indices[:top_x]] *= scale_factor

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class EntropicEnhancer(scripts.Script):

    def title(self):
        return "Entropic Enhancer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(
            False,
            label=self.title(),
        ) as enabled:
            top_x_slider = gr.Slider(
                min_inter=1,
                max_inter=100,
                step=1,
                value=1,
                label="Top X",
            )

            scale_factor_slider = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=1.0,
                label="Scale Factor",
            )

        return [enabled, top_x_slider, scale_factor_slider]

    @debug_log
    def process_before_every_sampling(self, p, *args, **kwargs):
        enabled, top_x, scale_factor = args

        if enabled is not True:
            return False

        unet = p.sd_model.forge_objects.unet
        unet = unet_patch(unet, *args)
        p.sd_model.forge_objects.unet = unet

        return True

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return