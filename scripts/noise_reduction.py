import random
import string

import gradio as gr
import torch
from modules import scripts
from modules.script_callbacks import on_cfg_denoiser
from modules.script_callbacks import remove_current_script_callbacks
from modules.ui_components import InputAccordion


def patch_freeu_v2(unet_patcher, b, s, t):
    model_channels = unet_patcher.model.diffusion_model.config.get(
        "model_channels"
    )
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        if FreeUForForge.doFreeU is False:
            return
      
        block_nr = transformer_options["block"][1]
        if hsp.device not in on_cpu_devices:
            try:
                hsp = Fourier_filter(
                    hsp, threshold=int(t[block_nr]), scale=s[block_nr]
                )
                h = Fourier_filter(
                    h, threshold=int(t[block_nr]), scale=b[block_nr]
                )
            except BaseException:
                print(
                    "Device",
                    hsp.device,
                    "does not support the torch.fft functions used in the FreeU node",
                )

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class FreeUForForge(scripts.Script):

    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
     
    def denoiser_callback(self, params):
        thisStep = params.sampling_step / (params.total_sampling_steps - 1)

        if (
            thisStep >= FreeUForForge.freeu_start
            and thisStep <= FreeUForForge.freeu_end
        ):
            FreeUForForge.doFreeU = True
        else:
            FreeUForForge.doFreeU = False

    def process_before_every_sampling(
        self, p, freeu_enabled, freeu_start, freeu_end, *sliders, **kwargs
    ):

        if not freeu_enabled:
            return
        sliders = list(sliders)

        t = [sliders.pop() for _ in range(9)]
        s = [sliders.pop() for _ in range(9)]
        b = [sliders.pop() for _ in range(9)]

        unet = p.sd_model.forge_objects.unet

      

        if model_channels is None:
            gr.Info("freeU is not supported for this model!")
            return

        FreeUForForge.freeu_start = freeu_start
        FreeUForForge.freeu_end = freeu_end
        on_cfg_denoiser(self.denoiser_callback)

        unet = patch_freeu_v2(unet, b, s, t)

        p.sd_model.forge_objects.unet = unet

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return
        
