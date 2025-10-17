import gradio as gr
import torch
from modules import scripts
from modules.script_callbacks import remove_current_script_callbacks
from modules.ui_components import InputAccordion


def unet_patch(unet, *args):
    def output_block_patch(h, hsp, transformer_options):
       
        return h, hsp
        
    m = unet.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class ExtensionScriptTemplate(scripts.Script):

    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        
        with InputAccordion(
            False,
            label=self.title(),
        ) as enabled:

            

        return enabled

    def process_before_every_sampling(
        self, p, *args, **kwargs
    ):

        if not enabled:
            return
      
        unet = p.sd_model.forge_objects.unet

        unet = unet_patch(unet, b, s, t)

        p.sd_model.forge_objects.unet = unet

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return
        
