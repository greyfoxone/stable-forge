from modules import scripts
from modules.ui_components import InputAccordion
import gradio as gr


def unet_patch(unet, cond):
    def patch(h, hsp, transformer_options):
        if "patches_replace" in transformer_options:
            patches = transformer_options["patches_replace"]
            for name, patch_dict in patches.items():
                if name == "attn1":
                    for block_key, patch in patch_dict.items():
                        if block_key[1] == 0:  # index 0
                            patch(cond)
        return h, hsp

    m = unet.clone()
    m.set_model_attn1_replace(patch)
    return m


class UNetLayerPromptScript(scripts.Script):
    load_script = False
    def title(self):
        return "UNET Layer Prompts"

    def show(self, is_ui):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            layer_prompt = gr.Textbox(
                label="Layer Prompt",
                placeholder="Enter prompt for UNET layers...",
                lines=3
            )
        return [enabled, layer_prompt]

    def process_before_every_sampling(self, p, enabled, layer_prompt, **kwargs):
        if not enabled:
            return
            
        cond = p.sd_model.get_learned_conditioning([layer_prompt])
        
        unet = p.sd_model.forge_objects.unet
        unet = unet_patch(unet, cond)
        p.sd_model.forge_objects.unet = unet