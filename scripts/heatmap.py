# Heat Map Script

from modules import scripts
from modules.processing import StableDiffusionProcessing, Processed
from modules.script_callbacks import remove_current_script_callbacks

# This Extension Script creates Heatmaps for each 
# block feature of the model during processing.
class Heatmap(scripts.Script):
    def title(self):
        return "Heatmaps"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        return

    def process_before_every_sampling(
        self, p: StableDiffusionProcessing, *args, **kwargs
    ):
        unet = p.sd_model.forge_objects.unet
        unet_patched = heatmap_patch(unet)
        p.sd_model.forge_objects.unet = unet_patched
        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return


def heatmap_patch(unet):
    def output_block_patch(h, hsp, transformer_options):
        return h, hsp
    
    unet_patched = unet.clone()
    unet_patched.set_model_output_block_patch(output_block_patch)
    return unet_patched