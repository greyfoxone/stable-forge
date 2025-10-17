import torch
import gradio as gr
import modules.scripts as scripts
from modules import shared, devices
from modules import processing
from modules.processing import Processed

class EmbeddingArithmetic(scripts.Script):
    def title(self):
        return "Embedding Arithmetic"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        return []

    def run(self, p, *args):
        if hasattr(p, 'init_images') and p.init_images is not None:
            return processing.process_images(p)

        if p.prompt.strip() != "":
            return processing.process_images(p)

        model = shared.sd_model
        device = shared.device
        num_images_per_prompt = p.batch_size

        with torch.no_grad():
            king_dict = model.encode_prompt("King", device, num_images_per_prompt, False, "", clip_skip=p.clip_skip)
            gent_dict = model.encode_prompt("Gentleman", device, num_images_per_prompt, False, "", clip_skip=p.clip_skip)
            diff_cross = king_dict['prompt_embeds'] - gent_dict['prompt_embeds']
            diff_pooled = king_dict['pooled_prompt_embeds'] - gent_dict['pooled_prompt_embeds']

        original_encode = model.encode_prompt

        def new_encode(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None, clip_skip=None, process_embeds=True, **kwargs):
            if prompt == "" and prompt_embeds is None:
                if negative_prompt.strip():
                    neg_dict = original_encode(negative_prompt, device, num_images_per_prompt, False, negative_prompt, clip_skip=clip_skip)
                    neg_cross = neg_dict['prompt_embeds']
                    neg_pooled = neg_dict['pooled_prompt_embeds']
                else:
                    neg_cross = torch.zeros_like(diff_cross)
                    neg_pooled = torch.zeros_like(diff_pooled)

                if do_classifier_free_guidance:
                    return {
                        'prompt_embeds': diff_cross,
                        'negative_prompt_embeds': neg_cross,
                        'pooled_prompt_embeds': diff_pooled,
                        'negative_pooled_prompt_embeds': neg_pooled,
                    }
                else:
                    return {
                        'prompt_embeds': diff_cross,
                        'pooled_prompt_embeds': diff_pooled,
                    }
            return original_encode(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, clip_skip, process_embeds, **kwargs)

        model.encode_prompt = new_encode
        try:
            proc = processing.process_images(p)
        finally:
            model.encode_prompt = original_encode
        return proc