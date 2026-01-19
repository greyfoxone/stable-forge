import torch
import gradio as gr
import modules.scripts as scripts
from modules import shared
from modules.shared import opts
from backend import memory_management

class EmbeddingArithmetic(scripts.Script):
    load_script = False
    def title(self):
        return "Embedding Arithmetic"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Embedding Arithmetic", open=False):
            enable = gr.Checkbox(label="Enable", value=False)
            prompt1 = gr.Textbox(label="Prompt 1", value="King", lines=1)
            prompt2 = gr.Textbox(label="Prompt 2", value="Gentleman", lines=1)
            factor = gr.Slider(minimum=-20.0, maximum=20.0, step=0.05, label="Factor", value=1.0)
        return [enable, prompt1, prompt2, factor]

    def process(self, p, enable, prompt1, prompt2, factor, **kwargs):
        if not enable or not prompt1.strip() or not prompt2.strip():
            return

        model = shared.sd_model
        bs = p.batch_size

        memory_management.load_model_gpu(model.forge_objects.clip.patcher)

        with torch.no_grad():
            cond_l_1 = model.text_processing_engine_l([prompt1])
            cond_l_2 = model.text_processing_engine_l([prompt2])
            diff_l = factor * (cond_l_1 - cond_l_2).repeat(bs, 1, 1)

            cond_g_1, pooled_1 = model.text_processing_engine_g([prompt1])
            cond_g_2, pooled_2 = model.text_processing_engine_g([prompt2])
            diff_g = factor * (cond_g_1 - cond_g_2).repeat(bs, 1, 1)
            diff_pooled = factor * (pooled_1 - pooled_2).repeat(bs, 1)

        original_get = model.get_learned_conditioning

        def new_get(prompt):
            is_neg = getattr(prompt, "is_negative_prompt", False)
            if is_neg:
                return original_get(prompt)
            cond_l = diff_l
            cond_g = diff_g
            clip_pooled = diff_pooled

            width = getattr(prompt, "width", 1024) or 1024
            height = getattr(prompt, "height", 1024) or 1024
            crop_w = opts.sdxl_crop_left
            crop_h = opts.sdxl_crop_top
            target_width = width
            target_height = height

            device = clip_pooled.device
            out = [
                model.embedder(torch.tensor([height], device=device)),
                model.embedder(torch.tensor([width], device=device)),
                model.embedder(torch.tensor([crop_h], device=device)),
                model.embedder(torch.tensor([crop_w], device=device)),
                model.embedder(torch.tensor([target_height], device=device)),
                model.embedder(torch.tensor([target_width], device=device))
            ]

            flat = torch.flatten(torch.cat(out)).unsqueeze(0).repeat(clip_pooled.shape[0], 1).to(device)

            cond = dict(
                crossattn=torch.cat([cond_l, cond_g], dim=2),
                vector=torch.cat([clip_pooled, flat], dim=1),
            )
            return cond

        model.get_learned_conditioning = new_get