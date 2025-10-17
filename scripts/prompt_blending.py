import torch
import modules.scripts as scripts
from modules import shared, processing
from modules.shared import opts
from backend import memory_management

class EmbeddingArithmetic(scripts.Script):
    def title(self):
        return "Embedding Arithmetic"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process(self, p, *args):
        model = shared.sd_model
        bs = p.batch_size

        memory_management.load_model_gpu(model.forge_objects.clip.patcher)
        # model.set_clip_skip(shared.opts.clip_skip)

        with torch.no_grad():
            cond_l_k = model.text_processing_engine_l(["Arab Woman"])
            cond_l_g = model.text_processing_engine_l(["White Man"])
            diff_l = (cond_l_g - cond_l_k).repeat(bs, 1, 1)

            cond_g_k, pooled_k = model.text_processing_engine_g(["Arab Woman"])
            cond_g_g, pooled_g = model.text_processing_engine_g(["White Man"])
            diff_g = (cond_g_g - cond_g_k).repeat(bs, 1, 1)
            diff_pooled = (pooled_g - pooled_k).repeat(bs, 1)

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

            out = [
                model.embedder(torch.tensor([height])),
                model.embedder(torch.tensor([width])),
                model.embedder(torch.tensor([crop_h])),
                model.embedder(torch.tensor([crop_w])),
                model.embedder(torch.tensor([target_height])),
                model.embedder(torch.tensor([target_width]))
            ]

            flat = torch.flatten(torch.cat(out)).unsqueeze(0).repeat(clip_pooled.shape[0], 1).to(clip_pooled.device)

            cond = dict(
                crossattn=torch.cat([cond_l, cond_g], dim=2),
                vector=torch.cat([clip_pooled, flat], dim=1),
            )
            return cond

        model.get_learned_conditioning = new_get