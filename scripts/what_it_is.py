import gradio as gr
import modules.scripts as scripts
import torch
from modules import devices
from modules import processing
from modules import shared
from modules.shared import state
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

blip_image_eval_size = 384

def print_t(t,p):
    cond = p.c.batch[0][0].schedules[0].cond
    print(f"{t}:\n {', '.join([str(float(a)) for a in cond['crossattn'][0,:5]])}")


class Script(scripts.Script):
    def title(self):
        return "What it is"

    def show(self, is_img2img):
        return False #scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            enable = gr.Checkbox(label="Enable", value=False)
            weight = gr.Slider(
                minimum=0, maximum=1, step=0.05, label="Weight", value=0.1
            )
            loop_count = gr.Slider(
                minimum=1, maximum=20, step=1, label="Loopy Count", value=20
            )
        return [enable, weight, loop_count]

    def process_before_every_sampling(self, p, enable, weight, loop_count, **kwargs):
        if not enable:
            return
       
        if not hasattr(p,'extra_data'):
            p.extra_data = {'loop':0}
            print_t("First",p)
            return
        p.cached_c = [None,None]
        print_t("Before sampling",p)
        cond = p.c.batch[0][0].schedules[0].cond
        i_tensor_786 = p.extra_data['i_tensor'][0,0:77,:]
        # v_tensor =  (1 - weight) * p.extra_data['last_p_tensor_786'] + weight * i_tensor_786
        v_tensor =  p.extra_data['last_p_tensor_786'] + weight *  (p.extra_data['last_p_tensor_786']  - i_tensor_786)
        cond['crossattn'] = torch.cat([v_tensor, cond['crossattn'][:, 768:]], dim=1)
        print_t("After sampling",p)

    def postprocess(self, p, processed, enable, weight, loop_count):
        if not enable:
            return
        
        if p.extra_data['loop'] > 0:
            return processed
        
        p.cached_c = [None,None]
        cond = p.c.batch[0][0].schedules[0].cond
        p.extra_data['last_p_tensor_786']  = cond["crossattn"].clone()[:, :768]
        last_image = processed.images[0]
        
        for loop in range(loop_count):
            print(f"\nloop:{loop + 1}")
            p.cached_c = [None,None]
            p.extra_data['loop'] = loop + 1
            p.extra_data['i_tensor'] = self.interrogate_tensor(last_image)
            print_t("Before processing",p)
            processed_loop = processing.process_images(p)
            print_t("After processing",p)
            cond = p.c.batch[0][0].schedules[0].cond
            p.extra_data['last_p_tensor_786']  = cond["crossattn"].clone()[:, :768]
            last_image = processed_loop.images[0]
            processed.images.append(last_image)
        
        # processed = processing.Processed(p, processed.images, p.seed, processed.info)
        return processed
    
    def interrogate_tensor(self, image):
        shared.interrogator.load()
        
        blip_model = shared.interrogator.blip_model
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(image).unsqueeze(0).type(shared.interrogator.dtype).to(devices.device_interrogate)

        with torch.no_grad():
            i_tensor = blip_model.visual_encoder(gpu_image)

        shared.interrogator.unload()
        return i_tensor


    # def interrogate_tensor(self, image):
    #     shared.interrogator.load()

    #     clip_image = (
    #         shared.interrogator.clip_preprocess(image)
    #         .unsqueeze(0)
    #         .type(shared.interrogator.dtype)
    #         .to(devices.device_interrogate)
    #     )
    #     with torch.no_grad(), devices.autocast():
    #         i_tensor = shared.interrogator.clip_model.encode_image(clip_image)
    #         i_tensor /= i_tensor.norm(dim=-1, keepdim=True)

    #     shared.interrogator.unload()
    #     print(f"i_tensor: {i_tensor[0,:5]}")
    #     return i_tensor

  