import gradio as gr
import modules.scripts as scripts
from modules import processing
from modules.ui_components import InputAccordion
from PIL import Image
from modules.shared import opts, state

class IterativeBlend(scripts.Script):
    def title(self):
        return "Iterative Blend"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(
            False,
            label=self.title(),
        ) as enabled:
            loop_count = gr.Slider(
                label="Loop count",
                minimum=1,
                maximum=20,
                value=1,
                step=1,
                interactive=True,
            )
            blend_percent = gr.Slider(
                label="Blend percent (original contribution)",
                minimum=0,
                maximum=100,
                value=50,
                step=1,
                interactive=True,
            )
            output_intermediates = gr.Checkbox(
                label="Output all intermediates", value=True, interactive=True
            )
        return [enabled, loop_count, blend_percent, output_intermediates]

    def process(self, p, enabled, loop_count, blend_percent, output_intermediates):
        if not enabled:
            return

        if hasattr(p, "blend_running") and p.blend_running:
            return
            
        opts.img2img_fix_steps = True
        p.blend_running = True

        original = p.init_images[0].copy()
        steps = p.steps
        current_inputs = p.init_images.copy()
        p.all_images = [current_inputs]
        p.all_captions = ["Original" for _ in current_inputs]
        
        alpha = blend_percent / 100.0

        for iter in range(int(loop_count) - 1):
            print("\n\n" + ("*" * 5) + f" Loop {iter +1 }" + ("*" * 5))
            p.init_images = current_inputs
            p.steps = steps
            inner_processed = processing.process_images(p)

            p.all_images.append(inner_processed.images)
            p.all_captions.append(f"Loop {iter + 1} Before Blending")

            # Blend outputs with original
            blended = []
            for img in inner_processed.images:
                orig_resized = original.resize(img.size)
                if orig_resized.mode != img.mode:
                    if orig_resized.mode == "RGBA" and img.mode == "RGB":
                        img = img.convert("RGBA")
                    elif orig_resized.mode == "RGB" and img.mode == "RGBA":
                        orig_resized = orig_resized.convert("RGB")
                blended_img = Image.blend(img, orig_resized, alpha)
                blended.append(blended_img)
            if output_intermediates or iter == int(loop_count) -1:
                p.all_images.append(blended)
                p.all_captions.append(f"Loop {iter + 1} After {blend_percent}% Blending")
                
            current_inputs = blended
        p.blend_running = False

    #        p.n_iter = 0  # Skip the outer sampling loop

    def postprocess(self, p, processed, enabled, loop_count, blend_percent, output_intermediates):
        if not enabled:
            return

        if hasattr(p, "blend_running") and p.blend_running:
            all_images = p.all_images
            return processing.Processed(p, all_images, processed.seed, processed.info)
        
        all_images = []
        for i, images in enumerate(p.all_images):
            for img in images:
                all_images.append([img, p.all_captions[i]])
            
        processed.images = all_images
        del p.all_images
        del p.all_captions
        return processed