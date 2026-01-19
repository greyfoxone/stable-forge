import gradio as gr
import modules.scripts as scripts
from modules import processing
from PIL import Image


class IterativeBlend(scripts.Script):
    def title(self):
        return "Iterative Blend"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion(label="Iterative Blend", open=False):
            loop_count = gr.Number(label="Loop count", value=1, minimum=1, step=1, interactive=True)
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
        return [loop_count, blend_percent, output_intermediates]

    def run(self, p, loop_count, blend_percent, output_intermediates):
        if loop_count < 1:
            loop_count = 1

        # Assume the first init_image is the original; resize others to match if
        # needed
        original = p.init_images[0].copy()

        current_inputs = p.init_images.copy()

        all_images = []
        all_prompts = []
        all_seeds = []
        infotexts = []

        original_n_iter = p.n_iter
        original_do_not_save_samples = p.do_not_save_samples
        p.do_not_save_samples = True  # Prevent saving intermediates automatically

        alpha = blend_percent / 100.0

        for iter in range(int(loop_count)):
            print(f"Loop {iter}")
            p.init_images = current_inputs
            processed = processing.process_images(p)

            if output_intermediates or iter == loop_count - 1:
                all_images += processed.images
                all_prompts += processed.all_prompts
                all_seeds += processed.all_seeds
                infotexts += processed.infotexts

            # Blend outputs with original
            blended = []
            for img in processed.images:
                orig_resized = original.resize(img.size)
                # alpha=0: img (output), alpha=1: original
                blended_img = Image.blend(img, orig_resized, alpha)
                blended.append(blended_img)

            current_inputs = blended

        p.do_not_save_samples = original_do_not_save_samples

        # Return the final Processed object
        return processing.Processed(
            p,
            images_list=all_images,
            seed=processed.seed,
            info=processed.info,
            subseed=processed.subseed,
            all_prompts=all_prompts,
            all_seeds=all_seeds,
            infotexts=infotexts,
            index_of_first_image=0,
        )