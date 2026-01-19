import gradio as gr
import modules.scripts as scripts
from modules import processing
from modules.ui_components import InputAccordion
from PIL import Image


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
                maximum=10,
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

        if hasattr(p, "iterative_blend_running") and p.iterative_blend_running:
            return

        p.iterative_blend_running = True

        p.iterative_all_images = []
        p.iterative_all_prompts = []
        p.iterative_all_seeds = []
        p.iterative_infotexts = []

        # Assume the first init_image is the original; resize others to match if
        # needed
        original = p.init_images[0].copy()

        current_inputs = p.init_images.copy()

        original_do_not_save_samples = p.do_not_save_samples
        p.do_not_save_samples = True  # Prevent saving intermediates automatically

        alpha = blend_percent / 100.0

        self.last_processed = None

        for iter in range(int(loop_count)):
            print(f"Loop {iter}")
            p.init_images = current_inputs
            inner_processed = processing.process_images(p)
            self.last_processed = inner_processed

            if output_intermediates or iter == loop_count - 1:
                p.iterative_all_images += inner_processed.images
                p.iterative_all_prompts += inner_processed.all_prompts
                p.iterative_all_seeds += inner_processed.all_seeds
                p.iterative_infotexts += inner_processed.infotexts

            # Blend outputs with original
            blended = []
            for img in inner_processed.images:
                orig_resized = original.resize(img.size)
                # alpha=0: img (output), alpha=1: original
                blended_img = Image.blend(img, orig_resized, alpha)
                blended.append(blended_img)

            current_inputs = blended

        p.do_not_save_samples = original_do_not_save_samples
        p.n_iter = 0  # Skip the outer sampling loop

    def postprocess(self, p, processed, enabled, loop_count, blend_percent, output_intermediates):
        if (
            not enabled
            or not hasattr(p, "iterative_blend_running")
            or not p.iterative_blend_running
        ):
            return

        processed.images = p.iterative_all_images
        processed.all_prompts = p.iterative_all_prompts
        processed.all_seeds = p.iterative_all_seeds
        processed.infotexts = p.iterative_infotexts

        if self.last_processed:
            processed.seed = self.last_processed.seed
            processed.info = self.last_processed.info
            processed.subseed = self.last_processed.subseed

        del p.iterative_blend_running
        del p.iterative_all_images
        del p.iterative_all_prompts
        del p.iterative_all_seeds
        del p.iterative_infotexts