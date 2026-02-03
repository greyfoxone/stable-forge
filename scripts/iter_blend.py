import gradio as gr
import modules.scripts as scripts
from modules import processing
from modules.shared import opts
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
                maximum=20,
                value=1,
                step=1,
                interactive=True,
            )
            blend_start = gr.Slider(
                label="Blend start percent",
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                interactive=True,
            )
            blend_end = gr.Slider(
                label="Blend end percent",
                minimum=0,
                maximum=100,
                value=100,
                step=1,
                interactive=True,
            )
            output_intermediates = gr.Checkbox(
                label="Output all intermediates", value=True, interactive=True
            )
        return [enabled, loop_count, blend_start, blend_end, output_intermediates]

    def _blend_images(self, images, original, alpha):
        """Blend images with original using specified alpha."""
        blended = []
        for img in images:
            orig_resized = original.resize(img.size)
            if orig_resized.mode != img.mode:
                if orig_resized.mode == "RGBA" and img.mode == "RGB":
                    img = img.convert("RGBA")
                elif orig_resized.mode == "RGB" and img.mode == "RGBA":
                    orig_resized = orig_resized.convert("RGB")
            blended_img = Image.blend(img, orig_resized, alpha)
            blended.append(blended_img)
        return blended

    def _run_blending_loop(
        self, p, original, steps, loop_count, blend_start, blend_end, output_intermediates
    ):
        """Runs the iterative blending loop."""
        start_alpha = blend_start / 100.0
        end_alpha = blend_end / 100.0

        current_inputs = p.init_images.copy()
        p.all_images = [current_inputs]
        p.all_captions = ["Original" for _ in current_inputs]

        for iter in range(int(loop_count) - 1):
            print("\n\n" + ("*" * 5) + f" Loop {iter + 1}" + ("*" * 5))
            alpha = start_alpha + (end_alpha - start_alpha) * (iter / max(1, loop_count - 1))
            p.init_images = current_inputs
            p.steps = steps
            inner_processed = processing.process_images(p)
            
            # Blend outputs with original
            blended = self._blend_images(inner_processed.images, original, alpha)
            if output_intermediates or iter == int(loop_count) - 2:
                p.all_images.append(inner_processed.images)
                p.all_captions.append(f"Loop {iter + 1} Before Blending")
                p.all_images.append(blended)
                p.all_captions.append(f"Loop {iter + 1} After Blending, {alpha} blend")

            current_inputs = blended

        return p.all_images, p.all_captions

    def process(self, p, enabled, loop_count, blend_start, blend_end, output_intermediates):
        if not enabled:
            return

        if hasattr(p, "blend_running") and p.blend_running:
            return

        opts.img2img_fix_steps = True
        p.blend_running = True

        original = p.init_images[0].copy()
        steps = p.steps

        # Run the blending loop
        all_images, all_captions = self._run_blending_loop(
            p, original, steps, loop_count, blend_start, blend_end, output_intermediates
        )

        # Store results
        p.all_images = all_images
        p.all_captions = all_captions

        p.blend_running = False

    def postprocess(
        self, p, processed, enabled, loop_count, blend_start, blend_end, output_intermediates
    ):
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