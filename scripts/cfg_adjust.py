import io

import gradio as gr
import matplotlib.pyplot as plt
import modules.scripts as scripts
import numpy as np
from modules import processing
from modules import shared
from modules.infotext_utils import PasteField
from PIL import Image


class Script(scripts.Script):
    cfg_fns = {
        "Linear": lambda start, end, x: start + (end - start) * x,
        "Square": lambda start, end, x: start + (end - start) * x**2,
        "Square Root": lambda start, end, x: start + (end - start) * x**0.5,
        # other fns
        "x^4": lambda start, end, x: start + (end - start) * x**4,
        "x^0.25": lambda start, end, x: start + (end - start) * x**0.25,
    }

    def title(self):
        return "CFG Adjust"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            cfg_start = gr.Slider(
                minimum=1, maximum=14, step=0.5, label="CFG Start", value=7.5
            )
            cfg_end = gr.Slider(
                minimum=1, maximum=14, step=0.5, label="CFG End", value=3.0
            )
            curve = gr.Dropdown(
                label="CFG Curve",
                choices=self.cfg_fns.keys(),
                value="x^0.25",
            )
            curve_graph = gr.Image(type="pil", visible=False)
            curve.select(
                fn=self.update_graph,
                inputs=[curve, cfg_start, cfg_end],
                outputs=[curve_graph],
            )
        self.infotext_fields = [
            PasteField(enabled, f"{self.title()}_enabled"),
            PasteField(cfg_start, f"{self.title()}_cfg_start"),
            PasteField(cfg_end, f"{self.title()}_cfg_end"),
            PasteField(curve, f"{self.title()}_curve"),
        ]
        return [enabled, cfg_start, cfg_end, curve]

    def update_graph(self, curve, cfg_start, cfg_end):
        steps = np.linspace(0, 1, 100)
        cfg_values = self.cfg_fns[curve](cfg_start, cfg_end, steps)

        plt.figure(figsize=(6, 4))
        plt.plot(steps, cfg_values, label=curve)
        plt.xlabel("Progress")
        plt.ylabel("CFG Value")
        plt.title(f"CFG Curve: {curve}")
        plt.grid(True)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        graph_img = Image.open(buf)
        plt.close()

        return gr.update(value=graph_img, visible=True)

    def process_before_every_sampling(self, p, *args, **kwargs):
        (enabled, cfg_start, cfg_end, curve) = args

        if not enabled:
            return

        def cfg_hook(module, args, kwargs):
            x, sigma = args
            step = shared.state.sampling_step
            progress = step / (p.steps - 1) if p.steps > 1 else 0
            kwargs["cond_scale"] = self.cfg_fns[curve](
                cfg_start, cfg_end, progress
            )
            return args, kwargs

        model = p.sampler.model_wrap_cfg
        self._handle = model.register_forward_pre_hook(
            cfg_hook, with_kwargs=True
        )

    def postprocess(self, p, processed, *args, **kwargs):
        (enabled, cfg_start, cfg_end, curve) = args

        if not enabled:
            return

        # Generate CFG vs Steps graph
        steps = np.arange(p.steps)  # 0 to steps-1
        cfg_values = [
            self.cfg_fns[curve](
                cfg_start, cfg_end, step / (p.steps - 1) if p.steps > 1 else 0
            )
            for step in steps
        ]

        # Cleanup
        if hasattr(p, "_cfg_hook_handle"):
            self._handle.remove()

    def run(self, p, *args, **kwargs):
        (enabled, cfg_start, cfg_end, curve) = args

        if not enabled:
            return

        return processing.process_images(p)