import io

import gradio as gr
import matplotlib.pyplot as plt
import modules.scripts as scripts
import numpy as np
from modules import shared
from modules.infotext_utils import PasteField
from PIL import Image


class Script(scripts.Script):
    cfg_fns_dict = {
        "x^0.25": lambda start, end, x: start + (end - start) * x**0.25,
        "x^4": lambda start, end, x: start + (end - start) * x**4,
        "Square Root": lambda start, end, x: start + (end - start) * x**0.5,
        "Square": lambda start, end, x: start + (end - start) * x**2,
        "Linear": lambda start, end, x: start + (end - start) * x,
        # other fns
    }

    def title(self):
        return "CFG Adjust"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=True, label=self.title()):
            enabled = gr.Checkbox(label="Enabled", value=False)
            cfg_start = gr.Slider(
                minimum=1, maximum=14, step=0.5, label="CFG Start", value=7.5
            )
            cfg_end = gr.Slider(
                minimum=1, maximum=14, step=0.5, label="CFG End", value=3.0
            )
            choices = list(self.cfg_fns_dict.keys())
            curve = gr.Dropdown(
                type="value",
                value="x^0.25",
                label="CFG Curve",
                choices=choices,
                #                allow_custom_value=True,
            )

            curve_graph = gr.Image(
                type="pil", visible=False, interactive=False, show_label=False
            )
            curve.select(
                fn=self.update_graph,
                inputs=[curve, cfg_start, cfg_end],
                outputs=[curve_graph],
            )

        output = [enabled, cfg_start, cfg_end, curve]

        self.infotext_fields = [
            PasteField(component,self. info_field_for(component))
            for component in output
        ]

        return output
    
    def info_field_for(self,component):
            return f"{self.title()}/{component.label}"
            
    def update_graph(self, curve, cfg_start, cfg_end):
        steps = np.linspace(0, 1, 100)
        cfg_values = self.cfg_fns_dict[curve](cfg_start, cfg_end, steps)

        plt.figure(figsize=(4, 3))
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
            kwargs["cond_scale"] = self.cfg_fns_dict[curve](
                cfg_start, cfg_end, progress
            )
            return args, kwargs

        model = p.sampler.model_wrap_cfg
        self._handle = model.register_forward_pre_hook(
            cfg_hook, with_kwargs=True
        )
        p.extra_generation_params.update(
            #dict from self.infotext_fields
            {self.infotext_fields[i][1] : value for i,value in enumerate(args)}
        )

    def postprocess(self, p, processed, *args, **kwargs):
        (enabled, cfg_start, cfg_end, curve) = args

        if not enabled:
            return
        print(processed.infotexts)
        # Cleanup
        if hasattr(p, "_cfg_hook_handle"):
            self._handle.remove()