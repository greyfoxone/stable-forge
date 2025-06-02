import random
import string

import gradio as gr
import torch
from modules import scripts
from modules.script_callbacks import on_cfg_denoiser
from modules.script_callbacks import remove_current_script_callbacks
from modules.ui_components import InputAccordion


def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    #    print(f"crow: {crow}   ccol: {ccol}")
    mask[
        ...,
        crow - threshold : crow + threshold,
        ccol - threshold : ccol + threshold,
    ] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


def patch_freeu_v2(unet_patcher, b, s, t):
    model_channels = unet_patcher.model.diffusion_model.config.get(
        "model_channels"
    )
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        process = FreeUForForge.doFreeU

        if process:
            #            print(f"transformer_options:{transformer_options['block']}")
            i = transformer_options["block"][1]
            #            print(f"h:{h.shape} -> {i} b,s = {b[i]} {s[i]} ")
            hidden_mean = h.mean(1).unsqueeze(1)
            # Describe what happens above
            B = hidden_mean.shape[0]
            hidden_max, _ = torch.max(
                hidden_mean.view(B, -1), dim=-1, keepdim=True
            )
            hidden_min, _ = torch.min(
                hidden_mean.view(B, -1), dim=-1, keepdim=True
            )
            hidden_mean = (
                hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)
            ) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

            h[:, : h.shape[1] // 2] = h[:, : h.shape[1] // 2] * (
                (b[i] - 1) * hidden_mean + 1
            )

            if hsp.device not in on_cpu_devices:
                try:
                    hsp = Fourier_filter(hsp, threshold=int(t[i]), scale=s[i])
                except BaseException:
                    print(
                        "Device",
                        hsp.device,
                        "does not support the torch.fft functions used in the FreeU node",
                    )

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


class FreeUForForge(scripts.Script):
    doFreeU = True

    def title(self):
        return "FreeU Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        s = []
        b = []
        t = []
        blocks = [
            [2, 1280, 36, 28],
            [2, 1280, 36, 28],
            [2, 1280, 36, 28],
            [2, 1280, 72, 56],
            [2, 640, 72, 56],
            [2, 640, 72, 56],
            [2, 640, 144, 112],
            [2, 320, 144, 112],
            [2, 320, 144, 112],
        ]
        with InputAccordion(
            False,
            label=self.title(),
            elem_id="extensions-freeu",
            elem_classes=["extensions-freeu"],
        ) as freeu_enabled:
            # add header row | amplify | tune down
            with gr.Row():
                gr.Markdown("Amplify (B)")
                gr.Markdown("Tune Down (S)")
                gr.Markdown("Threshold")

            for i in range(0, 9):
                t_max = int(blocks[i][2] / 2)
                random_string = "".join(
                    [random.choice(string.ascii_letters) for _ in range(5)]
                )

                with gr.Row():
                    slider_b = gr.Slider(
                        id=f"b_{i}_{random_string}",
                        label=f"B ",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.00,
                    )
                    slider_s = gr.Slider(
                        id=f"s_{i}_{random_string}",
                        label=f"S ",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.00,
                    )
                    slider_t = gr.Slider(
                        id=f"t_{i}_{random_string}",
                        label=f"T ",
                        minimum=1.0,
                        maximum=t_max,
                        step=1.0,
                        value=1.0,
                    )
                if (i + 1) % 3 == 0:
                    with gr.Row():
                        gr.Markdown(f"{blocks[i]}")

                b.append(slider_b)
                s.append(slider_s)
                t.append(slider_t)

            with gr.Row():
                freeu_start = gr.Slider(
                    label="Start step",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                )
            with gr.Row():
                freeu_end = gr.Slider(
                    label="End step",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                )

        return freeu_enabled, freeu_start, freeu_end, *b, *s, *t

    def denoiser_callback(self, params):
        thisStep = params.sampling_step / (params.total_sampling_steps - 1)

        if (
            thisStep >= FreeUForForge.freeu_start
            and thisStep <= FreeUForForge.freeu_end
        ):
            FreeUForForge.doFreeU = True
        else:
            FreeUForForge.doFreeU = False

    def process_before_every_sampling(
        self, p, freeu_enabled, freeu_start, freeu_end, *sliders, **kwargs
    ):

        if not freeu_enabled:
            return
        # chop off first 9 of sliders
        b = sliders[:9]
        s = sliders[9:18]
        t = sliders[18:]
        unet = p.sd_model.forge_objects.unet

        #   test if patchable
        model_channels = unet.model.diffusion_model.config.get("model_channels")

        if model_channels is None:
            gr.Info("freeU is not supported for this model!")
            return

        FreeUForForge.freeu_start = freeu_start
        FreeUForForge.freeu_end = freeu_end
        on_cfg_denoiser(self.denoiser_callback)

        unet = patch_freeu_v2(unet, b[::-1], s[::-1], t[::-1])

        p.sd_model.forge_objects.unet = unet

        #        # Below codes will add some logs to the texts below the image outputs on UI.
        #        # The extra_generation_params does not influence results.
        #        p.extra_generation_params.update(dict(
        #            freeu_enabled=freeu_enabled,
        #            freeu_b1=freeu_b1,
        #            freeu_b2=freeu_b2,
        #            freeu_s1=freeu_s1,
        #            freeu_s2=freeu_s2,
        #            freeu_start=freeu_start,
        #            freeu_end=freeu_end,
        #        ))

        return

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return