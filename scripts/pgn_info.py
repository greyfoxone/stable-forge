import gradio as gr
import modules.infotext_utils as parameters_copypaste
from modules import images
from modules import scripts
from modules.helper import class_debug_log
from modules.helper import pvars
from modules.ui_common import plaintext_to_html


@class_debug_log
class PGNinfo(scripts.Script):
    def title(self):
        return "PGNinfo"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Blocks() as interface:
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label=self.title(),
                        show_download_button=False,
                        mirror_webcam=False,
                        source="upload",
                        type="pil",
                        image_mode="RGBA",
                    )

                with gr.Column():
                    extracted_info = gr.Markdown("Generation Info")
                    geninfo = gr.Markdown(visible=False)

            image_input.change(
                fn=lambda img: self.run_pnginfo(img),
                inputs=[image_input],
                outputs=[extracted_info,geninfo],
            )
            with gr.Row():
                for label in ["txt2img", "img2img", "inpaint"]:
                    button = gr.Button(label)
                    parameters_copypaste.register_paste_params_button(
                        parameters_copypaste.ParamBinding(
                            paste_button=button,
                            tabname=label,
                            source_text_component=geninfo,
                            source_image_component=image_input,
                        )
                    )
        return []

    def run_pnginfo(self, image):
        if image is None:
            return

        geninfo, items = images.read_info_from_image(image)
        items = {**{"parameters": geninfo}, **items}

        info = ""
        for key, text in items.items():
            info += (
                f"""
    <div>
    <p><b>{plaintext_to_html(str(key))}</b></p>
    <p>{plaintext_to_html(str(text))}</p>
    </div>
    """.strip()
                + "\n"
            )

        if len(info) == 0:
            message = "Nothing found in the image."
            info = f"<div><p>{message}<p></div>"
        return info,geninfo