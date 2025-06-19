import os

import gi
import gradio as gr
import modules.scripts as scripts

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("GLib", "2.0")
from typing import Tuple

from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import Gtk
from gradio_imageslider import ImageSlider
from PIL import Image

os.environ["GRADIO_TEMP_DIR"] = os.getcwd()


class GrImageCompareSlider:
    def __init__(self):
        self.slider = None
        self.images = []
        self.init_image = None
        self.ui()

    def ui(self):
        with gr.Group():
            with gr.Accordion("compare", open=True) as slider_interface:
                self.slider_interface = slider_interface
                with gr.Row():
                    self.slider = ImageSlider(
                        label="slider!", type="pil", interactive=False
                    )
                with gr.Row():
                    gr.Button("Copy to Clipboard", variant="primary").click(
                        fn=self.copy_image_to_clipboard, inputs=[self.slider]
                    )

    def on_gallery_select(self, select_data: gr.SelectData, *args, **kwargs):
        print(select_data.index)
        print(self.images)
        if not self.images or not self.init_image:
            value = (self.init_image, self.images[0])
            return self.update_slider(value)
        
        value = (self.init_image, self.images[select_data.index])
        return self.update_slider(value)

    def on_gallery_change(self, event_data: gr.SelectData, *args, **kwargs):
        if not self.images or not self.init_image:
            value = (self.init_image, self.images[0])
            return self.update_slider(value)
        index = event_data.target.selected_index
        value = (
            self.init_image,
            (
                self.images[event_data.selected_index]
                if index is not None
                else self.images[0]
            ),
        )

        return self.update_slider(value)

    def update_slider(self, value):
        slider_update = gr.update(value=value, type="pil")
        slider_interface = gr.update(open=True)
        return [slider_update, slider_interface]

    def set_images(self, init_image, output_images):
        self.init_image = init_image
        self.images = output_images

    def output(self):
        return [self.slider, self.slider_interface]

    def copy_image_to_clipboard(
        self, slider_value: Tuple[Image.Image, Image.Image]
    ) -> None:
        image = slider_value[1]
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            image.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            image.width,
            image.height,
            image.size[0] * 3,
        )
        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clipboard.set_image(pixbuf)
        clipboard.store()


class Script(scripts.Script):
    def title(self):
        return "Image Comparison Slider"

    def show(self, is_img2img):
        return is_img2img and scripts.AlwaysVisible

    def after_component(self, component, **kwargs):
        if component.elem_id == "img2img_gallery":
            # Hook into gallery select event
            self.slider_ui = GrImageCompareSlider()

            component.select(
                fn=self.slider_ui.on_gallery_select,
                inputs=component,
                outputs=self.slider_ui.output(),
            )
            component.change(
                fn=self.slider_ui.on_gallery_change,
                inputs=component,
                outputs=self.slider_ui.output(),
            )

    def postprocess(self, p, processed, *args, **kwargs):
        if not hasattr(self, "slider_ui"):
            return
        init_image = p.init_images[0]
        self.slider_ui.set_images(init_image, processed.images)