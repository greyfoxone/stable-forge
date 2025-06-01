import math
import os
from pathlib import Path

import gradio as gr
import modules.scripts as scripts
from modules.images import read_info_from_image
from modules.infotext_utils import ParamBinding
from modules.infotext_utils import PasteField
from modules.infotext_utils import parse_generation_parameters
from modules.infotext_utils import register_paste_params_button
from modules.paths_internal import cwd
from modules.shared import opts
from modules_forge.main_entry import ui_checkpoint
from PIL import Image
from datetime import datetime

class InfoImage:
    def __init__(self, path: Path):
        self.path = path
        self.geninfo = ''
        self.items = {}

    @property
    def image(self):
        return Image.open(self.path)

    @property
    def info(self):
        output = ""
        no_border = "style='border:none'"
        def tag(tag, content, args=""):
            return f"<{tag} {args}>{content}</{tag}>"

        def td(content, args=""):
            return tag("td", content, no_border + args)

        def tr(content, args=""):
            return tag("tr", content, no_border + args)

        def table(content, args=""):
            return tag("table", content, no_border + args)
        
        # add tr with self.path.stat().st_ctime (pathlib path)in format "dd.mm.YY HH:MM"
        output += tr(td("time") +td(datetime.fromtimestamp(self.path.stat().st_ctime).strftime("%d.%m.%y %H:%M")))
        
        if "Prompt" in self.items:
            output += tr(td("Prompt:") + td(self.items["Prompt"]))

        if "Negative prompt" in self.items:
            output += tr(td("Negative Prompt:") + td(self.items["Negative prompt"]))

        output = table(output)
        
        return output


class GrNavbar:
    def __init__(self, total_pages):
        self.prev = None
        self.index = None
        self.next = None
        self.total_pages = total_pages
        self.ui()

    def ui(self):
        with gr.Row():
            self.start = gr.Button("|<")
            self.prev_10 = gr.Button("<<")
            self.prev = gr.Button("<")
            self.reload = gr.Button("R")
            self.page_display = gr.Textbox(value=f"1/{self.total_pages}", max_lines=1, interactive=False, container=False)
            self.index = gr.Textbox(value="1", max_lines=1, interactive=False, container=False,visible=False)
            self.next = gr.Button(">")
            self.next_10 = gr.Button(">>")
            self.end = gr.Button(">|")
            
            # events
            self.prev.click(fn=self.prev_page, inputs=[self.index], outputs=self.index)
            self.next.click(fn=self.next_page, inputs=[self.index], outputs=self.index)
            self.start.click(fn=lambda: "1", outputs=self.index)  
            self.end.click(fn=lambda: self.total_pages, outputs=self.index)
            
            self.prev_10.click(fn=self.prev_10_pages, inputs=[self.index], outputs=self.index)
            self.next_10.click(fn=self.next_10_pages, inputs=[self.index], outputs=self.index)
            
            self.index.change(fn=self.update_display, inputs=[self.index], outputs=self.page_display)
            
    # prev_10 and next_10
    def prev_10_pages(self, current_index):
        return str(max(1, int(current_index) - 10))
    
    def next_10_pages(self, current_index):
        return str(min(self.total_pages, int(current_index) + 10))
        
    def start(self, index):
        self.index.value = 1
        return 1

    def end(self):
        self.index.value = self.total_pages
        return self.total_pages
        
    def prev_page(self, index):
        if int(index) > 1:
            return str(int(index) - 1)
        return index

    def next_page(self, index):
        if int(index) <self. total_pages:
            return str(int(index) + 1)
        return index
        
    def update_display(self, index):
        return f"{index}/{self.total_pages}"

class GrSkipbar:
    def __init__(self):
        self.prev = None
        self.index = None
        self.next = None
        self.ui()

    def ui(self):
        with gr.Row():
            self.prev = gr.Button("Prev")
            self.index = gr.Textbox(value="1", max_lines=1, interactive=False, container=False)
            self.next = gr.Button("Next")


class GrHistRow:
    def __init__(self, tabname):
        self.gr_images = []
        self.gr_info = None
        self.ui(tabname)

    def ui(self, tabname):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    for i in range(4):
                        gr_image = gr.Image(
                            value=None,
                            type="filepath",
                            height=300,
                            show_download_button=False,
                            container=False,
                            show_share_button=False,
                            interactive=False,
                            mirror_webcam=False,
                            visible=False,
                        )
                        gr_image.click = gr_image.select
                        self.gr_images.append(gr_image)
            with gr.Column():
                self.gr_info = gr.Markdown(value=None, visible=False)
                self.gr_geninfo = gr.Markdown(value=None, visible=False)
        for gr_image in self.gr_images:
            register_paste_params_button(ParamBinding(gr_image, tabname, source_text_component=self.gr_geninfo, source_image_component=gr_image.value))

    def output(self):
        return self.gr_images + [self.gr_info, self.gr_geninfo]


class GrHistoryPage:
    def __init__(self, tabname="", num_rows=5):
        self.num_rows = num_rows
        self.rows = []
        self.ui(tabname)

    def ui(self, tabname):
        for _ in range(self.num_rows):
            self.rows.append(GrHistRow(tabname))

    def output(self):
        return [element for row in self.rows for element in row.output()]


class HistRow:
    def __init__(self, images, info, geninfo):
        self.images = images
        self.info = info
        self.geninfo = geninfo

    def update(self):
        updates = [gr.update(value=None, visible=False) for i in range(4)]
        updates.append(gr.update(value=self.info, visible=True))
        updates.append(gr.update(value=self.geninfo, visible=False))

        for i, image in enumerate(self.images):
            updates[i] = gr.update(value=image.path.as_posix(), visible=True)

        return updates


class HistoryPage:
    def __init__(self, rows, page_nr):
        self.rows = rows
        self.page_nr = page_nr

    def update(self):
        updates = []
        for row in self.rows:
            updates += row.update()
        return updates


class GroupedPages:
    def __init__(self, image_files):
        self.pages = []
        self.image_files = image_files
        self.img_nr = 0

        page_nr = 1
        while (page := self.next_page(page_nr)) is not None:
            self.pages.append(page)
            page_nr += 1

    def next_page(self, page_nr):
        rows = []
        for r in range(5):
            row = self.next_row()
            if row:
                rows.append(row)
        if not rows:
            return None
            
        return HistoryPage(rows, page_nr)

    def next_row(self):
        last_image = self.next_image()
        
        if not last_image:
            return None
        
        self.img_nr += 1
        images = [last_image]
        
        for i in range(3):
            img = self.next_image(last_image)
            if not img:
                break
            self.img_nr += 1
            last_image = img
            images.append(img)
        
        return HistRow(images, last_image.info, last_image.geninfo)

    def next_image(self, last_img = None):
        img_nr = self.img_nr + 1

        if img_nr > len(self.image_files) - 1:
            return None

        img = self.image_files[img_nr]
        
        if not last_img:
            return img

        if img.items["Prompt"] == last_img.items["Prompt"]:
            return img

        return None


class History:
    def __init__(self, tabname):
        self.image_files = []
        self.pages = []
        outdir = opts.__getattr__(f"outdir_{tabname}_samples")
        self.root_dirs = [Path(cwd) / outdir]
        self.load_images()
        self.ui(tabname)

    def load_images(self):
        self.image_files = []
        for root_dir in self.root_dirs:
            for file_path in root_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                ):
                    img = InfoImage(file_path)
                    geninfo, items = read_info_from_image(img.image)
                    img.geninfo = geninfo
                    img.items = parse_generation_parameters(geninfo)
                    self.image_files.append(img)

        self.image_files.sort(key=lambda x: x.path.stat().st_ctime, reverse=True)
        self.pages = GroupedPages(self.image_files).pages

    def ui(self, tabname):
        self.navbar = GrNavbar(total_pages=len(self.pages))
        self.page = GrHistoryPage(tabname)
        self.navbar.index.change(fn=self.update, inputs=[self.navbar.index], outputs=self.page.output())
        self.navbar.reload.click(fn=self.reload,inputs=[], outputs=self.page.output())
        
    def reload(self):
        self.load_images()
        return self.update(1)
        
    def update(self, page_number):
        return self.pages[int(page_number) - 1].update()


class Script(scripts.Script):

    def title(self):
        return "History"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        pass

    def after_component(self, component, **kwargs):
        if component.elem_id == "txt2img_extra_tabs":
            with gr.Blocks() as txt2img_history_interface:
                with gr.Accordion(self.title(), open=False):
                    history = History("txt2img")
                    txt2img_history_interface.load(fn=history.update, inputs=[history.navbar.index], outputs=history.page.output())

        elif component.elem_id == "img2img_extra_tabs":
            with gr.Blocks() as img2img_history_interface:
                with gr.Accordion(self.title(), open=False):
                    history = History("img2img")
                    img2img_history_interface.load(fn=history.update, inputs=[history.navbar.index], outputs=history.page.output())