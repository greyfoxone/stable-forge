import math
from datetime import datetime
from pathlib import Path

import gradio as gr
import modules.scripts as scripts
from modules.images import read_info_from_image
from modules.infotext_utils import parse_generation_parameters
from modules.paths_internal import cwd
from modules.shared import opts
from PIL import Image
from scripts.history.ui import UiHistoryPage
from scripts.history.ui import UiNavbar

IMAGES_PER_PAGE = 20


class ImageManager:
    def __init__(self, root_dirs: list[Path]):
        self.root_dirs = root_dirs
        self.image_files = []
        self.image_count = 0
        self.total_pages = 0
        self.collect()

    def collect(self):
        """Walks through directories and collects image file paths up to max_files."""
        self.image_count = 0
        self.image_files = []
        for root_dir in self.root_dirs:
            for file_path in root_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                ):
                    self.image_files.append(file_path)
                    self.image_count += 1
        self.total_pages = math.ceil(self.image_count / IMAGES_PER_PAGE)
        return self.image_files

    def get_image(self, index: int):
        """Returns InfoImage object for the image at given index."""
        if not 0 <= index < len(self.image_files):
            raise IndexError("Image index out of range")

        file_path = self.image_files[index]
        img = InfoImage(file_path)
        geninfo, items = read_info_from_image(img.image)
        img.geninfo = geninfo
        img.items = parse_generation_parameters(geninfo)
        return img

    def get_images_for_page(self, page_number: int):
        """Returns list of InfoImage objects for the given page number."""
        start_index = (page_number - 1) * IMAGES_PER_PAGE
        end_index = start_index + IMAGES_PER_PAGE

        images = []
        for i in range(start_index, min(end_index, len(self.image_files))):
            images.append(self.get_image(i))
        return images


class Script(scripts.Script):
    section = "tab-scripts"
    create_group = False
    #    load_script = False

    def title(self):
        return "History"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args):
        history = History(self.tabname)
        history.ui(self.tabname)


class History:
    def __init__(self, tabname):
        self.image_files = []
        outdir = opts.__getattr__(f"outdir_{tabname}_samples")
        root_dirs = [Path(cwd) / outdir, Path("/home/woj/Downloads/xxx")]
        self.image_manager = ImageManager(root_dirs)

    def ui(self, tabname):
        with gr.Tab(f"History {tabname}") as tab:
            # components
            self.ui_page_counter = gr.Textbox(
                value="1",
                max_lines=1,
                interactive=False,
                container=False,
                visible=False,
            )
            self.ui_navbar = UiNavbar()
            self.ui_history_page = UiHistoryPage(tabname, 10)

            # Events
            tab.select(fn=self.reload, inputs=[], outputs=[self.ui_page_counter])

            self.ui_navbar.page_number.change(
                fn=self.update,
                inputs=[self.ui_navbar.page_number],
                outputs=self.ui_history_page.output(),
            )

            self.ui_navbar.reload.click(fn=self.reload, inputs=[], outputs=[self.ui_page_counter])

            self.ui_page_counter.change(
                fn=self.update,
                inputs=[self.ui_navbar.page_number],
                outputs=self.ui_history_page.output(),
            )

    def reload(self):
        self.image_manager.collect()
        self.ui_navbar.total_pages = self.image_manager.total_pages
        return self.image_manager.total_pages

    def update(self, page_number):
        """Update the UI with images from the specified page."""
        images = self.image_manager.get_images_for_page(int(page_number))
        return self.ui_history_page.update(images)

class InfoImage:
    def __init__(self, path: Path):
        self.path = path
        self.geninfo = ""
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

        timestamp = self.path.stat().st_ctime
        time_str = datetime.fromtimestamp(timestamp).strftime("%d.%m.%y %H:%M")
        output += tr(td("time") + td(time_str))

        if "Prompt" in self.items:
            output += tr(td("Prompt:") + td(self.items["Prompt"]))

        if "Negative prompt" in self.items:
            output += tr(td("Negative Prompt:") + td(self.items["Negative prompt"]))

        output = table(output)

        return output