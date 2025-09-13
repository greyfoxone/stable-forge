from datetime import datetime
from pathlib import Path

import gradio as gr
import modules.scripts as scripts
from modules.images import read_info_from_image
from modules.infotext_utils import parse_generation_parameters
from modules.paths_internal import cwd
from modules.shared import opts
from PIL import Image
from scripts.history.ui import GrHistoryPage
from scripts.history.ui import GrNavbar


class Script(scripts.Script):
    section = "tab-scripts"
    create_group = False

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
        self.pages = []
        outdir = opts.__getattr__(f"outdir_{tabname}_samples")
        self.root_dirs = [Path(cwd) / outdir,Path('/home/woj/Downloads/xxx')]

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

        self.image_files.sort(
            key=lambda x: x.path.stat().st_ctime, reverse=True
        )
        self.pages = GroupedPages(self.image_files).pages

    def ui(self, tabname):
        with gr.Tab(f"History {tabname}") as tab:
            self.total_pages = gr.Textbox(
                value="1",
                max_lines=1,
                interactive=False,
                container=False,
                visible=False,
            )
            self.navbar = GrNavbar()
            self.page = GrHistoryPage(tabname)
            self.navbar.index.change(
                fn=self.update,
                inputs=[self.navbar.index],
                outputs=self.page.output(),
            )
            self.navbar.reload.click(
                fn=self.reload, inputs=[], outputs=[self.total_pages]
            )
            tab.select(fn=self.reload, inputs=[],outputs=[self.total_pages])
            self.total_pages.change(
                fn=self.update,
                inputs=[self.navbar.index],
                outputs=self.page.output(),)

    
    def reload(self):
        self.load_images()
        total_pages = len(self.pages)
        self.navbar.total_pages = total_pages
        return total_pages

    def update(self, page_number):
        return self.pages[int(page_number) - 1].update()


class HistoryPage:
    def __init__(self, rows, page_nr):
        self.rows = rows
        self.page_nr = page_nr

    def update(self):
        updates = []
        for row in self.rows:
            updates += row.update()
        return updates


class HistRow:
    def __init__(self, images, info):
        self.images = images
        self.info = info

    def update(self):
        images_updates = [
            gr.update(value=None, visible=False) for i in range(4)
        ]
        geninfos_updates = [
            gr.update(value=None, visible=False) for i in range(4)
        ]

        for i, image in enumerate(self.images):
            images_updates[i] = gr.update(
                value=image.path.as_posix(),
                min_width=160,
                width=160,
                visible=True,
                label=image.items["Seed"],
                show_label=True,
                container=True,
            )
            geninfos_updates[i] = gr.update(value=image.geninfo, visible=False)

        info_update = [gr.update(value=self.info, visible=True)]

        return images_updates + geninfos_updates + info_update


class GroupedPages:
    def __init__(self, image_files):
        self.pages = []
        self.image_files = image_files
        self.img_nr = -1

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

        return HistRow(images, last_image.info)

    def next_image(self, last_img=None):
        img_nr = self.img_nr + 1

        if img_nr > len(self.image_files) - 1:
            return None

        img = self.image_files[img_nr]

        if not last_img:
            return img

        if (
            img.items["Prompt"] == last_img.items["Prompt"]
            and img.items["Negative prompt"]
            == last_img.items["Negative prompt"]
        ):
            return img

        return None


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
            output += tr(
                td("Negative Prompt:") + td(self.items["Negative prompt"])
            )

        output = table(output)

        return output