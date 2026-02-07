import gradio as gr
from modules.infotext_utils import ParamBinding
from modules.infotext_utils import register_paste_params_button

history_image_width = 320


class UiHistoryPage:
    def __init__(self, tabname="", num_rows=10):
        self.num_rows = num_rows
        self.rows = []
        self.ui(tabname)

    def ui(self, tabname):
        for _ in range(self.num_rows):
            self.rows.append(UiHistRow(tabname))

    def output(self):
        return [element for row in self.rows for element in row.output()]

    def update(self, info_images=None):
        """Update all rows with InfoImage list"""
        updates = []

        # Process each row
        for i, row in enumerate(self.rows):
            image_data = info_images[i] if info_images and i < len(info_images) else None
            row_updates = row.update(image_data)
            updates.extend(row_updates)

        return updates


class UiHistRow:
    def __init__(self, tabname):
        self.gr_images = []
        self.gr_geninfos = []
        self.param_display = None
        self.ui(tabname)

    def ui(self, tabname):
        with gr.Row():
            for i in range(4):
                gr_image = gr.Image(
                    value=None,
                    type="filepath",
                    min_width=history_image_width,
                    width=history_image_width,
                    scale=0,
                    show_download_button=False,
                    container=False,
                    show_label=True,
                    show_share_button=False,
                    interactive=False,
                    mirror_webcam=False,
                    visible=True,
                )
                # use existing parameter copy binding from select method
                # gr_image.select(
                #     fn=self.image_options, inputs=[gr_image], outputs=[gr_image]
                # )
                gr_image.click = gr_image.select
                # hidden parameter info for copy parameters
                gr_geninfo = gr.Markdown(value=None, visible=False)

                binding = ParamBinding(
                    gr_image,
                    tabname,
                    source_text_component=gr_geninfo,
                    source_image_component=gr_image.value,
                )
                register_paste_params_button(binding)

                self.gr_images.append(gr_image)
                self.gr_geninfos.append(gr_geninfo)

        with gr.Row():
            self.param_display = gr.Markdown(value=None, visible=False)

    def output(self):
        return self.gr_images + self.gr_geninfos + [self.param_display]

    def update(self, images=None, info=None):
        """Update row with image data and info"""
        updates = []

        # Handle images
        for i, gr_image in enumerate(self.gr_images):
            if images and i < len(images):
                updates.append(
                    gr.update(
                        value=(
                            images[i].path.as_posix() if hasattr(images[i], "path") else images[i]
                        ),
                        label=(
                            images[i].items.get("Seed", "") if hasattr(images[i], "items") else ""
                        ),
                        visible=True,
                    )
                )
            else:
                updates.append(gr.update(value=None, visible=False))

        # Handle geninfo
        for i, gr_geninfo in enumerate(self.gr_geninfos):
            if images and i < len(images) and hasattr(images[i], "geninfo"):
                updates.append(gr.update(value=images[i].geninfo, visible=False))
            else:
                updates.append(gr.update(value=None, visible=False))

        # Handle param display
        updates.append(gr.update(value=info, visible=bool(info)))

        return updates

    def image_options(self, image):
        print(f"selected {image}")
        return [gr.update(min_width=160, width=160)]


class UiNavbar:
    def __init__(self):
        self.prev = None
        self.page_number = None
        self.next = None
        self.total_pages = 1
        self.ui()

    def ui(self):
        with gr.Row():
            self.start = gr.Button("|<", min_width=0)
            self.prev_few = gr.Button("<<", min_width=0)
            self.prev = gr.Button("<", min_width=0)
            self.reload = gr.Button("\U0001f504", min_width=0)
            self.page_display = gr.Textbox(
                value=f"1/{self.total_pages}",
                max_lines=1,
                interactive=False,
                container=False,
            )
            self.page_number = gr.Textbox(
                value="1",
                max_lines=1,
                interactive=False,
                container=False,
                visible=False,
            )

            self.next = gr.Button(">", min_width=0)
            self.next_few = gr.Button(">>", min_width=0)
            self.end = gr.Button(">|", min_width=0)

        # events
        self.prev.click(fn=self.prev_page, inputs=[self.page_number], outputs=self.page_number)
        self.next.click(fn=self.next_page, inputs=[self.page_number], outputs=self.page_number)
        self.start.click(fn=lambda: "1", outputs=self.page_number)
        self.end.click(fn=lambda: self.total_pages, outputs=self.page_number)

        self.prev_few.click(
            fn=self.prev_few_pages, inputs=[self.page_number], outputs=self.page_number
        )
        self.next_few.click(
            fn=self.next_few_pages, inputs=[self.page_number], outputs=self.page_number
        )

        self.page_number.change(
            fn=self.update_display,
            inputs=[self.page_number],
            outputs=self.page_display,
        )

    # prev_few and next_few
    def prev_few_pages(self, current_index):
        return str(max(1, int(current_index) - 5))

    def next_few_pages(self, current_index):
        return str(min(self.total_pages, int(current_index) + 5))

    def start(self, index):
        self.page_number.value = 1
        return 1

    def end(self):
        self.page_number.value = self.total_pages - 1

    def prev_page(self, index):
        if int(index) > 1:
            return str(int(index) - 1)
        return index

    def next_page(self, index):
        if int(index) < self.total_pages:
            return str(int(index) + 1)
        return index

    def update_display(self, index):
        return f"{index}/{self.total_pages}"