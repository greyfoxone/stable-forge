import gradio as gr
from modules.infotext_utils import ParamBinding
from modules.infotext_utils import register_paste_params_button


class GrHistoryPage:
    def __init__(self, tabname="", num_rows=10):
        self.num_rows = num_rows
        self.rows = []
        self.ui(tabname)

    def ui(self, tabname):
        for _ in range(self.num_rows):
            self.rows.append(GrHistRow(tabname))

    def output(self):
        return [element for row in self.rows for element in row.output()]


class GrHistRow:
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
                    min_width=160,
                    width=160,
                    show_download_button=False,
                    container=True,
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

    def image_options(self, image):
        print(f"selected {image}")
        return [gr.update(min_width=160, width=160)]


class GrNavbar:
    def __init__(self):
        self.prev = None
        self.index = None
        self.next = None
        self.total_pages = 1
        self.ui()

    def ui(self):
        with gr.Row():
            self.start = gr.Button("|<",min_width=0)
            self.prev_few = gr.Button("<<",min_width=0)
            self.prev = gr.Button("<",min_width=0)
            self.reload = gr.Button("\U0001f504",min_width=0)
            self.page_display = gr.Textbox(
                value=f"1/{self.total_pages}",
                max_lines=1,
                interactive=False,
                container=False,
            )
            self.index = gr.Textbox(
                value="1",
                max_lines=1,
                interactive=False,
                container=False,
                visible=False,
            )
            
            self.next = gr.Button(">",min_width=0)
            self.next_few = gr.Button(">>",min_width=0)
            self.end = gr.Button(">|",min_width=0)

        # events
        self.prev.click(
            fn=self.prev_page, inputs=[self.index], outputs=self.index
        )
        self.next.click(
            fn=self.next_page, inputs=[self.index], outputs=self.index
        )
        self.start.click(fn=lambda: "1", outputs=self.index)
        self.end.click(fn=lambda: self.total_pages, outputs=self.index)

        self.prev_few.click(
            fn=self.prev_few_pages, inputs=[self.index], outputs=self.index
        )
        self.next_few.click(
            fn=self.next_few_pages, inputs=[self.index], outputs=self.index
        )

        self.index.change(
            fn=self.update_display,
            inputs=[self.index],
            outputs=self.page_display,
        )

    # prev_few and next_few
    def prev_few_pages(self, current_index):
        return str(max(1, int(current_index) - 5))

    def next_few_pages(self, current_index):
        return str(min(self.total_pages, int(current_index) + 5))

    def start(self, index):
        self.index.value = 1
        return 1

    def end(self):
        self.index.value = self.total_pages -1 
        return self.total_pages - 1 

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