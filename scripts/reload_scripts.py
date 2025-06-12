import gradio as gr

from modules import scripts,shared,initialize,script_callbacks
from modules.ui_gradio_extensions import reload_javascript
import modules.infotext_utils as parameters_copypaste

class ReloadScripts(scripts.Script):
    def title(self):
        return "Reload Scripts"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(label="Reload UI", open=True):
            # button reload
            reload_button = gr.Button(value="Reload")
            reload_button.click(
                fn=self.reload_scripts,
                _js='restart_reload',
                inputs=[],
                outputs=[],
            )
    
    def reload_scripts(self):
        print("reload UI")
        scripts.load_scripts()
        print("reload UI request DONE")
