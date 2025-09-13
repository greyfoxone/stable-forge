import random
import string

import gradio as gr
import torch
from modules import scripts
from modules.script_callbacks import on_cfg_denoiser
from modules.script_callbacks import remove_current_script_callbacks
from modules.ui_components import InputAccordion


class NoiseReduction(scripts.Script):

    def title(self):
        return "Feature Noise Reduction"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

