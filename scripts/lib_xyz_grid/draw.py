import csv
from io import StringIO
from itertools import chain

import modules.sd_models
import modules.sd_samplers
import modules.sd_vae
from PIL import ImageDraw
from PIL import ImageFont


def _write_on_image(img, caption, font_size=90):
    ix, iy = img.size
    draw = ImageDraw.Draw(img)
    margin = 2
    fontsize = font_size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text_height = 90
    tx = draw.textbbox((0, 0), caption, font)
    draw.text((int((ix - tx[2]) / 2), text_height +
              margin), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2), text_height -
              margin), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2 + margin),
              text_height), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2 - margin),
              text_height), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2), text_height),
              caption, (255, 255, 255), font=font)
    return img


def list_to_csv_string(data_list):
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()


def csv_string_to_list_strip(data_str):
    return list(
        map(
            str.strip,
            chain.from_iterable(
                csv.reader(
                    StringIO(data_str),
                    skipinitialspace=True)),
        )
    )



class SharedSettingsStackHelper(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        modules.sd_models.reload_model_weights()
        modules.sd_vae.reload_vae_weights()


