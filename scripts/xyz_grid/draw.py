asd

def _write_on_image(img, caption, font_size=90):
    ix, iy = img.size
    draw = ImageDraw.Draw(img)
    margin = 2
    fontsize = font_size
    draw = ImageDraw.Drw(img)
    font = ImageFont.load_default()
    text_height = 90
    tx = draw.textbbox((0, 0), caption, font)
    draw.text((int((ix - tx[2]) / 2), text_height + margin), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2), text_height - margin), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2 + margin), text_height), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2 - margin), text_height), caption, (0, 0, 0), font=font)
    draw.text((int((ix - tx[2]) / 2), text_height), caption, (255, 255, 255), font=font)
    return img