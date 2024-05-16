import os
import matplotlib.pyplot as plt
import io
from PIL import Image


def save_images(images, save_folder, names=None):
    assert names is None or len(images) == len(names)
    os.makedirs(save_folder, exist_ok=True)
    for i, img in enumerate(images):
        if names is not None:
            img.save(os.path.join(save_folder, f"{names[i]}.png"))
        else:
            img.save(os.path.join(save_folder, f"{i}.png"))


def image_grid_as_image(images, n_rows, n_cols):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    for i in range(n_rows):
        for j in range(n_cols):
            ax[i, j].imshow(images[i * n_cols + j], cmap='gray')
            ax[i, j].axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def images_to_gif(images, save_path, duration=40):
    images[0].save(save_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
