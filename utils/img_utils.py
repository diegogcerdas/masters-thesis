import os


def save_images(images, save_folder, names=None):
    assert names is None or len(images) == len(names)
    os.makedirs(save_folder, exist_ok=True)
    for i, img in enumerate(images):
        if names is not None:
            img.save(os.path.join(save_folder, f"{names[i]}.png"))
        else:
            img.save(os.path.join(save_folder, f"{i}.png"))
