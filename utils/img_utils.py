import os


def save_images(images, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(save_folder, f"{i}.png"))
