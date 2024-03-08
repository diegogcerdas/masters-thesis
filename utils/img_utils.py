import os

def save_images(images, save_folder):
    for i, img in enumerate(images):
        img.save(os.path.join(save_folder, f"{i}.png"))