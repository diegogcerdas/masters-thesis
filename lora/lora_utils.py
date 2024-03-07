from pathlib import Path
import torch
import torch.utils.checkpoint
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler
import os


def log_validation(
    num_validation_images,
    validation_prompt,
    pipeline,
    pipeline_args,
    save_dir,
    device,
    seed,
    epoch=None,
):
    print(f"Running validation...\nGenerating {num_validation_images} images with prompt: {validation_prompt}")
    
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}
    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    for _ in range(num_validation_images):
        with torch.cuda.amp.autocast():
            image = pipeline(**pipeline_args, generator=generator).images[0]
            images.append(image)

    # Save images
    if epoch is not None:
        save_dir = os.path.join(save_dir, f"epoch_{epoch}")
    else:
        save_dir = os.path.join(save_dir, "final")
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(save_dir, f"{i}.png"))


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance images with the prompts for fine-tuning the model.
    It pre-processes the images and tokenizes the prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size,
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        text_inputs = tokenize_prompt(self.tokenizer, self.instance_prompt)
        example["instance_prompt_ids"] = text_inputs.input_ids
        return example


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    prompt_embeds = text_encoder(
        text_input_ids,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]
    return prompt_embeds