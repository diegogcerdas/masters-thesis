import os

import pytorch_lightning as pl
import torch
import torch.utils.data as data

from datasets.dreambooth import DreamBoothDataset
from models.lora_dreambooth import LoRADreamBooth

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    resolution = 512
    rank = 4
    instance_prompt = ""
    validation_prompt = ""
    num_validation_images = 25
    validation_epochs = 10
    max_train_epochs = 50
    inference_steps = 100
    output_dir = "./diffmorpher/train_0/outputs"
    instance_data_root = "./diffmorpher/train_0/data"
    seed = 0
    learning_rate = 1e-4
    batch_size = 8
    num_workers = 18
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    pl.seed_everything(seed)

    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_root,
        resolution=resolution,
    )
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    lora = LoRADreamBooth(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rank=rank,
        train_text_encoder=False,
        instance_prompt=instance_prompt,
        validation_prompt=validation_prompt,
        num_validation_images=num_validation_images,
        validation_epochs=validation_epochs,
        max_train_epochs=max_train_epochs,
        inference_steps=inference_steps,
        output_dir=output_dir,
        seed=seed,
        learning_rate=learning_rate,
        device=device,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_train_epochs,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(lora, train_dataloader)
