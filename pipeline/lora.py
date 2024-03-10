import pytorch_lightning as pl
import torch.utils.data as data

from datasets.dreambooth import DreamBoothDataset
from models.lora_dreambooth import LoRADreamBooth


def perform_lora(
    pretrained_model_name_or_path: str,
    lora_rank: int,
    train_text_encoder: bool,
    instance_prompt: str,
    validation_prompt: str,
    num_validation_images: int,
    validation_epochs: int,
    max_train_epochs: int,
    inference_steps: int,
    instance_data_root: str,
    output_dir: str,
    resolution: int,
    learning_rate: float,
    max_grad_norm: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: str,
) -> None:
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
        rank=lora_rank,
        train_text_encoder=train_text_encoder,
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
        gradient_clip_val=max_grad_norm,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(lora, train_dataloader)
