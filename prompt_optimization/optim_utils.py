import copy
from dataclasses import dataclass
from statistics import mean
from typing import List

import open_clip
import torch
from open_clip import CLIP, SimpleTokenizer
from PIL import Image
from sentence_transformers.util import (dot_score, normalize_embeddings,
                                        semantic_search)
from torch import nn
from torchvision.transforms import Compose


@dataclass
class ConfigPromptOptimization:
    iter: int
    lr: float
    weight_decay: float
    prompt_len: int
    prompt_bs: int
    loss_weight: float
    batch_size: int
    print_step: int
    print_new_best: bool


def optimize_prompt(
    clip_model: CLIP,
    clip_preprocess: Compose,
    target_images: List[Image.Image],
    config: ConfigPromptOptimization,
    device: str,
):
    token_embedding = clip_model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    # get target features
    all_target_features = get_target_feature(
        clip_model, clip_preprocess, target_images, device
    )
    # optimize prompt
    learned_prompt = optimize_prompt_loop(
        clip_model, tokenizer, token_embedding, all_target_features, config, device
    )
    return learned_prompt


def get_target_feature(
    clip_model: CLIP,
    clip_preprocess: Compose,
    target_images: List[Image.Image],
    device: str,
):
    with torch.no_grad():
        curr_images = [clip_preprocess(i).unsqueeze(0) for i in target_images]
        curr_images = torch.concatenate(curr_images).to(device)
        all_target_features = clip_model.encode_image(curr_images)
    return all_target_features


def optimize_prompt_loop(
    clip_model: CLIP,
    tokenizer: SimpleTokenizer,
    token_embedding: nn.Embedding,
    all_target_features: torch.Tensor,
    config: ConfigPromptOptimization,
    device: str,
):
    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(
        tokenizer, token_embedding, config, device
    )
    p_dim = prompt_embeds.shape[-1]

    # get optimizer
    input_optimizer = torch.optim.AdamW(
        [prompt_embeds], lr=config.lr, weight_decay=config.weight_decay
    )

    best_sim = -torch.inf
    best_text = ""

    for step in range(config.iter):
        # randomly sample sample images and get features
        curr_indx = torch.randperm(len(all_target_features))
        target_features = all_target_features[curr_indx][0 : config.batch_size]

        # forward projection
        projected_embeds, nn_indices = nn_project(
            prompt_embeds, token_embedding, print_hits=False
        )

        # get cosine similarity score with all target features
        with torch.no_grad():
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
            logits_per_image, _ = forward_text_embedding(
                clip_model, padded_embeds, dummy_ids, all_target_features
            )
            scores_per_prompt = logits_per_image.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()
            best_indx = scores_per_prompt.argmax().item()

        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True

        # padding
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)

        logits_per_image, _ = forward_text_embedding(
            clip_model, padded_embeds, dummy_ids, target_features
        )
        cosim_scores = logits_per_image
        loss = 1 - cosim_scores.mean()
        loss = loss * config.loss_weight

        (prompt_embeds.grad,) = torch.autograd.grad(loss, [tmp_embeds])

        input_optimizer.step()
        input_optimizer.zero_grad()

        curr_lr = input_optimizer.param_groups[0]["lr"]
        cosim_scores = cosim_scores.mean().item()

        decoded_text = decode_ids(nn_indices, tokenizer)[best_indx]
        if config.print_step is not None and (
            step % config.print_step == 0 or step == config.iter - 1
        ):
            per_step_message = f"step: {step}, lr: {curr_lr}"
            if not config.print_new_best:
                per_step_message = f"\n{per_step_message}, cosim: {universal_cosim_score:.3f}, text: {decoded_text}"
            print(per_step_message)

        if best_sim * config.loss_weight < universal_cosim_score * config.loss_weight:
            best_sim = universal_cosim_score
            best_text = decoded_text
            if config.print_new_best:
                print(f"new best cosine sim: {best_sim}")
                print(f"new best prompt: {best_text}")

    if config.print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")

    return best_text


def initialize_prompt(tokenizer, token_embedding, config, device):
    prompt_len = config.prompt_len

    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(
        len(tokenizer.encoder), (config.prompt_bs, prompt_len)
    ).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(
        " ".join(["<start_of_text>"] * prompt_len)
    )
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * config.prompt_bs).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False

    return prompt_embeds, dummy_embeds, dummy_ids


def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz, seq_len, emb_dim = curr_embeds.shape

        # Using the sentence transformers semantic search which is
        # a dot product exact kNN search between a set of
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1, emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds)  # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)

        hits = semantic_search(
            curr_embeds,
            embedding_matrix,
            query_chunk_size=curr_embeds.shape[0],
            top_k=1,
            score_function=dot_score,
        )

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")

        nn_indices = torch.tensor(
            [hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device
        )
        nn_indices = nn_indices.reshape((bsz, seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append("|".join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def forward_text_embedding(
    model, embeddings, ids, image_features, avg_text=False, return_feature=False
):
    text_features = encode_text_embedding(model, embeddings, ids, avg_text=avg_text)

    if return_feature:
        return text_features

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    # logit_scale = self.logit_scale.exp()
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


def encode_text_embedding(model, text_embedding, ids, avg_text=False):
    cast_dtype = model.transformer.get_cast_dtype()

    x = text_embedding + model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    if avg_text:
        x = x[torch.arange(x.shape[0]), : ids.argmax(dim=-1)]
        x[:, 1:-1]
        x = x.mean(dim=1) @ model.text_projection
    else:
        x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)] @ model.text_projection

    return x
