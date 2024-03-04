import copy
from dataclasses import dataclass
from typing import List

import open_clip
import torch
from open_clip import CLIP, SimpleTokenizer
from PIL import Image
from sentence_transformers.util import (dot_score, normalize_embeddings,
                                        semantic_search)
from torch import nn
from torchvision.transforms import Compose

from tqdm import tqdm


@dataclass
class ConfigPromptOptimization:
    iter: int
    lr: float
    weight_decay: float
    prompt_len: int
    prompt_bs: int
    loss_weight: float
    batch_size: int


def optimize_prompt(
    clip_model: CLIP,
    clip_preprocess: Compose,
    target_images: List[Image.Image],
    config: ConfigPromptOptimization,
    device: str,
):
    # Get target features
    all_target_features = get_target_feature(
        clip_model, clip_preprocess, target_images, device
    )
    # Optimize prompt
    learned_prompt = optimize_prompt_loop(
        clip_model, 
        open_clip.tokenizer._tokenizer, 
        clip_model.token_embedding, 
        all_target_features, 
        config, 
        device
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
    # Randomly initialize prompt
    ## prompt_embeds are trainable embeddings
    ## dummy_embeds are the embeddings of the dummy tokens
    ## initial_ids are the ids of the initial prompt
    prompt_embeds, dummy_embeds, initial_ids = initialize_prompt(
        tokenizer, token_embedding, config, device
    )
    p_dim = prompt_embeds.shape[-1]

    # Initialize optimizer
    input_optimizer = torch.optim.AdamW(
        [prompt_embeds], lr=config.lr, weight_decay=config.weight_decay
    )

    # Keep track of best prompt
    best_sim = -torch.inf
    best_text = ""

    for step in tqdm(range(config.iter)):

        # Random batch of target features
        curr_indx = torch.randperm(len(all_target_features))
        target_features = all_target_features[curr_indx][0 : config.batch_size]

        # Forward projection (semantic search) to get nearest neighbor features
        nn_embeds, nn_ids = forward_projection(prompt_embeds, token_embedding)

        # Get cosine similarity score with all target features
        with torch.no_grad():
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[initial_ids == -1] = nn_embeds.reshape(-1, p_dim)
            cosim_scores = forward_text_embedding(
                clip_model, padded_embeds, initial_ids, all_target_features
            )
            scores_per_prompt = cosim_scores.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()
            best_indx = scores_per_prompt.argmax().item()

        # Get cosine similarity score with current batch of target features
        tmp_embeds = nn_embeds.detach().clone()
        tmp_embeds.requires_grad = True
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[initial_ids == -1] = tmp_embeds.reshape(-1, p_dim)
        cosim_scores = forward_text_embedding(
            clip_model, padded_embeds, initial_ids, target_features
        )
        
        # Update prompt embeddings
        loss = (1 - cosim_scores.mean()) * config.loss_weight
        (prompt_embeds.grad,) = torch.autograd.grad(loss, [tmp_embeds])
        input_optimizer.step()
        input_optimizer.zero_grad()

        # Obtain current prompt text
        decoded_text = decode_ids(nn_ids, tokenizer)[best_indx]

        # Update best prompt
        if best_sim * config.loss_weight < universal_cosim_score * config.loss_weight:
            best_sim = universal_cosim_score
            best_text = decoded_text
            print(f"{step}: New best cosine sim: {best_sim}")
            print(f"{step}: New best prompt: {best_text}")
            print('-' * 50)

    return best_text


def initialize_prompt(
        tokenizer: SimpleTokenizer, 
        token_embedding: nn.Embedding, 
        config: ConfigPromptOptimization, 
        device: str
    ):
    prompt_len = config.prompt_len

    # Randomly initialize prompt embeddings
    prompt_ids = torch.randint(
        len(tokenizer.encoder), (config.prompt_bs, prompt_len)
    ).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # Initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(
        " ".join(["<start_of_text>"] * prompt_len)
    )
    initial_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    initial_ids = [i if i != 49406 else -1 for i in initial_ids]
    initial_ids = [49406] + initial_ids + [49407]
    initial_ids += [0] * (77 - len(initial_ids))
    initial_ids = torch.tensor([initial_ids] * config.prompt_bs).to(device)

    # For getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(initial_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False

    return prompt_embeds, dummy_embeds, initial_ids


def forward_projection(
        curr_embeds: torch.Tensor, 
        embedding_layer: nn.Embedding,
    ):
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

        nn_ids = torch.tensor(
            [hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device
        )
        nn_ids = nn_ids.reshape((bsz, seq_len))

        nn_embeds = embedding_layer(nn_ids)

    return nn_embeds, nn_ids


def decode_ids(
        input_ids: torch.Tensor, 
        tokenizer: SimpleTokenizer,
    ):
    input_ids = input_ids.detach().cpu().numpy()
    texts = []
    for input_ids_i in input_ids:
        texts.append(tokenizer.decode(input_ids_i))
    return texts


def forward_text_embedding(
    clip_model: CLIP, 
    text_embeddings: torch.Tensor, 
    ids: torch.Tensor, 
    image_features: torch.Tensor, 
):
    # Encode text to text-vision shared space
    text_features = encode_text_embedding(clip_model, text_embeddings, ids)

    # Normalize features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # Cosine similarity as logits
    sim_per_image = image_features @ text_features.t()

    return sim_per_image


def encode_text_embedding(
        clip_model: CLIP, 
        text_embeddings: torch.Tensor, 
        ids: torch.Tensor,
    ):
    cast_dtype = clip_model.transformer.get_cast_dtype()

    x = text_embeddings + clip_model.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)] @ clip_model.text_projection

    return x
