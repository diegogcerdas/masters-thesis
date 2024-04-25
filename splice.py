import torch
import os
import numpy as np
import torch.nn as nn
from sklearn import linear_model

class SPLICE(nn.Module):
    """Decomposes images into a sparse nonnegative linear combination of concept embeddings

    Parameters
    ----------
    image_mean : torch.tensor
        A {CLIP dimensionality} sized tensor measuring the average offset of all image embeddings for the provided CLIP backbone.
    dictionary : torch.tensor
        A {num_concepts x CLIP dimensionality} matrix used as the dictionary for the sparse nonnegative linear solver.
    clip : torch.nn.module, optional
        A CLIP backbone that implements encode_image() and encode_text(). If none, assumed that inputs to model are already CLIP embeddings (useful when working on large datasets where you don't want to forward pass through CLIP each time).
    solver : str, optional
        Either 'admm' or 'skl', by default 'skl'
    l1_penalty : float, optional
        The l1 penalty applied to the solver. Increase this for sparser solutions.
    return_weights : bool, optional
        Whether the model returns a sparse vector in {num_concepts} or the dense reconstructed embeddings, by default False
    decomp_text : bool, optional
        Whether the text encoder should also run decomposition, by default False
    text_mean : _type_, optional
        If decomposing text, a {CLIP dimensionality} sized tensor measuring the average offset of all text embeddings for the provided CLIP backbone. Only useful if decomp_text is True, by default None
    device : str, optional
        Torch device, "cuda", "cpu", etc. by default "cpu"
    """
    def __init__(self, dictionary, l1_penalty=0.01, device="cpu"):
        super().__init__()
        self.device = device
        self.dictionary = dictionary.to(self.device)
        self.l1_penalty = l1_penalty
        self.l1_penalty = l1_penalty/(2*1024)  # skl regularization is off by a factor of 2 times the dimensionality of the CLIP embedding. See SKL docs.

    def decompose(self, embedding):
        clf = linear_model.Lasso(alpha=self.l1_penalty, fit_intercept=False, positive=True, max_iter=10000, tol=1e-6)
        skl_weights = []
        for i in range(embedding.shape[0]):
            clf.fit(self.dictionary.T.cpu().numpy(), embedding[i,:].cpu().numpy())
            skl_weights.append(torch.tensor(clf.coef_))
        weights = torch.stack(skl_weights, dim=0).to(self.device)
        return weights

    def recompose_vector(self, weights):
        recon_vector = weights@self.dictionary
        recon_vector = torch.nn.functional.normalize(recon_vector, dim=1)
        return recon_vector
    
    def encode_vector(self, vector):
        weights = self.decompose(vector)
        recon_vector = self.recompose_vector(weights)
        return (weights, torch.diag(recon_vector @ vector.T).sum())

def load(
    concepts_dir: str,
    device = "cuda" if torch.cuda.is_available() else "cpu", 
    **kwargs
):

    filenames = [int(f.replace(".npy", "")) for f in os.listdir(concepts_dir) if f.endswith(".npy")]
    filenames = [os.path.join(concepts_dir, f"{f}.npy") for f in sorted(filenames)]

    concepts = []
    for filename in filenames:
            concept = torch.from_numpy(np.load(filename)).to(device)
            concepts.append(concept)

    concepts = torch.nn.functional.normalize(torch.stack(concepts).squeeze(), dim=1)
    concepts = torch.nn.functional.normalize(concepts-torch.mean(concepts, dim=0), dim=1)

    splice = SPLICE(
        dictionary=concepts,
        device=device,
        **kwargs
    )

    return splice

def decompose_vector(vector, splicemodel, device):

    splicemodel.eval()
    splicemodel.return_weights = True
    splicemodel.return_cosine = True

    (weights, cosine) = splicemodel.encode_vector(vector.to(device))
    l0_norm = torch.linalg.vector_norm(weights.squeeze(), ord=0).item()

    return weights, l0_norm, cosine.item()