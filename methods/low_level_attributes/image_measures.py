from datasets.nsd.nsd import NaturalScenesDataset
import torch
import os
from PIL import Image
import numpy as np
from methods.low_level_attributes.xtc_network import UNet
import visualpriors
from tqdm import tqdm
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


def save_for_nsd(dataset_root, subject, function, save_name):
    nsd = NaturalScenesDataset(dataset_root, subject, 'all')
    for i in tqdm(range(len(nsd)), desc=f"Saving {save_name}"):
        f = os.path.join(nsd.root, nsd.df.iloc[i]["filename"])
        img = Image.open(f)
        output = function(img)
        f = f.replace("training_images", save_name).replace(".png", ".npy")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.save(f, output)

def save_depths(dataset_root, subject):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    zoe = zoe.to(device).eval()

    def depth(img):
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255
        with torch.no_grad():
            depth = zoe.infer(img_tensor).squeeze().detach().cpu().numpy()
            depth = depth[None,:,:]
        return depth

    save_for_nsd(dataset_root, subject, depth, "depth")

def save_surface_normals(dataset_root, subject):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normal_model = UNet() 
    normals_path = 'data/xtc_checkpoints/rgb2normal_consistency.pth'
    normal_model.load_state_dict(torch.load(normals_path, map_location=device))
    normal_model = normal_model.to(device).eval()

    def surface_normal(img):
        img_tensor = torch.tensor(np.array(img.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255
        with torch.no_grad():
            normal = normal_model(img_tensor)
            normal = torch.nn.functional.interpolate(
                normal,
                size=img.size,
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).clamp(min=0, max=1)
            normal = normal.squeeze().permute(1,2,0).detach().cpu().numpy()
            normal = np.moveaxis(normal, -1, 0)
        return normal
    
    save_for_nsd(dataset_root, subject, surface_normal, "surface_normal")

def save_gaussian_curvatures(dataset_root, subject):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def gaussian_curvature(img):
        img_tensor = torch.tensor(np.array(img.resize((256, 256))).transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255
        principal_curvature = (visualpriors.feature_readout(img_tensor * 2 - 1, 'curvature', device=device) / 2. + 0.5)[:,:2]
        principal_curvature = torch.nn.functional.interpolate(
            principal_curvature,
            size=img.size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).clamp(min=0, max=1)
        principal_curvature = principal_curvature.squeeze().permute(1,2,0).detach().cpu().numpy()
        gaussian_curvature = np.prod(principal_curvature, -1)[None,:,:]
        return gaussian_curvature
    
    save_for_nsd(dataset_root, subject, gaussian_curvature, "gaussian_curvature")

def compute_warmth(img):
    hue = np.array(img.convert('HSV'))[:,:,[0]]
    saturation = np.array(img.convert('HSV'))[:,:,[1]]
    value = np.array(img.convert('HSV'))[:,:,[2]]
    measure = np.cos(hue/255*np.pi*2) * (saturation / 255) * (value / 255)
    measure = ((measure + 1) / 2)
    measure = np.moveaxis(measure, -1, 0)
    return measure

def compute_saturation(img):
    measure = np.array(img.convert('HSV'))[:,:,[1]]
    measure = np.moveaxis(measure, -1, 0) / 255
    return measure

def compute_brightness(img):
    measure = np.array(img.convert('HSV'))[:,:,[2]]
    measure = np.moveaxis(measure, -1, 0) / 255
    return measure

def compute_entropy(img):
    image = img_as_ubyte(np.array(img.convert('L')))
    measure = entropy(image, disk(5)) / np.log2(256)
    measure = measure[None,:,:]
    return measure