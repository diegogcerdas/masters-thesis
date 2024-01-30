import contextlib
import os
import urllib.request
import zipfile

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from pycocotools.coco import COCO


def download_coco_annotation_file(data_root: str):
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, "r")
    zip_file_object.extractall(path=data_root)


def coco_annotation(coco_id: int, data_root: str, expanded_nouns: bool = False):
    coco_annotation_file = os.path.join(data_root, "annotations", "{}_{}.json")
    # Download from https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata/experiments/nsd/
    # and place in data_root
    stim_descriptions = pd.read_csv(
        os.path.join(data_root, "nsd_stim_info_merged.csv"), index_col=0
    )
    subj_info = stim_descriptions.iloc[coco_id]
    annot_file = coco_annotation_file.format("captions", subj_info["cocoSplit"])
    with contextlib.redirect_stdout(None):
        coco = COCO(annot_file)
    coco_annot_IDs = coco.getAnnIds([subj_info["cocoId"]])
    coco_annot = coco.loadAnns(coco_annot_IDs)

    if expanded_nouns:
        nouns = set()
        for annot in coco_annot:
            caption = annot["caption"]
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            pos_tags = nltk.pos_tag(tokens)
            nns = [word for word, pos in pos_tags if pos == "NN"]
            nouns.update(nns)
            for n in nns:
                synsets = wn.synsets(n)
                if len(synsets) == 0:
                    continue
                hypernyms = hypernymy(n)
                nouns.update(hypernyms)
        return coco_annot, list(nouns)
    return coco_annot


def hypernymy(word: str):
    synset = wn.synsets(word)[0]
    h = lambda s: s.hypernyms()
    seen = set()

    def recurse(s):
        name = s.name().split(".")[0]
        if not name in seen:
            seen.add(name)
            for s1 in h(s):
                recurse(s1)

    recurse(synset)
    return list(seen)
