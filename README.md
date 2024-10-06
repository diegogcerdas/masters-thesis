# MSc Thesis: Brain-Targeted Natural Scene Manipulation with CLIP and Diffusion Models

MSc Artificial Intelligence - University of Amsterdam
- *Author*: Diego García Cerdas
- *Daily supervisor*: Dr. Iris Groen<sup>1</sup>
- *External supervisor*: Dr. Gemma Roig<sup>2</sup>
- *Examiner*: Dr. Pascal Mettes<sup>1</sup>
- *Second Reader*: MSc. Christina Sartzetaki<sup>1</sup>

<sup>1</sup> Video & Image Sense Lab\
<sup>2</sup> CVAI Lab, Goethe University Frankfurtv

![image](data/preview.svg)

## Setup

Please follow these steps to setup the environment and data needed for using this repository.

1. Create Python environment.

```
conda create -n thesis python=3.10
pip install -r requirements.txt
conda activate thesis
```

2. Download the `rgb2normal_consistency.pth` checkpoint for [XTC network](https://github.com/EPFL-VILAB/XTConsistency) (Zamir et al., 2020) and place it in a local folder `./data/xtc_checkpoints`.

3. Setup the Natural Scenes Dataset (Allen et al., 2022) subset from the Algonauts 2023 Challenge (Gifford et al., 2023):

    1. Access the data by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSehZkqZOUNk18uTjRTuLj7UYmRGz-OkdsU25AyO3Wm6iAb0VA/viewform).
    2. Extract the subject-specific zip-files in a local folder `./data/NSD/`.
    3. Run `setup.py` to perform data preprocessing.

## Manipulating an example image

You can manipulate an image of your choice to maximize or minimize activations in a region of interest through:

```
python run_img --img_path "dog.png" --prompt "a photo of a dog" --subject 1 --roi "PPA"
```

## Running our main experiment

The data for our main experiment (Section 4) can be obtained through:
```
python main_experiment.py --subject 1 --roi "PPA"
```



