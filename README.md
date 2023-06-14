# Multi-phase-Liver-Lesion-Segmentation

![Model architecture](docs/MULLET.svg)

## Introduction

## requirements

```
torch~=2.0.1
numpy~=1.24.3
argparse~=1.4.0
tqdm~=4.65.0
SimpleITK~=2.2.1
segmentation_models_pytorch~=0.3.2
scikit-image~=0.19.3
```

## Install
For use as integrative **framework** (this will create a copy of the MULLET code on your computer so that you can modify it as needed):
```bash
git clone https://github.com/shenhai1895/Multi-phase-Liver-Lesion-Segmentation.git
cd Multi-phase-Liver-Lesion-Segmentation
pip install -e .
```

## Usage

A trained weight is
available [here](https://drive.google.com/file/d/1JJxwhunUES6D3cYH1BzA7elslU0ymlPj/view?usp=sharing).
Also, some cases is available [here](https://drive.google.com/file/d/125EvD3rp1BXqMlD2ahsib0jl-B68bEa2/view?usp=sharing) for evaluation. 

Run inference in terminal:

```bash
mullet_predict -i /path_of_input -o /path_of_output --checkpoint_path /path_of_checkpoint --devices 0 1 2 3
```