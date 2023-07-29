# Multi-phase-Liver-Lesion-Segmentation

![Model architecture](docs/MULLET.svg)

## Introduction
This project is used for liver lesions segmentation from multi-phase abdomen CT images. MULLET has been deployed in multiple hospitals and used for clinical auxiliary diagnosis. If you have any business cooperation intentions, please contact Pujian Technology [http://www.pj-ai.com/](http://www.pj-ai.com/)


## Requirements

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
We recommend using anaconda to build a virtual Python environment. 
```
conda create -n mullet python=3.9
conda activate mullet
conda install pytorch torchvision numpy=1.24.3 tqdm=4.65.0 simpleitk=2.2.1 torchaudio scikit-image=0.19.3 pytorch-cuda=11.7 -c pytorch -c nvidia -c simpleitk
pip install argparse==1.4.0 segmentation_models_pytorch==0.3.2
```
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