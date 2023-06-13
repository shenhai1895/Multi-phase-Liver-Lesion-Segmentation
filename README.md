# Multi-phase-Liver-Lesion-Segmentation

![Model architecture](docs/MULLET.svg)

## requirements

```
torch~=2.0.0
numpy~=1.19.2
torchvision~=0.12.0
SimpleITK~=2.2.0
scikit-image~=0.19.2
```

## Usage

A trained weight is also
available [here](https://drive.google.com/file/d/1JJxwhunUES6D3cYH1BzA7elslU0ymlPj/view?usp=sharing).

First enter the root dir of this project:

```sh
cd Multi-phase-Liver-Lesion-Segmentation
```

Run tests

```sh
python test/test_segmentor.py --test_dir "/your_path/Multi-phase-Liver-Lesion-Segmentation/data" --checkpoint_path "/your_path/model.pth" --n_ctx 3 --devices 0 
```
