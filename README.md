# SRWKV

Official PyTorch implementation of SRWKV for medical image segmentation.

This release provides a compact and reproducible codebase with the core SRWKV model, a generic training pipeline, a generic evaluation pipeline, and a paired image-mask dataset loader.

## Overview

SRWKV is designed for efficient medical image segmentation with shape-guided RWKV decoding. The repository is intentionally kept minimal for research reuse and fair reproduction.

```text
SRWKV-main/
  dataset.py
  loss.py
  train.py
  test.py
  utils.py
  model/
    SRWKV.py
    encoder.py
    ccm.py
    main_blocks.py
    module/
    mmcls_custom/models/backbones/
      srwkv.py
      da_shift.py
      scan_scan_inv.py
      cuda/
        wkv_op.cpp
        wkv_cuda.cu
```

## Installation

```bash
git clone https://github.com/your-name/SRWKV-main.git
cd SRWKV-main
pip install -r requirements.txt
pip install ninja
```

The WKV CUDA extension is compiled automatically during the first run. Please make sure that PyTorch, CUDA, and the local CUDA toolkit are compatible.

## Model Weights

The encoder pretrained weights can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1Bmf6reZJoWySkWqI5KjqxQ?pwd=7fvn).

After downloading, place the checkpoint in a convenient path, for example:

```text
model/pretrained/net_S.pth
```

## Dataset Resources

The following public links are provided for reproducibility:

- ISIC17: [Google Drive](https://drive.google.com/file/d/1ZTOVI5Vp3KTQFDt5moJThJ_xYp2pKBAK/view?usp=sharing)
- ISIC18: [Google Drive](https://drive.google.com/file/d/1AOpPgSEAfgUS2w4rCGaJBbNYbRh3Z_FQ/view?usp=sharing)
- Synapse: [Google Drive](https://drive.google.com/file/d/1-eDXzTgXrTTo7hcrWZnh_wVEtB92PBNz/view?usp=sharing)
- ACDC: [Google Drive](https://drive.google.com/file/d/1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc/view)
- Polyp datasets: [Google Drive](https://drive.google.com/drive/folders/1XyjNgmPqikGxCaOdP0i6Xzf3deDIpbCV?usp=share_link)

For this clean release, `train.py` and `test.py` use a unified paired image-mask format:

```text
data/
  dataset_name/
    train/
      images/
      masks/
    val/
      images/
      masks/
```

Mask files should have the same file stem as their corresponding images. Common image formats such as `png`, `jpg`, `jpeg`, `bmp`, `tif`, and `tiff` are supported.

For example, a dataset can be organized as:

```text
data/
  dataset_name/
    train/
      images/
        case_0000.jpg
      masks/
        case_0000.png
    val/
      images/
        case_0001.jpg
      masks/
        case_0001.png
```

For Synapse and ACDC, the commonly distributed files may use `npz`, `h5`, or volumetric formats. Please convert them to paired 2D image-mask folders before using the generic scripts, or implement a dataset-specific loader following `dataset.py`.

The commonly used preprocessed Synapse package is usually organized as:

```text
data/
  Synapse/
    lists/
      list_Synapse/
        all.lst
        test_vol.txt
        train.txt
    test_vol_h5/
      casexxxx.npy.h5
    train_npz/
      casexxxx_slicexxx.npz
```

## Training

Train SRWKV from scratch:

```bash
python train.py \
  --data-path data/dataset_name \
  --work-dir log/SRWKV \
  --img-size 256 \
  --batch-size 24 \
  --epochs 300 \
  --sp-weight 1.0
```

Train SRWKV with encoder pretrained weights:

```bash
python train.py \
  --data-path data/dataset_name \
  --work-dir log/SRWKV \
  --img-size 256 \
  --batch-size 24 \
  --epochs 300 \
  --sp-weight 1.0 \
  --pretrained model/pretrained/net_S.pth
```

Resume training from a checkpoint:

```bash
python train.py \
  --data-path data/dataset_name \
  --work-dir log/SRWKV \
  --resume log/SRWKV/checkpoints/latest.pth
```

Checkpoints are saved to:

```text
log/SRWKV/checkpoints/
  latest.pth
  best.pth
```

## Evaluation

Evaluate the best checkpoint:

```bash
python test.py \
  --data-path data/dataset_name \
  --checkpoint log/SRWKV/checkpoints/best.pth \
  --img-size 256 \
  --batch-size 1
```

The evaluation script reports Dice and mIoU on the validation split.

## Notes

- The default dataset loader expects paired image and mask folders.
- The CUDA WKV operator is required for SRWKV inference and training.
- Please keep the training and evaluation resolution consistent when reporting results.

## Citation

If this repository is useful for your research, please cite our paper. The BibTeX entry will be updated after publication.
