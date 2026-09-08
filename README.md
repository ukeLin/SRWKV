# ✨ SRWKV

Official PyTorch implementation of **SRWKV: A Shape-Guided RWKV With Adaptive Receptive Fields for Efficient Medical Image Segmentation**.

> 🎉 Accepted by **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**.

[Installation](#-installation) · [Weights](#-model-weights) · [Datasets](#-dataset-resources) · [Training](#-training) · [Evaluation](#-evaluation) · [Citation](#-citation)

## 📂 Overview

```text
SRWKV-main/
  dataset.py
  loss.py
  train.py
  test.py
  utils.py
  model/
    SRWKV.py
    ...
```

## 🔧 Installation

```bash
git clone https://github.com/ukeLin/SRWKV.git
cd SRWKV
pip install -r requirements.txt
pip install ninja
```

The WKV CUDA extension is compiled automatically during the first run. Please make sure that PyTorch, CUDA, and the local CUDA toolkit are compatible.

## 📦 Model Weights

Download the encoder pretrained weights from [Baidu Netdisk](https://pan.baidu.com/s/1Bmf6reZJoWySkWqI5KjqxQ?pwd=7fvn) (code: `7fvn`) and place them at:

```text
model/pretrained/net_S.pth
```

## 🗂️ Dataset Resources

- ISIC17: [Google Drive](https://drive.google.com/file/d/1ZTOVI5Vp3KTQFDt5moJThJ_xYp2pKBAK/view?usp=sharing)
- ISIC18: [Google Drive](https://drive.google.com/file/d/1AOpPgSEAfgUS2w4rCGaJBbNYbRh3Z_FQ/view?usp=sharing)
- Synapse: [Google Drive](https://drive.google.com/file/d/1-eDXzTgXrTTo7hcrWZnh_wVEtB92PBNz/view?usp=sharing)
- ACDC: [Google Drive](https://drive.google.com/file/d/1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc/view)
- Polyp datasets: [Google Drive](https://drive.google.com/drive/folders/1XyjNgmPqikGxCaOdP0i6Xzf3deDIpbCV?usp=share_link)

`train.py` and `test.py` use paired image-mask folders:

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

For Synapse and ACDC, the commonly distributed files may use `npz`, `h5`, or volumetric formats. Please convert them to paired 2D image-mask folders before using the generic scripts, or implement a dataset-specific loader following `dataset.py`.

## 🚀 Training

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

Omit `--pretrained` to train from scratch. To resume, add `--resume log/SRWKV/checkpoints/latest.pth` to the command above.

Checkpoints (`latest.pth` and `best.pth`) are saved in `log/SRWKV/checkpoints/`.

## 📊 Evaluation

```bash
python test.py \
  --data-path data/dataset_name \
  --checkpoint log/SRWKV/checkpoints/best.pth \
  --img-size 256 \
  --batch-size 1
```

Reports Dice and mIoU on the validation split. Use the same image resolution for training and evaluation.

## 💙 Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@article{yusrwkv,
  title   = {SRWKV: A Shape-Guided RWKV With Adaptive Receptive Fields for Efficient Medical Image Segmentation},
  author  = {Yu, Chunlin and Li, Yinhao and Zhao, Zheng and Zhang, Taohong},
  journal = {IEEE Transactions on Neural Networks and Learning Systems}
}
```
