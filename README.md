# SRWKV




## Prepare data & Pretrained model

### Dataset:

#### ISIC datasets
- The ISIC17 divided into a 7:3 ratio, can be found here {[GoogleDrive](https://drive.google.com/file/d/1ZTOVI5Vp3KTQFDt5moJThJ_xYp2pKBAK/view?usp=sharing)}.
- The ISIC18 divided into a 7:3 ratio, can be found here {[GoogleDrive](https://drive.google.com/file/d/1AOpPgSEAfgUS2w4rCGaJBbNYbRh3Z_FQ/view?usp=sharing)}.
- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

#### Synapse datasets

- For the Synapse dataset, can be found here {[GoogleDrive](https://drive.google.com/file/d/1-eDXzTgXrTTo7hcrWZnh_wVEtB92PBNz/view?usp=sharing)}.

- After downloading the datasets, you are supposed to put them into './data/Synapse/', and the file format reference is as follows.

- './data/Synapse/'
  - lists
    - list_Synapse
      - all.lst
      - test_vol.txt
      - train.txt
  - test_vol_h5
    - casexxxx.npy.h5
  - train_npz
    - casexxxx_slicexxx.npz

#### ACDC Dataset
- Download the preprocessed ACDC dataset from [Zenodo](https://zenodo.org/records/15038913) and move into `dataset/acdc/` folder.

#### Pretrained weights for the encoder can be downloaded at [BaiduDrive](https://pan.baidu.com/s/1Bmf6reZJoWySkWqI5KjqxQ?pwd=7fvn).

#### Checkpoints for SRWKV can be downloaded at [BaiduDrive](https://pan.baidu.com/s/1k-XR6MIOF0CSQxP-B9pflQ?pwd=7ay1).

## Acknowledgements

We thank the authors of [TransUNet](https://github.com/Beckschen/TransUNet), [RWKV-UNet](https://github.com/juntaoJianggavin/RWKV-UNet), [VRWKV](https://github.com/OpenGVLab/Vision-RWKV), [MSVM-UNet](https://github.com/gndlwch2w/msvm-unet), and [Samba](https://github.com/Jia-hao999/Samba) for making their valuable code & data publicly available.
