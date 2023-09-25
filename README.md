# Salient Object Detection for DUTS dataset

This is the source code of the project "Salient Object Detection " of the course "Computer Vision Vin" 2023.

---
- [Salient Object Detection](#salient-object-detection)
  - [Abstract](#abstract)
  - [Folder Structure](#folder-structure)
  - [Model Zoo](#model-zoo)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Evaluating](#evaluating)
    - [Inference](#inference)
  - [Contributors](#contributors)
 
  ## Abstract 
Currently, existing image segmentation tasks mainly focus on segmenting objects with specific characteristics, e.g., salient, camouflaged, meticulous, or specific categories. Most of them have the same input/output formats, and barely use exclusive mechanisms designed for segmenting targets in their models, which means almost all tasks are dataset-dependent. Thus, it is very promising to formulate a category-agnostic DIS task for accurately segmenting objects with different structure complexities, regardless of their characteristics. Compared with semantic segmentation, the proposed DIS task usually focuses on images with single or a few targets, from which getting richer accurate details of each target is more feasible. 

In this project, we will investigate the powerful of salient object detection in the real world by experimenting it over a various methods to see whether and how it works with DUTS dataset.
## Folder Structure

```
CVVin23/
├── DUTS-TR/ - dataset for training
|   ├── image
│   ├── mask
├── DUTS-TE/ - dataset for testing
|   ├── image
│   ├── mask
├── model/ - u2net, u2net-dp, isnet, isnet-dp

├── metric/ 
|   ├── loss/ - loss for training
|   ├── metric- metric for evaluation
|
├── train_GTEncoder/ -training ground trust encoder for 
│   
├── train/ - train u2net, u2net-dp, isnet, isnet-dp
|
├── evaluation/ - evaluate models.
|
├── data_loader/ - custom dataset and dataloader
│
├── saved/ - trained models config, log-dir and logging output
│
├── visualization/ - for visualize loss function in training data and testing data
|
├── utils/ - utility functions
```

## Model Zoo 
<summary></summary>

<table style="margin-left:auto;margin-right:auto;font-size:1.4vw;padding:10px 10px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">Salient Object Detection</td>
  </tr>
  <tr>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/u2net/README.md">U2Net</a> (PR'2020)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">DIS</a> (ECCV'2022)</td>
  </tr>

</table>


## Usage

Install the required packages:

```
pip install -r utils/requirements.txt
```
<!-- pipreqs for get requirements.txt -->

Running private repository on Kaggle:
1. [Generate your token](https://github.com/settings/tokens)
2. Get repo address from `github.com/.../...git`: 
```bash
git clone https://your_personal_token@your_repo_address.git
cd CVP
```

### Using config files
Modify the configurations in `.yaml` config files, then run:

```bash
python scripts/train_dis.py [CONFIG] [RESUME] [DEVICE] [BATCH_SIZE] [EPOCHS]
```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

```bash
python scripts/train.py --resume path/to/the/ckpt
```

### Evaluating
```bash
python scripts/test.py
```
