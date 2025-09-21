<!-- README.md -->
<div align="center">

<!-- 标题：渐变透明 PNG 替代文字渐变（GitHub 不支持 background-clip:text） -->
<h1>Mitigating Occlusions in Virtual Try-On via A Simple-Yet-Effective Mask-Free Framework</h1>
<!-- 作者 -->
<p>
  <a href="https://github.com/du-chenghu">Chenghu Du</a><sup>1</sup>&nbsp;·&nbsp;
  <a href="https://github.com/">Shengwu Xiong</a><sup>2,3</sup>&nbsp;·&nbsp;
  <a href="https://github.com/">Junyin Wang</a><sup>1</sup>&nbsp;·&nbsp;
  <a href="https://github.com//">Yi Rong</a><sup>1✉</sup>&nbsp;·&nbsp;
  <a href="https://github.com//">Shili Xiong</a><sup>1✉</sup>
</p>

<!-- 单位 -->
<sup>1</sup> School of Computer Science and Artificial Intelligence, Wuhan University of Technology<br>
<sup>2</sup> Interdisciplinary Artificial Intelligence Research Institute, Wuhan College<br>
<sup>3</sup> Shanghai Artificial Intelligence Laboratory<br>
<sup>✉</sup> Corresponding authors

<!-- 徽章 -->
<a href="https://arxiv.org/abs/2507.01634"><img src='https://img.shields.io/badge/Paper-2507.01634-red' alt='Paper PDF'></a>
<a href="https://ghost233lism.github.io/OccFree-VTON-page/ "><img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
<a href='https://huggingface.co/ghost233lism/OccFree-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/spaces/ghost233lism/OccFree-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange' alt='Demo'></a>
<a href='https://huggingface.co/papers/2507.01634'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow'></a>

<!-- 语言切换 -->
<p>
  <strong>English</strong> | <a href="README_zh.md">简体中文</a>
</p>

</div>


## 📄 Abstract

This work tackles occlusion issues in Virtual Try-On (VTON).  
We taxonomize failures into:

1. **Inherent Occlusions** – “ghost” garments from the reference image that remain in the result.  
2. **Acquired Occlusions** – distorted human anatomy that visually blocks the new outfit.

To remove both, we propose a **mask-free VTON framework** with two plug-and-play operations:

- **Background Pre-Replacement** – swaps the background before generation so the model never confuses clothes with body/background, suppressing inherent occlusions.  
- **Covering-and-Eliminating** – enforces human-aware semantics, yielding anatomically plausible shapes and thus fewer acquired occlusions.

The operations are architecture-agnostic: drop them into GANs or diffusion models without re-design.  

![teaser](assets/teaser.png)

<div align="center">
<img src="assets/OccFree-VTON-video.gif" alt="video" width="100%">
</div>

## 🆕 News

*All dates are UTC.*

- **2025-09-20** 🚀 Project page & teaser image live.
- **2025-09-19** 🔥 Paper accepted at NeurIPS 2025.  

## 🚧 TODO List

**The `test code` has been released, the `training code` will be released soon.**
- [x] [2025-00-00] Release the **training script**.
- [x] [2025-00-00] Release the **pretrained model**.
- [x] [2025-09-21] Release the **testing script**.
- [x] [2025-09-21] Release the **manuscript**.

## 🏗 Model Architecture

![architecture](assets/architecture.png)

### 🔧 Installation
```
pip3 install -r requirements.txt
```

**or**

`conda create -n uscpfn python=3.6`

`source activate uscpfn     or     conda activate uscpfn`

`conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=11.7 -c pytorch`

`conda install cupy     or     pip install cupy==8.3.0`

`pip install opencv-python`

`git clone https://github.com/du-chenghu/OccFree-VTON.git`

`cd ./OccFree-VTON/`

### 📋 Requirements

- Python>=3.9
- torch==2.3.0
- torchvision==0.18.0
- torchaudio==2.3.0
- cuda==12.1

### ⚙️ Setup

```bash
git clone https://github.com/HVision-NKU/DepthAnythingAC.git
cd DepthAnythingAC
conda create -n depth_anything_ac python=3.9
conda activate depth_anything_ac
pip install -r requirements.txt
```

## 📦 Dataset

We train and evaluate on two standard VTON datasets:

| Dataset | Images | Resolution | Download | Annotation |
|---------|--------|------------|----------|------------|
| VITON-HD | ~13k | 1024×768 | [Google Drive]([https://drive.google.com/xxx](https://github.com/shadow2496/VITON-HD)) | keypoints, parse, cloth |
| DressCode | ~52k | 512×384 | [Official](https://dress-code.s3.eu-central-1.amazonaws.com) | keypoints, parse, cloth |

## 🚀 Usage

### Get the Model
Download the pre-trained checkpoints from [Hugging Face](https://huggingface.co/ghost233lism/OccFree-VTON):
```bash
mkdir checkpoints
cd checkpoints

# (Optional) Using huggingface mirrors
export HF_ENDPOINT=https://hf-mirror.com

# download OccFree-VTON model from huggingface
huggingface-cli download --resume-download ghost233lism/OccFree-VTON --local-dir ghost233lism/OccFree-VTON
```

We also provide the OccFree-VTON model on Google Drive: [Download](https://drive.google.com/drive/folders/1yjM7_V9XQlL-taoRTbMq7aoCh1-Xr-ya?usp=sharing)


### Inference
1. cd OccFree-VTON
```
cd OccFree-VTON-main
```
2. First, you need to download the [[Checkpoints for Test]](https://drive.google.com) and put these under the folder `checkpoints/`. The folder `checkpoints/` shold contain `ngd_model_final.pth` and `sig_model_final.pth`. 
3. Please put the test set of the dataset in the `dataset/`, i.e., the `dataset/` folder should contain the `test_pairs.txt` and `test/`.
4. To generate virtual try-on images, just run:
```
python test.py
```
5. The results will be saved in the folder `results/`.
> During inference, only a person image and a clothes image are fed into the network to generate the try-on image. **No human parsing results or human pose estimation results are needed for inference.**

> **To reproduce our results from the saved model, your test environment should be the same as our test environment.**

- Note that if you want to test **paired data (paired clothing-person images)**, please download the replaceable list <a href="https://drive.google.com/file/d/1yYvBoTUqwQjllS9XELoDQFuqZXdgqzL3/view?usp=sharing">here (test_pairs.txt)</a>.



## 📊 Visualization Results


## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{du2025mitigating,
  title={Mitigating Occlusions in Virtual Try-On via A Simple-Yet-Effective Mask-Free Framework},
  author={Du, Chenghu and Xiong, Shengwu and Wang, Junyin and Rong, Yi and Xiong, Shili},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 📬 Contact

For technical questions or commercial licensing, please contact 
duch@whut.edu.cn

## 🤝 Acknowledgements
Our code is based on the official implementation of [[CatVTON](https://github.com/Zheng-Chong/CatVTON)]. We thank the authors of [CatVTON] for the foundational work. We also thank the authors of VITON-HD and DressCode datasets for their excellent benchmarks, and the open-source communities of PyTorch, HuggingFace Diffusers and xformers.

## 📜 License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

---

<p align="center"> 
<img src="https://api.star-history.com/svg?repos=du-chenghu/OccFree-VTON&type=Date" style="width:70%"/>
</p>


