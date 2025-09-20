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


**[Abstract]** This paper investigates the occlusion problems in virtual try-on (VTON) tasks. According to how they affect the try-on results, the occlusion issues of existing VTON methods can be grouped into two categories: (1) Inherent Occlusions, which are the ghosts of the clothing from reference input images that exist in the try-on results. (2) Acquired Occlusions, where the spatial structures of the generated human body parts are disrupted and appear unreasonable. To this end, we analyze the causes of these two types of occlusions, and propose a novel mask-free VTON framework based on our analysis to  deal with these occlusions effectively. In this framework, we develop two simple-yet-powerful operations: (1) The background pre-replacement operation prevents the model from confusing the target clothing information with the human body or image background, thereby mitigating inherent occlusions. (2) The covering-and-eliminating operation enhances the model's ability of understanding and modeling human semantic structures, leading to more realistic human body generation and thus reducing acquired occlusions. Moreover, our method is highly generalizable, which can be applied in in-the-wild scenarios, and our proposed operations can also be easily integrated into different generative network architectures (e.g., GANs and diffusion models) in a plug-and-play manner. Extensive experiments on three VTON datasets validate the effectiveness and generalization ability of our method. Both qualitative and quantitative results demonstrate that our method outperforms recently proposed VTON benchmarks. 


![teaser](assets/teaser.png)

<div align="center">
<img src="assets/OccFree-VTON-video.gif" alt="video" width="100%">
</div>

## News

- **2025-09-21:** 🔥  The OccFree-VTON model and evaluation benchmarks are released

## TODO List
- **2025-07-03:** 🚀  OccFree-VTON was ranked **#3 Paper of the Day** on [HuggingFace Daily Papers](https://huggingface.co/papers/date/2025-07-03).
- **2025-07-03:** 🔥  The paper of [OccFree-VTON](https://arxiv.org/abs/2507.01634) is released. 
- **2025-07-02:** 🔥  The code of [OccFree-VTON](HVision-NKU/DepthAnythingAC) is released.
- [ ] Instructions for training dataset download and process.
- [ ] Jittor implementation of OccFree-VTON.
- [ ] Longer and more comprehensive video demo.
## Model Architecture

![architecture](assets/architecture.png)

## Installation

### Requirements

- Python>=3.9
- torch==2.3.0
- torchvision==0.18.0
- torchaudio==2.3.0
- cuda==12.1

### Setup

```bash
git clone https://github.com/HVision-NKU/DepthAnythingAC.git
cd DepthAnythingAC
conda create -n depth_anything_ac python=3.9
conda activate depth_anything_ac
pip install -r requirements.txt
```



## Usage
### Get Depth-Anything-AC Model
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


### Quick Inference

We provide the quick inference scripts for single/batch image input in `tools/`.  Please refer to [infer](./tools/README.md) for detailed information.

### Training
We provide the full training process of OccFree-VTON, including consistency regularization, spatial distance extraction/constraint and wide-used Affine-Invariant Loss Function.

Prepare your configuration in `configs/` file and run:

```bash
bash tools/train.sh <num_gpu> <port>
```

### Evaluation
We provide the direct evaluation for DA-2K, enhanced DA-2K, KITTI, NYU-D, Sintel, ETH3D, DIODE, NuScenes-Night, RobotCar-night, DS-rain/cloud/fog, KITTI-C benchmarks. You may refer to `configs/` for more details.

```bash
bash tools/val.sh <num_gpu> <port> <dataset>
```

## Results

### Quantitative Results

#### DA-2K Multi-Condition Robustness Results




## Citation

If you find this work useful, please consider citing:

```bibtex
@article{du2025mitigating,
  title={Mitigating Occlusions in Virtual Try-On via A Simple-Yet-Effective Mask-Free Framework},
  author={Du, Chenghu and Xiong, Shengwu and Wang, Junyin and Rong, Yi and Xiong, Shili},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```


## License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact 
sbysbysby123[AT]gmail.com or jin_modi[AT]mail.nankai.edu.cn

For commercial licensing, please contact andrewhoux[AT]gmail.com.

## Acknowledgements

We thank the authors of [DepthAnything](https://github.com/LiheYoung/Depth-Anything) and [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2) for their foundational work. We also acknowledge [DINOv2](https://github.com/facebookresearch/dinov2) for the robust visual encoder, [CorrMatch](https://github.com/BBBBchan/CorrMatch) for their codebase, and [RoboDepth](https://github.com/ldkong1205/RoboDepth) for their contributions.

<p align="center"> 
<img src="https://api.star-history.com/svg?repos=du-chenghu/OccFree-VTON&type=Date" style="width:70%"/>
</p>


