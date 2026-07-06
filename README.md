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


<div align="center">
<img src="assets/OccFree-VTON-video.gif" alt="video" width="100%">
</div>


## 📊 Visualization Results
<div align="center">
  <img src="static/images/appDress_01.png" alt="video" width="100%">
</div>
<div align="center">
  <img src="static/images/appUpper_01.png" alt="video" width="100%">
</div>
<div align="center">
  <img src="static/images/appLower_01.png" alt="video" width="100%">
</div>

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

## 📜 License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

---

<p align="center"> 
<img src="https://api.star-history.com/svg?repos=du-chenghu/OccFree-VTON&type=Date" style="width:70%"/>
</p>


