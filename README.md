<div align="center">
<h1> Dual Data Alignment (NeurIPS 2025 Spotlight)</h1>
<h3>Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable</h3>

Ruoxin Chen<sup>1</sup>, Junwei Xi<sup>2</sup>, Zhiyuan Yan<sup>3</sup>, Keyue Zhang<sup>1</sup>, Shuang Wu<sup>1</sup>,  
Jingyi Xie<sup>4</sup>, Xu Chen<sup>2</sup>, Lei Xu<sup>5</sup>, Isabel Guan<sup>6†</sup>, Taiping Yao<sup>1†</sup>, Shouhong Ding<sup>1</sup>


<sup>1</sup>Tencent YouTu Lab
<sup>2</sup>East China University of Science and Technology
<sup>3</sup>Peking University

<sup>4</sup>Renmin University of China
<sup>5</sup>Shenzhen University
<sup>6</sup>Hong Kong University of Science and Technology


[[GitHub](https://github.com/roy-ch/Dual-Data-Alignment)] [[Paper](https://arxiv.org/abs/2505.14359)] [[Dataset (Coming Soon)]()]

</div>

## 📣 News

- `2025/09` : 🎉 Accepted by NeurIPS 2025 as **Spotlight**.
<!-- - `2025/08` : 🏆 DDA (Ke-Yue Zhang's team) wins **1st Prize** at the [The 6th Face Anti-Spoofing Workshop: Unified Physical-Digital Attacks Detection@ICCV2025]((https://sites.google.com/view/face-anti-spoofing-challenge/winners-results/challengeiccv2025)) ! Notably, 🔥 our winner model is exclusively trained on DDA-aligned COCO, without using any competition-provided face data. **A model trained with no face data wins a face anti-spoofing competition**.-->
- `2025/10` : 🏆 **[ICCV 2025 FAS Challenge: 1st Prize](https://sites.google.com/view/face-anti-spoofing-challenge/winners-results/challengeiccv2025) (Ke-Yue Zhang’s team)**
  * Winning model trained exclusively on **DDA-aligned COCO** (no face data). **A model with zero face data won a face anti-spoofing competition.**
---


## 💡 Abstract

> *The rapid increase in AI-generated images (AIGIs) underscores the need for detection methods. Existing detectors are often trained on biased datasets, leading to overfitting on spurious correlations between non-causal image attributes and real/synthetic labels. While these biased features enhance performance on the training data, they result in substantial performance degradation when tested on unbiased datasets. A common solution is to perform data alignment through generative reconstruction, matching the content between real and synthetic images. However, we find that pixel-level alignment alone is inadequate, as the reconstructed images still suffer from frequency-level misalignment, perpetuating spurious correlations. To illustrate, we observe that reconstruction models restore the high-frequency details lost in real images, inadvertently creating a frequency-level misalignment, where synthetic images appear to have richer high-frequency content than real ones. This misalignment leads to models associating high-frequency features with synthetic labels, further reinforcing biased cues. To resolve this, we propose Dual Data Alignment (DDA), which aligns both the pixel and frequency domains. DDA generates synthetic images that closely resemble real ones by fusing real and synthetic image pairs in both domains, enhancing the detector's ability to identify forgeries without relying on biased features. Moreover, we introduce two new test sets: DDA-COCO, containing DDA-aligned synthetic images, and EvalGEN, featuring the latest generative models. Our extensive evaluations demonstrate that a detector trained exclusively on DDA-aligned MSCOCO improves across diverse benchmarks.*


<!-- 两图一行：bias 左边，benchmark 右边 -->
<!-- <div style="display:flex; justify-content:space-between; align-items:center; margin:20px 0;">
    <img src="assets/bias.png" style="max-width:48%; height:auto;" />
    <img src="assets/BenchmarkComparison.png" style="max-width:48%; height:auto;" />
</div> -->

<!-- motivation 居中 -->
<div style="text-align:center; margin:20px 0;">
    <img src="assets/motivation.png" style="max-width:60%; height:auto;" />
</div>

---

## DDA checkpoint for paper

The checkpoint has been released on [modelscope](https://modelscope.cn/datasets/roych1997/Dual_Data_Alignment/files).


## 🎯 ToDo List <a name="todo"></a>

- [x] Release arxiv paper with complete BibTeX citation
- [x] Release checkpoint and inference code
- [ ] Release training set and training script
- [ ] Release code for DDA data construction
- [ ] Release model and code for ICCV 2025 FAS Challenge


## 📨 Contact

If you have any questions or suggestions, please feel free to contact us 
at [cusmochen@tencent.com](cusmochen@tencent.com) or add us on WeChat (ID: 18818203081).

## ✍️ Citing
If you find this repository useful for your work, please consider citing it as follows:
```
@inproceedings{chen2025dda,
  title={Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable},
  author={Chen, Ruoxin and Xi, Junwei and Yan, Zhiyuan and Zhang, Ke-Yue and Wu, Shuang and Xie, Jingyi and Chen, Xu and Xu, Lei and Guan, Isabel and Yao, Taiping and Ding, Shouhong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
