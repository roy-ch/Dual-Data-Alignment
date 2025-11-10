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
## Results of a single DDA model on 11 benchmarks
JPEG compression with a quality factor of 96 is applied to the synthetic images in GenImage, ForenSynths, and AIGCDetectionBenchmark to mitigate format bias. The number of generators used in each dataset is reported: G refers to GAN, D to Diffusion, and AR to Auto-Regressive models. Among the 11 benchmarks, Chameleon, Synthwildx, WildRF, and Bfree-Online are the 4 in-the-wild datasets. Notably, DDA is **the first detector** to achieve over 80% cross-data accuracy on Chameleon.

| Benchmark | NPR (CVPR'24) | UnivFD (CVPR'23) | FatFormer (CVPR'24) | SAFE (KDD'25) | C2P-CLIP (AAAI'25) | AIDE (ICLR'25) | DRCT (ICML'24) | AlignedForensics (ICLR'25) | DDA (ours) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GenImage (1G + 7D) | 51.5 ± 6.3 | 64.1 ± 10.8 | 62.8 ± 10.4 | 50.3 ± 1.2 | 74.4 ± 8.4 | 61.2 ± 11.9 | 84.7 ± 2.7 | 79.0 ± 22.7 | **91.7 ± 7.8** |
| DRCT-2M (16D) | 37.3 ± 15.0 | 61.8 ± 8.9 | 52.2 ± 5.7 | 59.3 ± 19.2 | 59.2 ± 9.9 | 64.6 ± 11.8 | 90.5 ± 7.4 | 95.5 ± 6.1 | **98.1 ± 1.4** |
| DDA-COCO (5D) | 42.2 ± 5.4 | 52.4 ± 1.5 | 51.7 ± 1.5 | 49.9 ± 0.3 | 51.3 ± 0.6 | 50.0 ± 0.4 | 60.2 ± 4.3 | 86.5 ± 19.1 | **92.2 ± 10.6** |
| EvalGEN (3D + 2AR) | 2.9 ± 2.7 | 15.4 ± 14.2 | 45.6 ± 33.1 | 1.1 ± 0.6 | 38.9 ± 31.2 | 19.1 ± 11.1 | 77.8 ± 5.4 | 68.0 ± 20.7 | **97.2 ± 4.2** |
| Synthbuster (9D) | 50.0 ± 2.6 | 67.8 ± 14.4 | 56.1 ± 10.7 | 46.5 ± 20.8 | 68.5 ± 11.4 | 53.9 ± 18.6 | 84.8 ± 3.6 | 77.4 ± 25.0 | **90.1 ± 5.6** |
| ForenSynths (11G) | 47.9 ± 22.6 | 77.7 ± 16.1 | 90.0 ± 11.8 | 49.7 ± 2.7 | **92.0 ± 10.1** | 59.4 ± 24.6 | 73.9 ± 13.4 | 53.9 ± 7.1 | 81.4 ± 13.9 |
| AIGCDetectionBenchmark (7G + 10D) | 53.1 ± 12.2 | 72.5 ± 17.3 | 85.0 ± 14.9 | 50.3 ± 1.1 | 81.4 ± 15.6 | 63.6 ± 13.9 | 81.4 ± 12.2 | 66.6 ± 21.6 | **87.8 ± 12.6** |
| Chameleon (Unknown) | 59.9 | 50.7 | 51.2 | 59.2 | 51.1 | 63.1 | 56.6 | 71.0 | **82.4** |
| Synthwildx (3D) | 49.8 ± 10.0 | 52.3 ± 11.3 | 52.1 ± 8.2 | 49.1 ± 0.7 | 57.1 ± 4.2 | 48.8 ± 0.8 | 55.1 ± 1.8 | 78.8 ± 17.8 | **90.9 ± 3.1** |
| WildRF (Unknown) | 63.5 ± 13.6 | 55.3 ± 5.7 | 58.9 ± 8.0 | 57.2 ± 18.5 | 59.6 ± 7.7 | 58.4 ± 12.9 | 50.6 ± 3.5 | 80.1 ± 10.3 | **90.3 ± 3.5** |
| Bfree-Online (Unknown) | 49.5 | 49.0 | 50.0 | 50.5 | 50.0 | 53.1 | 55.7 | 68.5 | **95.1** |
| **Avg ACC** | 46.1 ± 16.1 | 56.3 ± 16.5 | 59.6 ± 14.6 | 47.6 ± 16.0 | 62.1 ± 15.6 | 54.1 ± 12.8 | 70.1 ± 14.6 | 75.0 ± 11.1 | **90.7 ± 5.3** |
| **Min ACC** | 2.9 | 15.4 | 45.6 | 1.1 | 38.9 | 19.1 | 50.6 | 53.9 | **81.4** |

---

## Checkpoints

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
@inproceedings{chen2025dual,
  title={Dual Data Alignment Makes {AI}-Generated Image Detector Easier Generalizable},
  author={Ruoxin Chen and Junwei Xi and Zhiyuan Yan and Ke-Yue Zhang and Shuang Wu and Jingyi Xie and Xu Chen and Lei Xu and Isabel Guan and Taiping Yao and Shouhong Ding},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=C39ShJwtD5}
}
```
