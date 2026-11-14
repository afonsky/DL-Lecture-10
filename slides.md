---
theme: seriph
addons:
  - "@twitwi/slidev-addon-ultracharger"
addonsConfig:
  ultracharger:
    inlineSvg:
      markersWorkaround: false
    disable:
      - metaFooter
      - tocFooter
background: /logo/mountain.jpg
highlighter: shiki
routerMode: hash
lineNumbers: false
duration: 80min

css: unocss
title: Deep Learning
subtitle: Computer Vision II
date: 16/11/2026
venue: HSE
author: Alexey Boldyrev
---

# <span style="font-size:28.0pt" v-html="$slidev.configs.title?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:32.0pt" v-html="$slidev.configs.subtitle?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:18.0pt" v-html="$slidev.configs.author?.replaceAll(' ', '<br/>')"></span>

<span style="font-size:18.0pt" v-html="$slidev.configs.date?.replaceAll(' ', '<br/>')"></span>

<div class="abs-tl mx-5 my-10">
  <img src="/logo/FCS_logo_full_L.svg" class="h-18">
</div>

<div class="abs-tr mx-5 my-5">
  <img src="/logo/DSBA_logo.png" class="h-28">
</div>

<style>
  :deep(footer) { padding-bottom: 3em !important; }
</style>


---
src: ./slides/0_introduction.md
---

---
src: ./slides/0_outline.md
---

---
src: ./slides/0b_from_models_to_systems.md
---

---
src: ./slides/1_ssd.md
---

---
src: ./slides/1b_ssd_anchor_counts.md
---

---
src: ./slides/2_yolo.md
---

---
src: ./slides/3_detection_comparison.md
---

---
src: ./slides/3b_detection_metrics.md
---

---
src: ./slides/4_semantic_segmentation.md
---

---
src: ./slides/4b_segmentation_flavors_visual.md
---

---
src: ./slides/5_transposed_conv.md
---

---
src: ./slides/5b_transposed_conv_practice.md
---

---
src: ./slides/6_fcn.md
---

---
src: ./slides/7_unet_modern_seg.md
---

---
src: ./slides/6b_segmentation_practice.md
---

---
src: ./slides/8_transfer_learning.md
---

---
src: ./slides/8b_finetune_modern_api.md
---

---
src: ./slides/9_augmentation.md
---

---
src: ./slides/9b_augmentation_lab.md
---

---
src: ./slides/10_neural_style.md
---

---
src: ./slides/11_practical_tips.md
---

---
src: ./slides/11b_sota_2026.md
---

---
src: ./slides/11c_model_choice_experts_resources.md
---

---
src: ./slides/0_end.md
---

---
src: ./slides/loss_functions_for_semantic_segmentation.md
---