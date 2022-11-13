# **Generating Textual Adversaries with Minimal Perturbation**

This repository contains codes and resources associated to the paper: 

Xingyi zhao, Lu Zhang, Depeng Xu and Shuhan Yuan. 2022. **Generating Textual Adversaries with Minimal Perturbation**. In findings of the Association for Computational Linguistics: EMNLP 2022.

## Usage
You can find the fine tuned model from [textattack](https://huggingface.co/textattack).

To craft adversarial examples based on TAMPERS, run(attack "textattack/bert-base-uncased-rotten-tomatoes" for example):

```
python tampers.py --data_path data/MR.csv --victim_model "textattack/bert-base-uncased-rotten-tomatoes" --num 1000 --output_dir attack_result/
```

* --data_path: We take MR dataset as an example. To reproduce our experiment, dataset can be find [TAMPERS](https://drive.google.com/drive/folders/1ZCwZj39bwE2goUFr8_UiDkfoRg_NMO7Q). For more dataset, you can check [TextFooler](https://github.com/jind11/TextFooler). **Our code is based on binary classification task.**
