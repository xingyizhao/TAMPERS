# **Generating Textual Adversaries with Minimal Perturbation**

This repository contains codes and resources associated to the paper: 

Xingyi zhao, Lu Zhang, Depeng Xu and Shuhan Yuan. 2022. **Generating Textual Adversaries with Minimal Perturbation**. In findings of the Association for Computational Linguistics: EMNLP 2022.

## Usage
To craft adversarial examples based on TAMPERS, run(attack "textattack/bert-base-uncased-rotten-tomatoes" for example):

```
python tampers.py --data_path data/MR.csv --victim_model "textattack/bert-base-uncased-rotten-tomatoes" --num 1000 --output_dir attack_result/
```

* --data_path: We take MR dataset for example. To reproduce our experiment, dataset can be find [TAMPERS](https://drive.google.com/drive/folders/1ZCwZj39bwE2goUFr8_UiDkfoRg_NMO7Q). For more dataset, you can check [TextFooler](https://github.com/jind11/TextFooler). **Our code is based on binary classification task.**
* --victim_model: You can find the fine tuned model from [textattack](https://huggingface.co/textattack). In our experiment, we use four fine tuned models corresponding to their dataset. [IMDB](https://huggingface.co/textattack/bert-base-uncased-imdb?text=I+like+you.+I+love+you), [MR](https://huggingface.co/textattack/bert-base-uncased-rotten-tomatoes?text=I+like+you.+I+love+you), [YELP](https://huggingface.co/textattack/bert-base-uncased-yelp-polarity?text=I+like+you.+I+love+you) and [SST2](https://huggingface.co/textattack/bert-base-uncased-SST-2?text=I+like+you.+I+love+you).   
