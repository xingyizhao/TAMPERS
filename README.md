# **Generating Textual Adversaries with Minimal Perturbation**

This repository contains codes and resources associated to the paper: 

Xingyi zhao, Lu Zhang, Depeng Xu and Shuhan Yuan. 2022. **Generating Textual Adversaries with Minimal Perturbation**. In findings of the Association for Computational Linguistics: EMNLP 2022.

## Dependencies
* Python 3.8
* PyTorch 1.11.0
* transformers 4.20.1
* cuda version 11.5

## Usage
To craft adversarial examples based on TAMPERS, run(attack "textattack/bert-base-uncased-rotten-tomatoes" for example):

```
python tampers.py --data_path data/MR.csv --victim_model "textattack/bert-base-uncased-rotten-tomatoes" --num 1000 --output_dir attack_result/
```

* --data_path: We take MR dataset for example. To reproduce our experiments, datasets can be find [TAMPERS](https://drive.google.com/drive/folders/1ZCwZj39bwE2goUFr8_UiDkfoRg_NMO7Q). For more datasets, you can check [TextFooler](https://github.com/jind11/TextFooler). **Our code is based on the binary classification task.**
* --victim_model: You can find the fine tuned models from [huggingface-textattack](https://huggingface.co/textattack). In our experiments, we use four fine tuned models corresponding to their datasets. [IMDB](https://huggingface.co/textattack/bert-base-uncased-imdb?text=I+like+you.+I+love+you), [MR](https://huggingface.co/textattack/bert-base-uncased-rotten-tomatoes?text=I+like+you.+I+love+you), [YELP](https://huggingface.co/textattack/bert-base-uncased-yelp-polarity?text=I+like+you.+I+love+you) and [SST2](https://huggingface.co/textattack/bert-base-uncased-SST-2?text=I+like+you.+I+love+you).   
* --num: Number of texts you want to attack.
* --output_dir: Output file. You need to create an empty file at first. 

## Baselines
To run the baselines, you can refer to [TextAttack](https://github.com/QData/TextAttack).

Two issues should be claimed here: 

1. Running bert attack will take long time in this package. See the issue [here](https://github.com/QData/TextAttack/issues/586). Therefore, we just follow the setting of
[TextDefender](https://github.com/RockyLzy/TextDefender/blob/master/textattack/transformations/word_swap_masked_lm.py) and ignore word to replace tokenized as multiple sub-words.

2. Using USE to compute the semantic similarity, we correct the code. In the TextFooler and bert-attack code, they forget to divide the angle between the two embedding by pi. The correct computation should be: **1 - arccos(cosine_similarity(u, v)) / pi**. See [here](https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance). 

## Demo of results:
We give a result for example(results are saved in attack_result), results below are based on MR dataset. In our paper, we sampled **five different 1000 samples and take an average value as the final results**.
![image](https://user-images.githubusercontent.com/90595479/201507041-68df97c5-edb4-4626-a94d-bd3f92d38d47.png)

## Citation:
