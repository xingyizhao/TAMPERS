import argparse
import re
import spacy
import math
import nltk
import random
import OpenHowNet
import pandas as pd
import gensim.downloader as api
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from nltk.corpus import wordnet as wn
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset

# load the corpus and necessary language resource
nlp = spacy.load("en_core_web_sm")
nltk.download("wordnet")
nltk.download("words")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('omw-1.4')
OpenHowNet.download()
hownet_dict = OpenHowNet.HowNetDict(init_sim=True)
hownet_dict.initialize_babelnet_dict()

