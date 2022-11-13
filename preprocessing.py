from library import *


def tokenize_into_words(or_text):
    """"Tokenizer the text into words rather than Bert-Token"""
    words_list = []

    doc = nlp(or_text)
    for token in doc:
        words_list.append(token)
    return words_list


def preprocess_review(review):
    """Most of punctuation is kept when we do sentiment analysis"""
    review = review.replace("<br />", "")
    review = review.replace("...", "")
    review = review.replace("..", "")
    review = review.replace("-", "")
    return review


def compute_words(review):
    """When compute length of one review (perturb rate = replaced words / length), we delete all punctuation."""
    """Our perturb rate is more strict: punctuations are not include in dominator"""
    temp_string = re.sub(r"[^\w\s]", "", review)
    count_list = temp_string.split()
    return len(count_list)
