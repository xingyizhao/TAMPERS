import spacy
import nltk
import random
import pandas as pd
from nltk.corpus import wordnet as wn
import OpenHowNet
from tqdm.auto import tqdm
import gensim.downloader as api
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import re

# load the corpus and necessary pre_trained model
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

# Victim model
fine_tune_model_name = "textattack/bert-base-uncased-imdb"
fine_tune_model = AutoModelForSequenceClassification.from_pretrained(fine_tune_model_name)
fine_tune_model_tokenizer = AutoTokenizer.from_pretrained(fine_tune_model_name)
sentiment_analysis_classifier = pipeline("sentiment-analysis", model=fine_tune_model,
                                         tokenizer=fine_tune_model_tokenizer, device=1)

'''Load Dataset'''


def imdb_dataloader(data_path):
    file = pd.read_csv(data_path, low_memory=False)
    data, label = list(file["review"]), list(file["sentiment"])

    for index in range(0, len(data), 1):
        yield data[index], label[index]


class ReviewDataset(Dataset):

    def __init__(self, review_data):
        self.review_data = review_data

    def __len__(self):
        return len(self.review_data)

    def __getitem__(self, index):
        return self.review_data[index]


def tokenize_into_words(reviews):
    words_list = []

    doc = nlp(reviews)
    for token in doc:
        words_list.append(token)

    return words_list


'''Do data preprocess'''


def preprocess_review(reviews):
    reviews = reviews.replace("<br />", "")
    reviews = reviews.replace("...", "")
    reviews = reviews.replace("..", "")
    reviews = reviews.replace("-", "")
    return reviews


# Compute length of review
def compute_words(reviews):
    temp_string = re.sub(r"[^\w\s]", "", reviews)
    count_list = temp_string.split()
    return len(count_list)


'''Get Candidate from wordnet and Hownet'''
# define the similarity between two words
model_vector = api.load("glove-wiki-gigaword-50")


def similarity_compute(language_model, word_1, word_2):
    similarity_score = language_model.similarity(word_1, word_2)
    return similarity_score


def load_stopwords(stop_words_path):
    with open(stop_words_path) as file:
        buffer = file.readlines()
        stop_words = [item.strip() for item in buffer]
    return stop_words


# Only consider: Nouns, Verbs, Adjectives and Adverbs in finding synonym in wordnet and hownet
def find_word_wordnet(token, stop_words_list):
    synonym_set = set()
    if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "ADV":
        if token.text not in stop_words_list:
            synset = wn.synsets(token.text)

            for item in synset:
                for sys in item.lemmas():
                    # pos_tagging = tokenize_into_words(sys.name())[0].pos_  # get the pos tag of each replacement
                    if "_" not in sys.name() and "-" not in sys.name():  # get rid of phrase
                        synonym_set.add(sys.name())
            synonym_set.add(token.text)

    return list(synonym_set)


# Find the similar word[have similar sememe] --- > hownet
def find_word_hownet(token, stop_words_list):
    synonym_set = set()
    if token.pos_ == "VERB" or token.pos_ == "NOUN" or token.pos_ == "ADJ" or token.pos_ == "ADV":
        if token.text not in stop_words_list:
            sememe_list = hownet_dict.get_sense(token.text)
            temp = []

            if len(sememe_list) > 0:
                for i in range(len(sememe_list)):
                    sense = sememe_list[i]
                    same_sense_word = hownet_dict.get_sense_synonyms(sense)

                    for en_words in same_sense_word:
                        # pos_tagging = tokenize_into_words(en_words.en_word)[0].pos_  # get the pos tag of each replacement
                        if "_" not in en_words.en_word and "-" not in en_words.en_word and " " not in en_words.en_word:
                            synonym_set.add(en_words.en_word)

            synonym_list = list(filter(None, list(synonym_set)))

            if len(synonym_list) < 10:
                return synonym_list
            else:  # get top-K most similar sememe in the synonym_list
                for word in synonym_list:
                    try:
                        temp.append((word, similarity_compute(model_vector, token.text, word)))
                    except KeyError:  # if the word is not in this model, go on
                        pass
                temp.sort(key=lambda term: term[1], reverse=True)
                synonym_list = [item[0] for item in temp[0:50]]  # reurn at most 50 words to get the candidate

            return synonym_list


# Combine two candidates[wordnet and hownet] into one list
def combine_candidate(token, stop_words_list):
    substitution_candidates = []
    if find_word_hownet(token, stop_words_list) is None or len(find_word_wordnet(token, stop_words_list)) == 0:
        return list()
    elif len(find_word_wordnet(token, stop_words_list)) > 0 and len(find_word_hownet(token, stop_words_list)) >= 0:
        synonym_set = set(find_word_wordnet(token, stop_words_list) + find_word_hownet(token, stop_words_list))
        # filter the synonym_set by pos checker[allow noun and verb swap]
        temp_candidates = list(synonym_set)
        for substitution in temp_candidates:
            pos_tagging = tokenize_into_words(substitution)[0].pos_
            if token.pos_ == "NOUN" or token.pos_ == "VERB":
                substitution_candidates.append(substitution)
            elif token.pos_ == pos_tagging:
                substitution_candidates.append(substitution)

        return substitution_candidates


'''Get Impact Score and Do Greedy Search '''


# get the result of original review
def get_result_org(original_string, feature):
    result = sentiment_analysis_classifier(original_string, truncation="only_first")
    feature.query_times += 1
    return result[0]["label"], result[0]["score"]


# get impact_score of each content word:original_tokens -- > spacy, original_lable --> str, original_score --> 0-1 float
def get_impact_scores(original_tokens, original_label, original_score, batch_size, stop_words_list):
    token_impact_score = []
    content_word_candidates = []  # save the replacement of content word

    for token in original_tokens:  # original_tokens is a list that tokenizing from the whole data
        token_impact_score.append((token, 0))  # initialize the impact score of each token

    for index, token in enumerate(token_impact_score):
        replacement = combine_candidate(token[0], stop_words_list)
        if len(replacement) == 0:
            continue
        else:
            review_list = []
            # get all the replacement review
            for word in replacement:
                temp_token_impact_score = token_impact_score.copy()
                temp_token_impact_score[index] = (tokenize_into_words(word + " ")[0], 0)
                review_list.append("".join([token[0].text_with_ws for token in temp_token_impact_score]))
            # # compute the impact score of content word
            potential_adv_review = ReviewDataset(review_list)

            score_list = []
            for i, result in enumerate(sentiment_analysis_classifier(potential_adv_review, batch_size=batch_size,
                                                                     truncation="only_first")):
                if result["label"] == original_label:
                    score_list.append(original_score - result["score"])
                else:
                    score_list.append(original_score - (1 - result["score"]))
            impact_score = max(score_list)
            token_impact_score[index] = (token[0], impact_score)  # keep the maximum changing score as impact score
            content_word_candidates.append((index, token[0], impact_score, replacement))

    return token_impact_score, content_word_candidates


class Feature(object):
    def __init__(self):
        self.query_times = 0
        self.perturbation = 0
        self.success_num = 0


def greedy_search_word_substitution(feature, impact_score_list, content_words_candidates, original_label,
                                    original_score,
                                    batch_size):
    content_words_candidates.sort(key=lambda item: item[2], reverse=True)
    temp_impact_score_list = [item[0] for item in impact_score_list]
    activating_points = []  # record the perturbation points in the sentence
    for item in content_words_candidates:
        review_list = []
        max_score = 0
        optimal_index = 0
        feature.perturbation += 1
        activating_points.append(item[0])

        for word in item[3]:
            temp_impact_score_list[item[0]] = tokenize_into_words(word + " ")[0]
            review_list.append("".join([token.text_with_ws for token in temp_impact_score_list]))
        potential_adv_review = ReviewDataset(review_list)

        for index, result in enumerate(
                sentiment_analysis_classifier(potential_adv_review, batch_size=batch_size, truncation="only_first")):
            feature.query_times += 1
            if result["label"] != original_label:
                return activating_points, review_list[index], True
            else:
                if original_score - result["score"] > max_score:
                    max_score = original_score - result["score"]
                    optimal_index = index

        temp_impact_score_list[item[0]] = tokenize_into_words(item[3][optimal_index] + " ")[0]  # keep the best answer
    return None, None, False


# Doing further optimization by genetic algorithm
def attack_main():
    success_num = 0  # record the success number
    num_count = 0  # control the number we need to attack
    attack_count = 0  # only attack the text that is classified correctly
    skip_count = 0  # skip the examples that are classified incorrectly
    perturbation_nums = 0  # record the overall perturbation numbers
    length_sum = 0

    # global content_words_candidates, review, original_label

    ad_example_df = pd.DataFrame(
        columns=["original review", "original sentiment", "adversarial example", "current sentiment"])
    bar = tqdm(imdb_dataloader("IMDB Dataset.csv"))
    stop_words_list = load_stopwords("stopwords")
    imdb_attack_feature = Feature()

    for review, sentiment in bar:
        review = preprocess_review(review)

        original_label, original_score = get_result_org(review, imdb_attack_feature)
        sentiment = ("LABEL_1" if sentiment == "positive" else "LABEL_0")

        if original_label == sentiment:

            attack_count += 1
            batch_size = 10
            length_sum += compute_words(review)

            impact_score_list, content_words_candidates = get_impact_scores(tokenize_into_words(review),
                                                                            original_label, original_score,
                                                                            batch_size, stop_words_list)
            activating_points, adversarial_text, success_flag = greedy_search_word_substitution(imdb_attack_feature,
                                                                                                impact_score_list,
                                                                                                content_words_candidates,
                                                                                                original_label,
                                                                                                original_score,
                                                                                                batch_size)

            if success_flag is True:
                success_num += 1
                #if len(activating_points) / compute_words(review) >= (
                      #  0 / 100):  # we need to do genetic algorithm to optimize the adversarial with 8% and more
                if len(activating_points) > 1:
                    # genetic part -- activating points reduce based on impact score
                    class Individual(object):
                        def __init__(self, ontology, variation_points):
                            self.ontology = ontology
                            self.variation_points = variation_points
                            self.perturbation = self.perturbation_num()
                            self.fitness_score = self.cal_fitness()

                        @staticmethod
                        def tokenized_words(review_x):
                            words_list = []
                            global nlp
                            doc = nlp(review_x)
                            for token in doc:
                                words_list.append(token)
                            return words_list

                        @staticmethod
                        def mutate_point(variation):
                            '''create a mutated replacement'''
                            for item in content_words_candidates:
                                if item[0] == variation:
                                    return Individual.tokenized_words(random.choice(item[3]) + " ")[0]

                        def create_population(self):
                            '''randomly generate a new sentence based on the varaiation point'''
                            ontology = Individual.tokenized_words(
                                self.ontology)  # get the tokenized list of the original part: get the position of mutation points
                            individual = ontology.copy()
                            for point in self.variation_points:
                                candidate = list()
                                for item in content_words_candidates:
                                    if item[0] == point:
                                        candidate = item[3]
                                individual[point] = Individual.tokenized_words(random.choice(candidate) + " ")[
                                    0]  # In each iteration, initialize a variation point
                            return "".join([word.text_with_ws for word in individual])

                        def mate(self, parent_2):
                            '''performing mating and produce new data based on variation point'''
                            child_data = list()

                            for index, (gw1, gw2) in enumerate(zip(Individual.tokenized_words(self.ontology),
                                                                   Individual.tokenized_words(parent_2.ontology))):

                                if index not in self.variation_points:
                                    child_data.append(gw1)
                                else:
                                    # random probability
                                    prob = random.random()
                                    # if the probability is less than 0.45, get the word from parent 1
                                    if prob < 0.40:
                                        child_data.append(gw1)
                                    # if the probability is less than 0.90 and more than 0.45, get the word from parent 2
                                    elif prob < 0.80:
                                        child_data.append(gw2)
                                    # otherwise, choose an option from the original candidate
                                    else:
                                        child_data.append(Individual.mutate_point(index))
                            child = "".join([word.text_with_ws for word in child_data])
                            return Individual(child, self.variation_points)

                        def perturbation_num(self):
                            perb_num = 0
                            x_orig_token = Individual.tokenized_words(review)  # tokenized list of the original review
                            x_cur_token = Individual.tokenized_words(self.ontology)
                            for token_orig, token_cur in zip(x_orig_token, x_cur_token):
                                if token_orig.text_with_ws != token_cur.text_with_ws:
                                    perb_num += 1  # record the perturbation number
                            return perb_num

                        '''singular objective function'''

                        def cal_fitness(self):
                            global sentiment_analysis_classifier
                            result = sentiment_analysis_classifier(self.ontology, truncation="only_first")
                            if original_label != result[0]["label"]:
                                fitness_score = 1
                            else:
                                fitness_score = 1 - result[0]["score"]
                            return fitness_score

                    # do activating points reduce --- reduce points from tail to the head
                    temp_adversarial_example = adversarial_text
                    temp_perturbation = len(activating_points)
                    activating_length = len(activating_points)

                    for tabu in range(activating_length):

                        population_size = 10
                        generation = 1
                        max_iteration = 100
                        population = []
                        flag = False  # sign whether find new adversarial example in the new generation

                        instance = Individual(review, activating_points)
                        # generate the first generation
                        for _ in range(population_size):
                            population.append(Individual(instance.create_population(), activating_points))

                        for _ in range(max_iteration):
                            population.sort(key=lambda x: x.fitness_score, reverse=True)
                            print("Generation: {}\tString: {}\tFitness: {}".format(generation, population[0].ontology,
                                                                                   population[0].fitness_score))

                            if population[0].fitness_score == 1:
                                flag = True
                                break

                            new_generation = []

                            s = int((20 * population_size) / 100)
                            new_generation.extend(population[:s])
                            s = int((80 * population_size) / 100)
                            for _ in range(s):
                                parent_1 = random.choice(population[:5])
                                parent_2 = random.choice(population[:5])
                                child = parent_1.mate(parent_2)
                                new_generation.append(child)

                            population = new_generation
                            generation += 1

                        if flag is True:
                            # save current adversarial_example and number of perturbation
                            temp_adversarial_example = population[0].ontology
                            temp_perturbation = len(activating_points)
                        else:
                            current_label = ("LABEL_0" if original_label == "LABEL_1" else "LABEL_1")
                            ad_example_df.loc[len(ad_example_df)] = [review, original_label, temp_adversarial_example,
                                                                     current_label]
                            perturbation_nums += temp_perturbation
                            break

                        activating_points.pop()  # remove the smallest impact score in the list
                else:
                    current_label = ("LABEL_0" if original_label == "LABEL_1" else "LABEL_1")
                    ad_example_df.loc[len(ad_example_df)] = [review, original_label, adversarial_text, current_label]
                    perturbation_nums += len(activating_points)

        else:
            skip_count += 1

        num_count += 1
        if num_count == 50:
            break
    ad_example_df.to_csv("adversarial_text_50.csv")
    avg_len = length_sum / attack_count
    print()
    print("**********************************Attack Information********************************")
    print("The average length of data is {}".format(avg_len))
    print("Success number is {}".format(success_num))
    print("Skip number is {}".format(skip_count))
    print("The success rate is {}".format(success_num / attack_count))
    print("Only Greedy: The average perturbation are {}".format(imdb_attack_feature.perturbation / attack_count))
    print("Only Greedy: The average perturbation rate are {}".format(
        imdb_attack_feature.perturbation / attack_count / avg_len))
    print("Greedy + APR-GA: The average perturbation are {}".format(perturbation_nums / attack_count))
    print("Greedy + APR-GA: The average perturbation rate are {}".format(perturbation_nums / attack_count / avg_len))


if __name__ == '__main__':
    attack_main()
