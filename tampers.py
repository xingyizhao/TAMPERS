from library import *
from config import Config
from preprocessing import tokenize_into_words, preprocess_review, compute_words
from dataset import ReviewDataset, dataloader


def similarity_compute(glove_embedding, word_1, word_2):
    """Compute similarity of two words"""
    similarity_score = glove_embedding.similarity(word_1, word_2)
    return similarity_score


def load_stopwords(stop_words_path):
    """We do not do any substitution in stopwords of a review"""
    with open(stop_words_path) as file:
        buffer = file.readlines()
        stop_words = [item.strip() for item in buffer]
    return stop_words


def find_word_wordnet(token, stop_words_list):
    """Find synonyms of a word in wordnet: All synonyms of a word are saved in synonym_set"""
    synonym_set = set()

    # Content Words: Nouns, Verbs, Adjectives and Adverbs in wordnet
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


def find_word_hownet(token, stop_words_list, glove_embedding):
    """Find similar word[similar words mean the words have the same or similar sememe] in hownet:
       All similar words of a word are saved in synonym_set
    """
    synonym_set = set()

    # Content Words: Nouns, Verbs, Adjectives and Adverbs in wordnet
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
                        temp.append((word, similarity_compute(glove_embedding, token.text, word)))
                    except KeyError:  # if the word is not in this model, go on
                        pass
                temp.sort(key=lambda term: term[1], reverse=True)
                synonym_list = [item[0] for item in temp[0:50]]

            return synonym_list  # return at most 50 words to get the candidate


def combine_candidate(token, stop_words_list, glove_embedding):
    """Combine two candidates[wordnet and hownet] into one list
       Here, we have two strategies getting our candidate.
       1. We only replace the word that must have sememe and synonyms at the same time
       2. We can replace the word if they have sememe or synonyms
       In our paper, we apply the first strategy, because we want the replaced word is meaningful enough. You can
       uncomment the code and apply the second strategy, you will get better results.
    """
    substitution_candidates = []

    """Candidate Strategy One"""
    if find_word_hownet(token, stop_words_list, glove_embedding) is None or len(
            find_word_wordnet(token, stop_words_list)) == 0:
        return list()  # Here, we think if a word has no sememe or synonym, we will not replace it

    elif len(find_word_wordnet(token, stop_words_list)) > 0 and len(
            find_word_hownet(token, stop_words_list, glove_embedding)) >= 0:
        synonym_set = set(
            find_word_wordnet(token, stop_words_list) + find_word_hownet(token, stop_words_list, glove_embedding))

        # filter the synonym_set by pos checker
        temp_candidates = list(synonym_set)
        for substitution in temp_candidates:
            pos_tagging = tokenize_into_words(substitution)[0].pos_
            if token.pos_ == "NOUN" or token.pos_ == "VERB":  # we allow nouns and verbs inter-changeably
                substitution_candidates.append(substitution)
            elif token.pos_ == pos_tagging:  # the substitution's pos should be the same with the original word
                substitution_candidates.append(substitution)

        # If a word has sememe and synonyms, we look this word as a meaning word and are ready to replace it
        return substitution_candidates

    """Candidate Strategy Two"""
    # if find_word_hownet(token, stop_words_list, glove_embedding) is None and len(
    #         find_word_wordnet(token, stop_words_list)) == 0:
    #     return list()
    #
    # elif find_word_hownet(token, stop_words_list, glove_embedding) is None and len(
    #         find_word_wordnet(token, stop_words_list)) >= 0:
    #     temp_candidates = find_word_wordnet(token, stop_words_list)
    #     for substitution in temp_candidates:
    #         pos_tagging = tokenize_into_words(substitution)[0].pos_
    #         if token.pos_ == "NOUN" or token.pos_ == "VERB":  # we allow nouns and verbs inter-changeably
    #             substitution_candidates.append(substitution)
    #         elif token.pos_ == pos_tagging:  # the substitution's pos should be the same with the original word
    #             substitution_candidates.append(substitution)
    #
    #     return substitution_candidates
    #
    # elif len(find_word_hownet(token, stop_words_list, glove_embedding)) > 0 and len(
    #         find_word_wordnet(token, stop_words_list)) == 0:
    #     temp_candidates = find_word_hownet(token, stop_words_list, glove_embedding)
    #     for substitution in temp_candidates:
    #         pos_tagging = tokenize_into_words(substitution)[0].pos_
    #         if token.pos_ == "NOUN" or token.pos_ == "VERB":  # we allow nouns and verbs inter-changeably
    #             substitution_candidates.append(substitution)
    #         elif token.pos_ == pos_tagging:  # the substitution's pos should be the same with the original word
    #             substitution_candidates.append(substitution)
    #
    #     return substitution_candidates
    #
    # else:
    #     synonym_set = set(
    #         find_word_wordnet(token, stop_words_list) + find_word_hownet(token, stop_words_list, glove_embedding))
    #
    #     # filter the synonym_set by pos checker
    #     temp_candidates = list(synonym_set)
    #     for substitution in temp_candidates:
    #         pos_tagging = tokenize_into_words(substitution)[0].pos_
    #         if token.pos_ == "NOUN" or token.pos_ == "VERB":  # we allow nouns and verbs inter-changeably
    #             substitution_candidates.append(substitution)
    #         elif token.pos_ == pos_tagging:  # the substitution's pos should be the same with the original word
    #             substitution_candidates.append(substitution)
    #
    #     # If a word has sememe and synonyms, we look this word as a meaning word and are ready to replace it
    #     return substitution_candidates


def get_result_org(original_string):
    """Get predictions of review before do substitution"""
    result = sentiment_analysis_classifier(original_string, truncation="only_first")
    return result[0]["label"], result[0]["score"]


def get_importance_scores(original_tokens, original_label, original_score, batch_size, stop_words_list,
                          glove_embedding):
    """get impact_score of each content word:
       original_tokens -- > spacy, original_label --> str, original_score --> 0-1 float"""
    token_impact_score = []
    content_word_candidates = []  # save the replacement of content word

    for token in original_tokens:  # original_tokens is a list that tokenizing from the whole data
        token_impact_score.append((token, 0))  # initialize the impact score of each token

    for index, token in enumerate(token_impact_score):
        replacement = combine_candidate(token[0], stop_words_list, glove_embedding)
        if len(replacement) == 0:
            continue
        else:
            review_list = []
            # get all the replacement review
            for word in replacement:
                temp_token_impact_score = token_impact_score.copy()
                temp_token_impact_score[index] = (tokenize_into_words(word + " ")[0], 0)
                review_list.append("".join([token[0].text_with_ws for token in temp_token_impact_score]))
            # compute the impact score of content word
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


class USE(object):
    def __init__(self):
        super(USE, self).__init__()

        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi  # Renewed angular similarity

    def semantic_sim(self, sents1, sents2):
        sents1 = [s.lower() for s in sents1]
        sents2 = [s.lower() for s in sents2]
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores[0]


def search_space_reduction(impact_score_list, content_words_candidates, original_label, original_score, batch_size):
    """Do search space reduction: find the initial adversarial text and its perturbation positions"""
    content_words_candidates.sort(key=lambda item: item[2], reverse=True)
    temp_impact_score_list = [item[0] for item in impact_score_list]
    activating_points = []  # record the perturbation points in the sentence

    for item in content_words_candidates:
        review_list = []
        max_score = 0
        optimal_index = 0
        activating_points.append(item[0])

        for word in item[3]:
            temp_impact_score_list[item[0]] = tokenize_into_words(word + " ")[0]
            review_list.append("".join([token.text_with_ws for token in temp_impact_score_list]))
        potential_adv_review = ReviewDataset(review_list)

        for index, result in enumerate(sentiment_analysis_classifier(
                potential_adv_review, batch_size=batch_size, truncation="only_first")):

            if result["label"] != original_label:
                return activating_points, review_list[index], True
            else:
                if original_score - result["score"] > max_score:
                    max_score = original_score - result["score"]
                    optimal_index = index

        temp_impact_score_list[item[0]] = tokenize_into_words(item[3][optimal_index] + " ")[0]  # keep the best answer
    return None, None, False


class Individual:
    """
    Genetic Algorithm: Initialization, Selection, Crossover(Mate) and Mutation
    Each individual represents a text

    Five global variable: nlp, content_words_candidates, review, sentiment_analysis_classifier, original_label
    """

    def __init__(self, ontology, variation_points):
        self.ontology = ontology
        self.variation_points = variation_points
        self.perturbation = self.perturbation_num()
        self.fitness_score = self.cal_fitness()  # fitness score in genetic algorithm

    @staticmethod
    def tokenized_words(text):
        words_list = []
        doc = nlp(text)

        for token in doc:
            words_list.append(token)
        return words_list

    @staticmethod
    def mutate_point(variation):
        """create a mutated replacement"""
        for item in content_words_candidates:
            if item[0] == variation:  # do mutation only in variation point
                return Individual.tokenized_words(random.choice(item[3]) + " ")[0]

    def create_population(self):
        """randomly generate a new sentence based on the variation point"""
        ontology = Individual.tokenized_words(self.ontology)  # get the tokenized list of the original part
        individual = ontology.copy()
        for point in self.variation_points:
            candidate = list()
            for item in content_words_candidates:
                if item[0] == point:
                    candidate = item[3]
            individual[point] = Individual.tokenized_words(random.choice(candidate) + " ")[0]
        return "".join([word.text_with_ws for word in individual])

    def mate(self, parent_2):
        """performing mating and produce new data based on variation point"""
        child_data = list()

        for index, (gw1, gw2) in enumerate(zip(Individual.tokenized_words(self.ontology),
                                               Individual.tokenized_words(parent_2.ontology))):

            if index not in self.variation_points:
                child_data.append(gw1)
            else:
                # random probability
                prob = random.random()
                # if the probability is less than 0.40, get the word from parent 1
                if prob < 0.40:
                    child_data.append(gw1)
                # if the probability is less than 0.80 and more than 0.40, get the word from parent 2
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

    def cal_fitness(self):
        """Define our fitness score(similar to objective function)"""
        result = sentiment_analysis_classifier(self.ontology, truncation="only_first")
        if original_label != result[0]["label"]:
            fitness_score = 1
        else:
            fitness_score = 1 - result[0]["score"]
        return fitness_score


# Argument:
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="./data/IMDB.csv")
parser.add_argument("--victim_model", type=str, help="textattack/bert-base-uncased-imdb")
parser.add_argument("--num", type=str)
parser.add_argument("--output_dir", type=str, help="./attack_result/")

args = parser.parse_args()
data_path = str(args.data_path)  # dataset path (csv)
victim_model = str(args.victim_model)  # fine tuned model from textattack : https://huggingface.co/textattack
num = int(args.num)  # number of examples attacking
output_dir = str(args.output_dir)  # file path of attacking result

# victim model
# If you want to attack your fine-tune model, you can upload your model to hugging-face and then use the code below
model_name = victim_model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model_tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analysis_classifier = pipeline('sentiment-analysis', model=model, tokenizer=model_tokenizer, device=0)

# initialize configuration
configuration = Config()

print("***********************START ATTACK***********************")
success_num = 0  # record adversarial text number
num_count = 0  # number of examples attacking
attack_count = 0  # follow (https://textattack.readthedocs.io/en/latest/): we only attack correct classification
skip_count = 0  # skip the incorrect classification
perturbation_nums = 0  # overall perturbation number
length_sum = 0  # overall length of entire dataset

adversarial_example_df = pd.DataFrame(
    columns=["original review", "original sentiment", "adversarial example", "current sentiment"]
)  # format of our output
bar = tqdm(dataloader(data_path))  # load dataset
stop_words_list = load_stopwords("./stopwords")  # load stopwords[Stopwords are kept when we compute perturb rate]

for text, sentiment in bar:
    review = preprocess_review(text)
    original_label, original_score = get_result_org(review)
    sentiment = ("LABEL_1" if sentiment == 1 else "LABEL_0")

    if original_label == sentiment:
        attack_count += 1
        batch_size = configuration.batch_size
        length_sum += compute_words(review)

        importance_score_list, content_words_candidates = get_importance_scores(tokenize_into_words(review),
                                                                                original_label, original_score,
                                                                                batch_size, stop_words_list,
                                                                                configuration.glove_embedding)
        activating_points, adversarial_text, success_flag = search_space_reduction(importance_score_list,
                                                                                   content_words_candidates,
                                                                                   original_label,
                                                                                   original_score,
                                                                                   batch_size)

        if success_flag is True:  # Find an initialization in search_space_reduction
            success_num += 1

            # Do Iterative Search if the perturbations in text more than 1
            if len(activating_points) > 1:
                temp_adversarial_example = adversarial_text
                temp_perturbation = len(activating_points)
                activating_length = len(activating_points)  # activating_points here means the position perturbed

                for tabu in range(activating_length):
                    population_size = configuration.population_size
                    generation = 1  # Initial the first generation
                    max_iteration = configuration.max_iteration
                    population = []  # save each individual created in it [We have 10 instances in each generation]
                    flag = False  # sign whether find new adversarial example in the new generation

                    instance = Individual(review, activating_points)

                    for _ in range(population_size):  # first generation
                        population.append(Individual(instance.create_population(), activating_points))

                    for _ in range(configuration.max_iteration):  # reproduce next generation until we get max_iteration
                        population.sort(key=lambda x: x.fitness_score, reverse=True)  # Descending order by fitness score

                        # uncomment this line to check the optimization process in genetic algorithm
                        # print("Generation: {}\tString: {}Fitness score: {}".format(generation,
                        #                                                            population[0].ontology,
                        #                                                            population[0].fitness_score))

                        if population[0].fitness_score == 1:
                            flag = True
                            break  # Stop current iteration if we find an adversarial text in one generation

                        new_generation = []

                        elitism = int((20 * population_size) / 100)
                        new_generation.extend(population[:elitism])  # 20 percent will be elitism

                        # select parent from the former generation
                        remains = int((80 * population_size) / 100)
                        select_parent = int(configuration.population_size / 2)
                        for _ in range(remains):
                            parent_1 = random.choice(population[:select_parent])
                            parent_2 = random.choice(population[:select_parent])
                            child = parent_1.mate(parent_2)
                            new_generation.append(child)

                        population = new_generation
                        generation += 1  # finish one generation

                    if flag is True:
                        # save current adversarial_example and number of perturbation
                        temp_adversarial_example = population[0].ontology
                        temp_perturbation = len(activating_points)
                    else:
                        ad_label = ("negative" if original_label == "LABEL_1" else "positive")
                        label = ("negative" if original_label == "LABEL_0" else "positive")
                        adversarial_example_df.loc[len(adversarial_example_df)] = [
                            review, label, temp_adversarial_example, ad_label
                        ]
                        perturbation_nums += temp_perturbation
                        break

                    # perturbed words has been sorted according to their important score and indices are saved.
                    activating_points.pop()  # remove the word with smallest important score in the list

            else:  # if the adversarial text only has one perturbed word, we just return it
                ad_label = ("negative" if original_label == "LABEL_1" else "positive")
                label = ("negative" if original_label == "LABEL_0" else "positive")
                adversarial_example_df.loc[len(adversarial_example_df)] = [review, label, adversarial_text, ad_label]
                perturbation_nums += len(activating_points)

    else:
        skip_count += 1

    num_count += 1
    if num_count == num:
        break

adversarial_example_df.to_csv(output_dir + "result.csv")  # save attack result

# compute semantic similarity using USE
tf.disable_eager_execution()
use = USE()
text_dataframe = pd.read_csv(output_dir + "result.csv")
original_text = list(text_dataframe["original review"])
adversarial_text = list(text_dataframe["adversarial example"])

sum_sim = 0
for review_1, review_2 in tqdm(zip(original_text, adversarial_text)):
    sim = float(use.semantic_sim([review_1], [review_2]))
    sum_sim += sim

print()
print("***************************Attack Information***************************")
# we only count the text that is classified correctly
print("Average length of dataset you already attack: {}".format(length_sum / attack_count))
print("Number of text you attack:{}".format(num))
print("Number of text that is not classified correctly: {}".format(skip_count))  # skip number in textattack
print("Number of text that is attacked successfully: {}".format(success_num))
print("Number of text that we fail to attack: {}".format(attack_count - success_num))
print("Average perturbations in this data are: {}".format(perturbation_nums / attack_count))
print()
print("***************************TAMPERS Metrics***************************")
print("Original Acc: {}".format(attack_count / num))
print("Attacked Acc: {}".format((attack_count - success_num) / num))
print("Success Rate: {}".format(success_num / attack_count))  # Follow textattack: attack_count: correct classification)
print("Perturb Rate: {}".format(perturbation_nums / attack_count / (length_sum / attack_count)))
print("Semantic Similarity: {}".format(sum_sim / success_num))
