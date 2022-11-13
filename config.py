from library import api


class Config:
    def __init__(self):
        self.glove_embedding = api.load("glove-wiki-gigaword-50")  # choose 50 words similar to the original one
        self.batch_size = 10  # usually set 10 [number of text querying the pipeline everytime]
        self.population_size = 10  # genetic algorithm population size
        self.max_iteration = 100  # Max generation number
