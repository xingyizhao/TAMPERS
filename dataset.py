from library import *


class ReviewDataset(Dataset):
    def __init__(self, review_data):
        self.review_data = review_data

    def __len__(self):
        return len(self.review_data)

    def __getitem__(self, index):
        return self.review_data[index]


def dataloader(data_path):
    file = pd.read_csv(data_path, low_memory=False)
    data, label = list(file["review"]), list(file["sentiment"])

    for index in range(0, len(data), 1):
        yield data[index], label[index]


