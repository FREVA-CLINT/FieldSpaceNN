import torch


class DataNormalizer:
    def __init__(self, data_stats_dir):
        self.data_stats = torch.load(data_stats_dir, weights_only=False)

    def normalize(self, data, index):
        pass

    def renormalize(self, data, index):
        pass

class MinMaxNormalizer(DataNormalizer):
    def __init__(self, data_stats_dir):
        super().__init__(data_stats_dir)

    def normalize(self, data, index):
        return (data - self.data_stats['min'][index]) / self.data_stats['max'][index]

    def renormalize(self, data, index):
        return data * self.data_stats['max'][index] + self.data_stats['min'][index]
