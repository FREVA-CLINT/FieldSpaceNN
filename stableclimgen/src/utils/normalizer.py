import numpy as np
from torchvision import transforms

class DataNormalizer:
    def __init__(self, normalization, n_dim, data=None, data_stats=None, norm_quantile=1.0):
        data_std, data_mean, data_min, data_max = [], [], [], []

        self.normalization = normalization

        for i in range(n_dim):
            if data is not None:
                data_std.append(np.nanstd(data[i]))
                data_mean.append(np.nanmean(data[i]))
                data_min.append(np.nanquantile(data[i], 1.0 - norm_quantile))
                data_max.append(np.nanquantile(data[i] - data_min[-1], norm_quantile))
            else:
                data_std.append(data_stats['std'][i])
                data_mean.append(data_stats['mean'][i])
                data_min.append(data_stats['min'][i])
                data_max.append(data_stats['max'][i])
        self.data_stats = {'mean': data_mean, 'std': data_std, 'min': data_min, 'max': data_max}

    def normalize(self, data, index):
        if self.normalization == 'std':
            return (data - self.data_stats['mean'][index]) / self.data_stats['std'][index]
        elif self.normalization == 'img':
            return (data - self.data_stats['min'][index]) / self.data_stats['max'][index]
        else:
            return data

    def renormalize(self, data, index):
        if self.normalization == 'std':
            return self.data_stats['std'][index] * data + self.data_stats['mean'][index]
        elif self.normalization == 'img':
            return data * self.data_stats['max'][index] + self.data_stats['min'][index]
        else:
            return data
