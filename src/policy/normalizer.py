import numpy as np

class Normalizer:
    def __init__(self, params):
        self.params = params

    def normalize_(self, x):
        return Normalizer.normalize(x, self.params)
    
    def unnormalize_(self, x):
        return Normalizer.unnormalize(x, self.params)

    @staticmethod
    def normalize(x, params):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x = Normalizer._normalize(x, params)
        x = x.reshape(x_shape)
        return x

    @staticmethod
    def unnormalize(x, params):
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x = Normalizer._unnormalize(x, params)
        x = x.reshape(x_shape)
        return x

    @staticmethod
    def _normalize(x, params):
        if params.type == "z_score":
            return Normalizer.z_score_normalize(x, params["mean_val"], params["std_val"])
        elif params.type == "min_max":
            return Normalizer.min_max_normalize(x, params["min_val"], params["max_val"])
        else:
            raise NotImplementedError
        
    @staticmethod
    def _unnormalize(x, params):
        if params.type == "z_score":
            return Normalizer.z_score_unnormalize(x, params["mean_val"], params["std_val"])
        elif params.type == "min_max":
            return Normalizer.min_max_unnormalize(x, params["min_val"], params["max_val"])
        else:
            raise NotImplementedError

    @staticmethod
    def z_score_normalize(x, mean_val, std_val):
        mean_val = np.array(mean_val)
        std_val = np.array(std_val)
        return (x - mean_val) / std_val

    @staticmethod
    def z_score_unnormalize(x, mean_val, std_val):
        mean_val = np.array(mean_val)
        std_val = np.array(std_val)
        return x * std_val + mean_val

    @staticmethod
    def min_max_normalize(x, min_val, max_val):
        min_val = np.array(min_val)
        max_val = np.array(max_val)
        return (x - min_val) / (max_val - min_val) * 2.0 - 1.0

    @staticmethod
    def min_max_unnormalize(x, min_val, max_val):
        min_val = np.array(min_val)
        max_val = np.array(max_val)
        return (x + 1.0) / 2.0 * (max_val - min_val) + min_val