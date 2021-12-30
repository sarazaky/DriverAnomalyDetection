import numpy as np

from perpare_dataset import load_normal_data


def get_normal_vector(schema_file, batch_size, input_shape, test_model):
    """
   # Arguments
       schema_file: path to the csv schema file for train dataset
       batch_size:
       input_shape: 2D shape

   # return: normal_vector with shape (128, 1)
   """
    normal_ds = load_normal_data(schema_file, batch_size, input_shape)

    # get normal_vector
    normal_v = test_model.predict(normal_ds)
    normal_vector = np.mean(normal_v, axis=0, keepdims=True).reshape(-1, 1)

    return normal_vector


def predict_class(test_data, normal_vector, threshold):
    """
    # Arguments
    test_data: each row represent feature vector of a photo

    # return: predicted classes of shape (n, 1)
    """
    sim = np.dot(test_data, normal_vector)
    return (sim < threshold).astype(int)  # true for an anomaly


class MakePrediction:
    def __init__(self, test_model, normal_vector, threshold):
        self.test_model = test_model
        self.normal_vector = normal_vector
        self.threshold = threshold


    def predict(self, test_ds):
        test_v = self.test_model.predict(test_ds)
        result = predict_class(test_v, self.normal_vector, self.threshold)

        return result






