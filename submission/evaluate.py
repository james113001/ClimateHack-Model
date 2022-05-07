import numpy as np
import tensorflow as tf
from model import Model

from climatehack import BaseEvaluator


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        
        self.model = tf.keras.models.load_model('saved_model/my_model', compile=False)

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        
        # (12, 128, 128, 1)
        features = np.expand_dims(data, axis = -1)
        
        # (1, 64, 64, 1)
        last_image = np.expand_dims(tf.image.resize_with_crop_or_pad(features[-1], 64, 64), axis = 0)
#         last_image = np.expand_dims(features[-1], axis = 0)
        
        # (1, 1, 64, 64, 1)
        last_image = np.expand_dims(last_image, axis = 0)
        
        #opt flow
        model = Model(data)
        predictionopt = model.generate() 
        assert predictionopt.shape == (24, 64, 64)
#         assert predictionopt.shape == (24, 128, 128)
        # (24, 64, 64, 1)
        predictionopt = np.expand_dims(predictionopt, axis = -1)
        #lstm
        prediction = []
        for i in range(24):
            if i == 0:
                prediction.append(self.model.predict(last_image))
                last_image = prediction[-1]
            elif i < 9:
#                 prediction.append(self.model.predict(last_image))
                predopt = np.expand_dims(tf.image.resize_with_crop_or_pad(predictionopt[i], 64, 64), axis = 0)
                predopt = np.expand_dims(predopt, axis = 0)            
                prediction.append(predopt)
                last_image = prediction[-1]
            else:
                # (1, 64, 64, 1)
                predopt = np.expand_dims(tf.image.resize_with_crop_or_pad(predictionopt[i], 64, 64), axis = 0)
#                 predopt = np.expand_dims(predictionopt[i], axis = 0)
                # (1, 1, 64, 64, 1)
                predopt = np.expand_dims(predopt, axis = 0)
                prediction.append(  (self.model.predict(last_image)  + predopt ) / 2 ) #take avg of two models while training
                last_image = prediction[-1]
        prediction = np.array(prediction)
        prediction = np.squeeze(np.squeeze(prediction, axis = 1), axis = 1)
        prediction= np.squeeze(prediction, axis = -1)
#         prediction= prediction[:, 32:96, 32:96]
        assert prediction.shape == (24, 64, 64)
        
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
