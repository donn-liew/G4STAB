# G4STAB: G-quadruplex Thermodynamic Stability Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16.1](https://img.shields.io/badge/tensorflow-2.16.1-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning ensemble model for predicting G-quadruplex (G4) melting temperatures based on DNA sequence, salt concentration, and pH conditions.

G4STAB is developed in Python 3.10.15 and uses the following modules:
 - TensorFlow 2.16.1
 - CUDA Toolkit 11.8.0
 - Keras 3.3.3

For a complete list of dependencies, refer to `environment.yml`.

Trained models are stored in the `trained_models` folder. All models use identical architectures and hyperparameters, trained on the same dataset but with different random shuffles of the training data.

## Making Melting Temperature Prediction

**For melting temperature prediction of putative G4 sequence:**
- Use all 5 models to make predictions
- Take the average of the 5 predictions as your final result
- This ensemble approach provides more robust and accurate predictions than using a single model
