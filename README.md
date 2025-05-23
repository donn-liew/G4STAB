# G4STAB

A multi-input deep learning model to predict G-quadruplex (G4) thermodynamic stability based on sequence and salt concentration.

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
