# G4STAB: G-quadruplex Thermodynamic Stability Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16.1](https://img.shields.io/badge/tensorflow-2.16.1-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning ensemble model for predicting G-quadruplex (G4) melting temperatures based on DNA sequence, salt concentration, and pH conditions.

## Overview

G4STAB uses an ensemble of ten deep neural networks to predict the thermodynamic stability (melting temperature) of G-quadruplex structures. The model takes into account:

- **DNA sequence**: Primary structure of the G-quadruplex forming sequence
- **Salt concentration**: K⁺, Na⁺, and other (NH₄⁺/Li⁺) concentrations (mM)
- **pH**: Solution pH value

## Features

- **Sequence-based prediction**: Analyze G4 sequences directly
- **Multi-condition support**: Account for salt concentration and pH effects
- **Ensemble modeling**: Uses 10 models for robust predictions
- **Uncertainty estimation**: Provides prediction confidence intervals
- **Command-line interface**: Easy-to-use CLI for batch processing
- **Batch processing**: Handle multiple sequences efficiently

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8.0 (for GPU acceleration, optional)

### Using Conda (Recommended)

```bash
git clone https://github.com/yourusername/g4stab.git
cd g4stab

conda env create -f environment.yml
conda activate tf
```

### Using pip

```bash
git clone https://github.com/yourusername/g4stab.git
cd g4stab

# Install dependencies
pip install tensorflow==2.16.1 pandas numpy matplotlib tqdm scikit-learn
```

### Input File Format

For batch processing, prepare a CSV file with the following columns:

```csv
sequence,salt_k,salt_na,salt_other,ph
GGGTTAGGGTTAGGGTTAGGG,50,50,0,7.0
GGGTTGGGTTGGGTTGGGT,150,100,5,7.4
```
### Required columns:
- sequence: DNA sequence containing G-quadruplex forming region

### Optional columns:
- salt_k: Potassium concentration (mM), default: 50
- salt_na: Sodium concentration (mM), default: 50
- salt_other: Other salt concentration - NH₄⁺/Li⁺ (mM), default: 0
- ph: pH value, default: 7.0

## Usage

### Command Line Interface

#### Single sequence prediction

```bash
python g4stab_predictor.py -s "GGGTTAGGGTTAGGGTTAGGG"
```

#### Batch prediction from file

```bash
python g4stab_predictor.py -f input_sequences.csv -o predictions.csv
```

#### Custom conditions

```bash
python g4stab_predictor.py -s "GGGTTAGGGTTAGGGTTAGGG" --salt 150 100 1 --ph 7.4
```

### Python API

```python
from g4stab_predictor import G4StabPredictor

# Initialize predictor
predictor = G4StabPredictor("trained_models")

# Single sequence
result = predictor.predict("GGGTTAGGGTTAGGGTTAGGG")
print(f"Predicted Tm: {result['ensemble_mean'].iloc[0]:.1f}°C")

# Multiple sequences with custom conditions
sequences = ["GGGTTAGGGTTAGGGTTAGGG", "GGGTTGGGTTGGGTTGGGT"]
salt_conc = [[150, 100, 0], [50, 100, 5]]  # [K+, Na+, Other] in mM
ph_values = [7.0, 7.4]

results = predictor.predict(sequences, salt_conc, ph_values)
print(results[['sequence', 'ensemble_mean', 'ensemble_std']])
```

## Citation

If you use G4STAB in your research, please cite:

```bibtex
@article{g4stab2024,
  title={G4STAB: Deep Learning Prediction of G-quadruplex Thermodynamic Stability},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024},
  doi={10.xxxx/xxxxx}
}
```

## Step 8: License Section

```markdown
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
