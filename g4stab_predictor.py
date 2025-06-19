"""
G4STAB: G-quadruplex Thermodynamic Stability Predictor

A tool for predicting G-quadruplex (G4) melting temperatures based on sequence,
salt concentration, and pH using ensemble deep learning models.
"""

import os, sys, argparse, warnings, re, itertools, collections, pickle
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
except ImportError as e:
    print(f"Error: Required packages missing. Install with: pip install tensorflow==2.16.1 scikit-learn")
    sys.exit(1)

def oneHot(list_of_strings, max_sequence=0, verbose=False):
    """One-hot encode DNA sequences."""
    def efficient_padding(matrix, max_sequence):
        padding_length = max_sequence - matrix.shape[0]
        pad_before = padding_length // 2
        pad_after = padding_length - pad_before
        pads = [(pad_before, pad_after), (0, 0)]
        return np.pad(matrix, pads, 'constant', constant_values=0).astype(np.int8)

    def efficient_slicing(matrix, max_sequence):
        if len(matrix) > max_sequence:
            start = (len(matrix) - max_sequence) // 2
            end = start + max_sequence
            return matrix[start:end]
        return matrix
    
    nucleotide_garden = {'A':0, 'T':1, 'C':2, 'G':3, 'U':1}
    matrix_compilation = []
    
    for string in tqdm(list_of_strings, desc='One-hot encoding...', disable=not verbose):
        matrix = np.zeros((len(string), 4), dtype=np.int8)
        
        for idl, letter in enumerate(string):
            if letter in nucleotide_garden:
                matrix[idl, nucleotide_garden[letter]] = 1
        
        if max_sequence > len(string):
            matrix_compilation.append(efficient_padding(matrix=matrix, max_sequence=max_sequence))
        elif max_sequence < len(string):
            matrix_compilation.append(efficient_slicing(matrix=matrix, max_sequence=max_sequence))
        else:
            matrix_compilation.append(matrix)
    
    return np.array(matrix_compilation, dtype=np.int8)

def generate_kmers(sequence, get_kmers_legend=False):
    """Generate k-mer features for a sequence."""
    all_kmers = [''.join(p) for k in range(1, 5) for p in itertools.product('ATCG', repeat=k)]
    kmer_counts = {kmer: 0 for kmer in all_kmers}
    if get_kmers_legend:
        return np.array(list(kmer_counts.keys())).reshape(20, 17)
    kmer_list = [sequence[i:i+k] for k in range(1, 5) for i in range(len(sequence) - k + 1)]
    
    counted_kmers = collections.Counter(kmer_list)
    kmer_counts.update(counted_kmers)
    return np.array(list(kmer_counts.values())).reshape(20, 17)

def expand_sequence(sequence):
    """Expand sequence notation like G3T4 -> GGGTTT"""
    def expand_simple(seq):
        return re.sub(r'([ATCG])(\d+)', 
                     lambda m: m.group(1) * int(m.group(2)), 
                     seq)
    
    max_iterations = 100
    iteration = 0
    
    while '(' in sequence and iteration < max_iterations:
        sequence = re.sub(r'\(([^()]*)\)(\d+)',
                         lambda m: expand_simple(m.group(1)) * int(m.group(2)),
                         sequence, 1)
        iteration += 1
    
    if iteration >= max_iterations:
        raise ValueError("Maximum iteration limit reached. Check for invalid input pattern.")

    sequence = expand_simple(sequence)
    sequence = sequence.replace('-', '')
    sequence = sequence.replace(' ', '')
    
    return sequence

def load_scalers(scalers_dir="trained_models", data_file="Dataset (G4STAB) Supplementary Table 1.csv"):
    """Load or create scalers for normalization."""
    scaler_salt_path = os.path.join(scalers_dir, "scaler_salt.pkl")
    scaler_ph_path = os.path.join(scalers_dir, "scaler_ph.pkl")

    try:
        with open(scaler_salt_path, 'rb') as f:
            scaler_salt = pickle.load(f)
        with open(scaler_ph_path, 'rb') as f:
            scaler_ph = pickle.load(f)
        return {'scaler_salt': scaler_salt, 'scaler_ph': scaler_ph}
    except FileNotFoundError as e:
        raise RuntimeError(f"Scalers not found: {e}. Make sure {scalers_dir}/ contains scaler_salt.pkl and scaler_ph.pkl")

def seqmap(sequence, conc=False, ph=False):
    """
    Feature engineering function for G4 sequences.
    
    Args:
        sequence: List-like of sequences or single string
        conc: List-like of [K+, Na+, (NH4+/Li+)], units: mM
        ph: List-like of pH values
    """
    tuners = load_scalers()
    
    def processing(sequence, conc, ph):
        X_ohe = oneHot([sequence], max_sequence=100)[0]
        X_kmer = generate_kmers(sequence)
        X_salt = tuners['scaler_salt'].transform(np.array(conc).reshape(1, -1))[0]
        X_pH = tuners['scaler_ph'].transform(np.expand_dims(np.array([ph]), axis=1))[0]
        
        return X_ohe, X_kmer, X_salt, X_pH
    
    if isinstance(sequence, str):
        return [np.expand_dims(x, 0) for x in processing(sequence, conc=conc, ph=ph)]
    else:
        results = [processing(a, b, c) for a, b, c in zip(sequence, conc, ph)]
        
        X_ohe = np.array([result[0] for result in results], dtype=np.int8)
        X_kmer = np.array([result[1] for result in results], dtype=np.int16)
        X_salt = np.array([result[2] for result in results], dtype=np.float32)
        X_pH = np.array([result[3] for result in results], dtype=np.float32)
        
        return X_ohe, X_kmer, X_salt, X_pH

class G4StabPredictor:
    """
    G-quadruplex thermodynamic stability predictor using ensemble deep learning models.
    
    This class loads pre-trained models and makes predictions on G4 sequences
    considering salt concentration and pH conditions.
    """
    
    def __init__(self, models_dir: str = "trained_models"):
        """
        Initialize the predictor with trained models.
        
        Args:
            models_dir (str): Directory containing the trained model files
        """
        self.models_dir = Path(models_dir)
        self.models = []
        self.model_names = []
        self._load_models()
        
    def _load_models(self):
        """Load all trained models from the models directory."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found")
        
        model_files = list(self.models_dir.glob("*.h5")) + list(self.models_dir.glob("*.keras"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in '{self.models_dir}'")
        
        print(f"Loading {len(model_files)} models...")
        
        for model_file in sorted(model_files):
            try:
                model = load_model(model_file, compile=False)
                self.models.append(model)
                self.model_names.append(model_file.stem)
                print(f"  ✓ Loaded {model_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to load {model_file.name}: {e}")
        
        if not self.models:
            raise RuntimeError("No models could be loaded successfully")
        
        print(f"Successfully loaded {len(self.models)} models")
    
    def _validate_sequence(self, sequence: str) -> bool:
        """
        Validate if sequence contains valid nucleotides.
        
        Args:
            sequence (str): DNA sequence to validate
            
        Returns:
            bool: True if sequence is valid
        """
        valid_nucleotides = set('ATCG')
        return all(nt.upper() in valid_nucleotides for nt in sequence)
    
    def _prepare_features(self, sequences: List[str], salt_concentrations: List[List[float]], 
                         ph_values: List[float]) -> List[np.ndarray]:
        """
        Prepare input features for the models using seqmap function.
        
        Args:
            sequences: List of DNA sequences
            salt_concentrations: List of salt concentration arrays [K+, Na+, (NH4+/Li+)]
            ph_values: List of pH values
            
        Returns:
            List of feature arrays for model input [X_ohe, X_kmer, X_salt, X_ph]
        """
        try:
            # Use seqmap function for feature engineering
            X_ohe, X_kmer, X_salt, X_ph = seqmap(sequences, salt_concentrations, ph_values)
            return [X_ohe, X_kmer, X_salt, X_ph]
        except Exception as e:
            raise RuntimeError(f"Feature engineering failed: {e}. Make sure 'Dataset (G4STAB) Supplementary Table 1.csv' is available for scaler loading.")
    
    def _create_model_inputs(self, model, features: List[np.ndarray]) -> dict:
        """
        Create properly formatted input dictionary for a model.
        
        Args:
            model: Keras model
            features: List of feature arrays
            
        Returns:
            dict: Input dictionary mapping input names to arrays
        """
        input_shapes = {inp.name: inp.shape[1:] for inp in model.inputs}
        input_dict = {}
        
        for feature_array in features:
            for input_name, expected_shape in input_shapes.items():
                if input_name in input_dict:
                    continue
                    
                current_shape = feature_array.shape[1:]
                
                # Direct match
                if current_shape == expected_shape:
                    input_dict[input_name] = feature_array
                # Add single dimension
                elif current_shape + (1,) == expected_shape:
                    input_dict[input_name] = np.expand_dims(feature_array, axis=-1)
                # Add two dimensions
                elif current_shape + (1, 1) == expected_shape:
                    expanded = np.expand_dims(np.expand_dims(feature_array, axis=-1), axis=-1)
                    input_dict[input_name] = expanded
        
        return input_dict
    
    def predict(self, sequences: Union[str, List[str]], 
                salt_concentrations: Union[List[float], List[List[float]]] = None,
                ph_values: Union[float, List[float]] = None) -> pd.DataFrame:
        """
        Predict melting temperatures for G4 sequences.
        
        Args:
            sequences: Single sequence string or list of sequences
            salt_concentrations: Salt concentrations [K+, Na+, (NH4+/Li+)] in mM.
                                Single list for all sequences or list of lists for each sequence.
                                Default: [100, 0, 0]
            ph_values: pH value(s). Single value for all sequences or list for each sequence.
                      Default: 7.0
                      
        Returns:
            pd.DataFrame: Results with sequences, individual model predictions, and ensemble mean
        """
        # Normalize inputs to lists
        if isinstance(sequences, str):
            sequences = [sequences]
        
        n_sequences = len(sequences)
        
        # Set default salt concentrations
        if salt_concentrations is None:
            salt_concentrations = [[100.0, 0.0, 0.0]] * n_sequences
        elif isinstance(salt_concentrations[0], (int, float)):
            # Single salt concentration for all sequences
            salt_concentrations = [salt_concentrations] * n_sequences
        
        # Set default pH values
        if ph_values is None:
            ph_values = [7.0] * n_sequences
        elif isinstance(ph_values, (int, float)):
            ph_values = [ph_values] * n_sequences
        
        # Validate inputs
        if len(salt_concentrations) != n_sequences:
            raise ValueError("Number of salt concentrations must match number of sequences")
        if len(ph_values) != n_sequences:
            raise ValueError("Number of pH values must match number of sequences")
        
        # Validate sequences
        for i, seq in enumerate(sequences):
            if not self._validate_sequence(seq):
                raise ValueError(f"Invalid sequence at index {i}: {seq}")
        
        print(f"Predicting melting temperatures for {n_sequences} sequence(s)...")
        
        # Prepare features
        try:
            features = self._prepare_features(sequences, salt_concentrations, ph_values)
        except Exception as e:
            raise RuntimeError(f"Feature preparation failed: {e}")
        
        # Make predictions with each model
        predictions = []
        for i, model in enumerate(self.models):
            try:
                model_inputs = self._create_model_inputs(model, features)
                pred = model.predict(model_inputs, verbose=0)
                # Convert from normalized scale (0-1) to temperature (°C)
                predictions.append(pred.flatten() * 100)
            except Exception as e:
                print(f"Warning: Model {self.model_names[i]} failed: {e}")
                predictions.append(np.full(n_sequences, np.nan))
        
        # Create results DataFrame
        results = pd.DataFrame({
            'sequence': sequences,
            'salt_k': [sc[0] for sc in salt_concentrations],
            'salt_na': [sc[1] for sc in salt_concentrations], 
            'salt_other': [sc[2] for sc in salt_concentrations],
            'ph': ph_values
        })
        
        # Add individual model predictions
        for name, pred in zip(self.model_names, predictions):
            results[name] = pred
        
        # Calculate ensemble statistics
        pred_matrix = np.array(predictions).T
        results['ensemble_mean'] = np.nanmean(pred_matrix, axis=1)
        results['ensemble_std'] = np.nanstd(pred_matrix, axis=1)
        
        return results

def main():
    """Command-line interface for G4STAB predictor."""
    parser = argparse.ArgumentParser(
        description='Predict G-quadruplex melting temperatures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -s GGGTTAGGGTTAGGGTTAGGG
  %(prog)s -f sequences.csv -o predictions.csv
  %(prog)s -s GGGTTAGGGTTAGGGTTAGGG --salt 150 100 5 --ph 7.4
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-s', '--sequence', type=str,
                           help='Single DNA sequence to predict')
    input_group.add_argument('-f', '--file', type=str,
                           help='CSV file with sequences (must have "sequence" column)')
    
    parser.add_argument('-o', '--output', type=str,
                       help='Output CSV file (default: print to stdout)')
    parser.add_argument('--models-dir', type=str, default='trained_models',
                       help='Directory containing trained models (default: trained_models)')
    parser.add_argument('--salt', type=float, nargs=3, default=[100.0, 0.0, 0.0], 
                       metavar=('K+', 'Na+', 'Other'),
                       help='Salt concentrations in mM: K+ Na+ Other(NH4+/Li+) (default: 100 0 0)')
    parser.add_argument('--ph', type=float, default=7.0,
                       help='pH value (default: 7.0)')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = G4StabPredictor(args.models_dir)
        
        # Prepare input
        if args.sequence:
            sequences = [args.sequence]
            salt_concentrations = [args.salt]
            ph_values = [args.ph]
        else:
            # Read from file
            try:
                df = pd.read_csv(args.file)
                if 'sequence' not in df.columns:
                    raise ValueError("Input file must contain a 'sequence' column")
                
                sequences = df['sequence'].tolist()
                
                # Check for salt and pH columns, otherwise use defaults
                if all(col in df.columns for col in ['salt_k', 'salt_na', 'salt_other']):
                    salt_concentrations = df[['salt_k', 'salt_na', 'salt_other']].values.tolist()
                else:
                    salt_concentrations = [args.salt] * len(sequences)
                
                if 'ph' in df.columns:
                    ph_values = df['ph'].tolist()
                else:
                    ph_values = [args.ph] * len(sequences)
                    
            except Exception as e:
                print(f"Error reading input file: {e}")
                sys.exit(1)
        
        # Make predictions
        results = predictor.predict(sequences, salt_concentrations, ph_values)
        
        # Output results
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print(results.to_string(index=False))
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
