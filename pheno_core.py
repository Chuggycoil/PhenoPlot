import json
import os
import sys
import numpy as np
import crypto_utils

# Original Trait Weights
TRAIT_WEIGHTS = np.array([
    3.2,  # Prognathism
    3.5,  # Hair Texture
    4.2,  # Eye Shape
    2.4,  # FWHR
    3.0,  # Cephalic Index
    2.5,  # Height-Length
    3.5,  # Nasal Index
    2.2,  # Facial Index
    2.0,  # Hair Color
    3.8,  # Skin Color
    2.0,  # Eye Color
    0.5,  # Height
    0.5   # Somatotype
])

# Secret 13D scalar multipliers and offsets for exact reversible transformation
SECRET_MULTIPLIERS = np.array([
    0.024512, 0.081245, 0.019842, 0.003412, 0.006214,
    0.007125, 0.005124, 0.006812, 0.041258, 0.038124,
    0.051248, 0.002814, 0.091245
], dtype=float)

SECRET_OFFSETS = np.array([
    0.1245, 0.0512, 0.2145, 0.1842, 0.0912,
    0.1145, 0.0712, 0.1542, 0.0214, 0.1984,
    0.0412, 0.3124, 0.0812
], dtype=float)

# Fixed 16D Permutation Map (13 scaled traits + 3 deterministic noise buffers)
PERMUTATION_16D = [7, 2, 14, 0, 11, 5, 1, 15, 9, 3, 12, 8, 4, 13, 6, 10]
INVERSE_PERMUTATION_16D = np.argsort(PERMUTATION_16D)


def round_to_half(val):
    """Rounds a float value to the nearest 0.5 step."""
    return round(float(val) * 2) / 2.0


def encode_coordinates(vec13):
    """Losslessly encodes a 13D raw coordinate vector into a 16D obfuscated array."""
    arr = np.array(vec13, dtype=float)
    scaled_13 = (arr * SECRET_MULTIPLIERS) + SECRET_OFFSETS
    
    seed_val = int(np.sum(np.abs(arr) * 100)) % 1000
    rng = np.random.RandomState(seed_val)
    padding_3 = rng.uniform(0.1, 0.9, size=3)
    
    combined_16 = np.concatenate([scaled_13, padding_3])
    shuffled_16 = combined_16[PERMUTATION_16D]
    return np.round(shuffled_16, 6).tolist()


def decode_coordinates(vec16):
    """Losslessly decodes a 16D obfuscated array back to exact 13D raw coordinates in RAM."""
    arr = np.array(vec16, dtype=float)
    if len(arr) != 16:
        return vec16.tolist() if isinstance(vec16, np.ndarray) else list(vec16)
        
    unshuffled_16 = arr[INVERSE_PERMUTATION_16D]
    scaled_13 = unshuffled_16[:13]
    raw_13 = (scaled_13 - SECRET_OFFSETS) / SECRET_MULTIPLIERS
    return np.round(raw_13, 4).tolist()


def is_obfuscated(vec):
    """Checks if a vector is in 16D obfuscated format."""
    return len(vec) == 16 and all(0.0 <= x <= 2.5 for x in vec)


def transform_features(raw_features):
    """Original non-linear feature transformation."""
    transformed = list(raw_features)
    
    nasal_val = transformed[6]
    if nasal_val > 72.0:
        transformed[6] = 72.0 + ((nasal_val - 72.0) * 1.5)

    skin_val = transformed[9]
    if skin_val >= 3.0:
        transformed[9] = 2.0 + ((skin_val - 2.0) * 3.8)

    eye_shape_val = transformed[2]
    if eye_shape_val >= 1.8:
        transformed[2] = 1.0 + ((eye_shape_val - 1.0) * 2.5)

    return transformed


def calculate_13d_distance(vec1, vec2):
    """Calculates weighted 13-dimensional Euclidean distance."""
    v1 = np.array(transform_features(vec1), dtype=float)
    v2 = np.array(transform_features(vec2), dtype=float)
    
    weighted_diff = (v1 - v2) * TRAIT_WEIGHTS
    return float(np.sqrt(np.sum(weighted_diff ** 2)))


def calculate_similarity_score(dist):
    """Converts a 13D distance into a percentage similarity score."""
    score = 100.0 * np.exp(-0.12 * dist)
    return round(float(score), 2)


def calculate_coordinate_midpoint(vec1, vec2):
    """Calculates 50/50 vector midpoint between two 13D coordinate sets."""
    arr1 = np.array(vec1, dtype=float)
    arr2 = np.array(vec2, dtype=float)
    return ((arr1 + arr2) / 2.0).tolist()


def parse_coordinate_string(coord_str):
    """Parses raw text coordinate input strings or full 'Sample: [...]' lines into clean 13D float vectors."""
    clean_str = coord_str.strip()
    
    if ":" in clean_str:
        clean_str = clean_str.split(":", 1)[1].strip()
        
    if clean_str.startswith("[") and clean_str.endswith("]"):
        clean_str = clean_str[1:-1]
    
    tokens = [t.strip() for t in clean_str.split(",") if t.strip()]
    parts = []
    for t in tokens:
        try:
            parts.append(float(t))
        except ValueError:
            continue

    if len(parts) == 16:
        return decode_coordinates(parts)
    elif len(parts) == 13:
        return parts
    else:
        raise ValueError(f"Expected 13 or 16 coordinate values, got {len(parts)}.")


def parse_coordinate_with_name(coord_str, default_name="Custom_Raw_Input"):
    """Parses coordinate text input, extracting custom name if provided (e.g. 'Name: [...]')."""
    clean_str = coord_str.strip()
    extracted_name = default_name
    
    if ":" in clean_str:
        parts = clean_str.split(":", 1)
        possible_name = parts[0].replace('"', '').replace("'", '').strip()
        if possible_name:
            extracted_name = possible_name
        clean_str = parts[1].strip()
        
    vec = parse_coordinate_string(clean_str)
    return extracted_name, vec


def format_coordinate(name, vec):
    """Formats sample name and vector into iconic PPC string format for display."""
    obfuscated_vec = encode_coordinates(vec) if len(vec) == 13 else vec
    formatted_vec = ", ".join([f"{v:.6f}" for v in obfuscated_vec])
    return f'"{name}": [{formatted_vec}]'


def get_bundle_path(filename):
    """Helper to locate assets inside PyInstaller bundles or local workspace."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)


def load_phenotypes(filepath="phenotypes.ppc"):
    """Loads database cleanly from any encrypted .ppc file path."""
    if os.path.isabs(filepath) and os.path.exists(filepath):
        target_path = filepath
    else:
        target_path = get_bundle_path(filepath if filepath.endswith(".ppc") else "phenotypes.ppc")

    raw_dict = {}
    if os.path.exists(target_path):
        raw_dict = crypto_utils.load_ppc_file(target_path)
    elif os.path.exists("phenotypes.ppc"):
        raw_dict = crypto_utils.load_ppc_file("phenotypes.ppc")

    decoded_dict = {}
    for name, vec in raw_dict.items():
        if is_obfuscated(vec):
            decoded_dict[name] = decode_coordinates(vec)
        else:
            decoded_dict[name] = vec

    return decoded_dict


def save_phenotypes(data, filepath="phenotypes.ppc"):
    """Encodes coordinates into 16D obfuscated format and saves to encrypted .ppc."""
    encoded_dict = {}
    for name, vec in data.items():
        if len(vec) == 13:
            encoded_dict[name] = encode_coordinates(vec)
        else:
            encoded_dict[name] = vec
            
    crypto_utils.save_ppc_file(encoded_dict, filepath)


def apply_gender_calibration(vector, is_female=False):
    calibrated = list(vector)
    if is_female:
        calibrated[11] = round_to_half(calibrated[11] * 1.075)
        calibrated[3] = round(calibrated[3] * 0.96, 2)
        calibrated[4] = round(calibrated[4] * 1.02, 2)
        calibrated[7] = round(calibrated[7] * 1.02, 2)
    return calibrated
