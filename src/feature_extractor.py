# Moduł do ekstrakcji cech z sygnałów audio

import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

# Import modułów z cechami
from time_features import extract_time_features
from spectral_features import extract_spectral_features_averaged

# Ekstrakcja wszystkich cech dla pojedynczego sygnału
def extract_all_features(signal: np.ndarray, sr: int, 
                        filename: str, class_name: str,
                        window_length_zcr: int = 1000,
                        spectral_window_ms: float = 20.0) -> dict:
    # Ekstrakcja cech czasowych
    time_feats = extract_time_features(signal, sr, 
                                       window_length=window_length_zcr)
    
    # Ekstrakcja cech spektralnych (uśrednionych)
    spectral_feats = extract_spectral_features_averaged(signal, sr,
                                                        window_size_ms=spectral_window_ms)
    
    # Połączenie wszystkich cech w jeden słownik
    features = {
        # Metadane
        'filename': filename,
        'class': class_name,
        
        # Cechy czasowe
        'rise_time': time_feats['rise_time'],
        'decay_time': time_feats['decay_time'],
        'zcr': time_feats['zcr'],
        
        # Cechy spektralne
        'even_components': spectral_feats['even_components'],
        'odd_components': spectral_feats['odd_components'],
        'spectral_centroid': spectral_feats['spectral_centroid'],
        'spectral_irregularity': spectral_feats['spectral_irregularity']
    }
    
    return features

# Ekstrakcja cech dla całego zbioru danych
def extract_features_from_dataset(audio_data: List[Tuple[np.ndarray, int, str, str]],
                                  window_length_zcr: int = 1000,
                                  spectral_window_ms: float = 20.0,
                                  show_progress: bool = True) -> pd.DataFrame:
    features_list = []
    
    # Iterator z progress barem (opcjonalnie)
    iterator = tqdm(audio_data, desc="Ekstrakcja cech") if show_progress else audio_data
    
    for signal, sr, class_name, filename in iterator:
        try:
            # Ekstrakcja cech dla pojedynczego pliku
            features = extract_all_features(signal, sr, filename, class_name,
                                          window_length_zcr=window_length_zcr,
                                          spectral_window_ms=spectral_window_ms)
            
            features_list.append(features)
            
        except Exception as e:
            print(f"\n❌ Błąd przy ekstrakcji cech z {filename}: {e}")
            continue
    
    # Konwersja do DataFrame
    df = pd.DataFrame(features_list)
    
    # Uporządkowanie kolumn: najpierw metadane, potem cechy
    column_order = [
        'filename', 'class',
        'rise_time', 'decay_time', 'zcr',
        'even_components', 'odd_components', 
        'spectral_centroid', 'spectral_irregularity'
    ]
    
    df = df[column_order]
    
    return df

# Zwraca listę nazw cech
def get_feature_names(include_metadata: bool = False) -> List[str]:
    if include_metadata:
        return [
            'filename', 'class',
            'rise_time', 'decay_time', 'zcr',
            'even_components', 'odd_components', 
            'spectral_centroid', 'spectral_irregularity'
        ]
    else:
        return [
            'rise_time', 'decay_time', 'zcr',
            'even_components', 'odd_components', 
            'spectral_centroid', 'spectral_irregularity'
        ]

# Zapis cech do pliku CSV
def save_features_to_csv(df: pd.DataFrame, filepath: str = '../results/features.csv'):
    import os
    
    # Stwórz folder results jeśli nie istnieje
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Zapis do CSV
    df.to_csv(filepath, index=False, float_format='%.6f')
    print(f"✅ Cechy zapisane do: {filepath}")
