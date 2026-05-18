# Moduł do obliczania cech czasowych sygnału audio

import numpy as np

# Funkcja do obliczania czasu narastania sygnału
def calculate_rise_time(signal: np.ndarray, sr: int) -> float:
    # Znajdź indeks maksymalnej amplitudy (wartość bezwzględna)
    max_idx = np.argmax(np.abs(signal))
    
    # Oblicz czas w sekundach
    rise_time = max_idx / sr
    
    return rise_time

# Funkcja do obliczania czasu wybrzmiewania sygnału
def calculate_decay_time(signal: np.ndarray, sr: int, threshold: float = 0.1) -> float:
    # Znajdź indeks maksymalnej amplitudy
    max_idx = np.argmax(np.abs(signal))
    max_amplitude = np.abs(signal[max_idx])
    
    # Wartość progowa
    threshold_value = threshold * max_amplitude
    
    # Sygnał od maksimum do końca
    signal_after_max = np.abs(signal[max_idx:])
    
    # Znajdź indeks, gdzie sygnał spadnie poniżej progu
    # (pierwszy punkt poniżej progu)
    below_threshold = np.where(signal_after_max < threshold_value)[0]
    
    if len(below_threshold) > 0:
        # Indeks względem max_idx
        decay_idx = below_threshold[0]
        decay_time = decay_idx / sr
    else:
        # Jeśli sygnał nigdy nie spadnie poniżej progu,
        # decay_time to czas do końca sygnału
        decay_time = (len(signal) - max_idx) / sr
    
    return decay_time

# Funkcja do obliczania Zero Crossing Rate (ZCR)
def calculate_zcr(signal: np.ndarray, max_idx: int, window_length: int = 1000) -> float:
    # Wyciągnij okno od max_idx
    end_idx = min(max_idx + window_length, len(signal))
    window = signal[max_idx:end_idx]
    
    # Jeśli okno jest za krótkie, użyj tego co jest
    if len(window) < 2:
        return 0.0
    
    # Oblicz liczbę przecięć zera
    # Przecięcie zera: zmiana znaku między sąsiednimi próbkami
    sign_changes = np.sum(np.abs(np.diff(np.sign(window)))) / 2
    
    # ZCR jako stosunek do długości okna
    zcr = sign_changes / len(window)
    
    return zcr

# Główna funkcja do ekstrakcji cech czasowych
def extract_time_features(signal: np.ndarray, sr: int, 
                         window_length: int = 1000,
                         decay_threshold: float = 0.1) -> dict:
    # Znajdź indeks maksymalnej amplitudy (będzie używany wielokrotnie)
    max_idx = np.argmax(np.abs(signal))
    
    # Oblicz wszystkie cechy
    rise_time = calculate_rise_time(signal, sr)
    decay_time = calculate_decay_time(signal, sr, threshold=decay_threshold)
    zcr = calculate_zcr(signal, max_idx, window_length=window_length)
    
    features = {
        'rise_time': rise_time,
        'decay_time': decay_time,
        'zcr': zcr,
        'max_idx': max_idx  # Może być przydatne do debugowania
    }
    
    return features
