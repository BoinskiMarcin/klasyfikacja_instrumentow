#Moduł do obliczania cech spektralnych sygnału audio

import numpy as np
from scipy import signal as sp_signal
from typing import Tuple, List

# Funkcja do przygotowania widma z oknowaniem
def prepare_spectrum(signal: np.ndarray, sr: int, 
                     window_size_ms: float = 20.0,
                     hop_ratio: float = 0.5,
                     window_type: str = 'hamming') -> Tuple[List[np.ndarray], np.ndarray]:
    # Oblicz rozmiar okna w próbkach
    window_samples = int(window_size_ms * sr / 1000.0)
    
    # Oblicz hop_length (przesunięcie okna)
    hop_length = int(window_samples * hop_ratio)
    
    # Stwórz funkcję okna
    if window_type == 'hamming':
        window = np.hamming(window_samples)
    elif window_type == 'hann':
        window = np.hanning(window_samples)
    elif window_type == 'blackman':
        window = np.blackman(window_samples)
    else:
        window = np.hamming(window_samples)  # domyślnie Hamming
    
    # Lista do przechowywania widm
    spectra = []
    
    # Podziel sygnał na okna i oblicz FFT dla każdego
    for start_idx in range(0, len(signal) - window_samples + 1, hop_length):
        # Wyciągnij okno
        windowed_signal = signal[start_idx:start_idx + window_samples] * window
        
        # FFT
        fft_result = np.fft.rfft(windowed_signal)
        
        # Magnitude spectrum (jednostronne widmo amplitudowe)
        magnitude = np.abs(fft_result)
        
        spectra.append(magnitude)
    
    # Oblicz tablicę częstotliwości (tylko dla dodatnich częstotliwości)
    frequencies = np.fft.rfftfreq(window_samples, 1.0/sr)
    
    return spectra, frequencies

# Funkcja do obliczania składowych parzystych i nieparzystych
def calculate_even_odd_components(spectrum: np.ndarray) -> Tuple[float, float]:
    # Indeksy parzyste
    even_indices = np.arange(0, len(spectrum), 2)
    even_sum = np.sum(spectrum[even_indices])
    
    # Indeksy nieparzyste
    odd_indices = np.arange(1, len(spectrum), 2)
    odd_sum = np.sum(spectrum[odd_indices])
    
    return even_sum, odd_sum

# Funkcja do obliczania środka ciężkości widma
def calculate_spectral_centroid(spectrum: np.ndarray, frequencies: np.ndarray) -> float:
    # Unikaj dzielenia przez zero
    total_magnitude = np.sum(spectrum)
    
    if total_magnitude == 0:
        return 0.0
    
    # Centroid = Σ(f[i] * magnitude[i]) / Σ(magnitude[i])
    centroid = np.sum(frequencies * spectrum) / total_magnitude
    
    return centroid

# Funkcja do obliczania nieregularności widma
def calculate_spectral_irregularity(spectrum: np.ndarray) -> float:
    # Unikaj dzielenia przez zero
    total_magnitude = np.sum(spectrum)
    
    if total_magnitude == 0 or len(spectrum) < 2:
        return 0.0
    
    # IR = Σ|magnitude[i] - magnitude[i+1]| / Σ(magnitude[i])
    differences = np.abs(np.diff(spectrum))
    irregularity = np.sum(differences) / total_magnitude
    
    return irregularity

# Główna funkcja do ekstrakcji cech spektralnych
def extract_spectral_features_averaged(signal: np.ndarray, sr: int,
                                      window_size_ms: float = 20.0,
                                      hop_ratio: float = 0.5) -> dict:
    # Przygotuj widma dla wszystkich okien
    spectra, frequencies = prepare_spectrum(signal, sr, window_size_ms, hop_ratio)
    
    if len(spectra) == 0:
        # Sygnał za krótki - zwróć zera
        return {
            'even_components': 0.0,
            'odd_components': 0.0,
            'spectral_centroid': 0.0,
            'spectral_irregularity': 0.0,
            'num_windows': 0
        }
    
    # Listy do przechowywania wartości dla każdego okna
    even_values = []
    odd_values = []
    centroid_values = []
    irregularity_values = []
    
    # Oblicz cechy dla każdego okna
    for spectrum in spectra:
        # Even/Odd components
        even, odd = calculate_even_odd_components(spectrum)
        even_values.append(even)
        odd_values.append(odd)
        
        # Spectral Centroid
        centroid = calculate_spectral_centroid(spectrum, frequencies)
        centroid_values.append(centroid)
        
        # Spectral Irregularity
        irregularity = calculate_spectral_irregularity(spectrum)
        irregularity_values.append(irregularity)
    
    # Uśrednij wartości ze wszystkich okien
    features = {
        'even_components': np.mean(even_values),
        'odd_components': np.mean(odd_values),
        'spectral_centroid': np.mean(centroid_values),
        'spectral_irregularity': np.mean(irregularity_values),
        'num_windows': len(spectra)
    }
    
    return features
