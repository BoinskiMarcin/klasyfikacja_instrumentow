# Moduł do wczytywania plików audio .wav

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import scipy.io.wavfile as wav

# Wczytywanie pojedynczego pliku .wav
def load_audio_file(filepath: str) -> Optional[Tuple[np.ndarray, int, str, str]]:
    try:
        path = Path(filepath)
        
        # Wczytanie pliku
        sr, data = wav.read(filepath)
        
        # Konwersja do float i normalizacja do zakresu [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128.0
        
        # Jeśli stereo, konwertuj na mono (średnia z kanałów)
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)
        
        # Nazwa klasy (folder nadrzędny)
        class_name = path.parent.name
        
        # Nazwa pliku
        filename = path.name
        
        return data, sr, class_name, filename
        
    except Exception as e:
        print(f"❌ Błąd przy wczytywaniu {filepath}: {e}")
        return None

# Wczytywanie wszystkich plików .wav
def load_all_audio_files(data_dir: str = '../data') -> List[Tuple[np.ndarray, int, str, str]]:
    data_path = Path(data_dir)
    
    # Sprawdzenie czy katalog istnieje
    if not data_path.exists():
        raise FileNotFoundError(f"Katalog '{data_dir}' nie istnieje!")
    
    # Znalezienie wszystkich plików .wav
    wav_files = list(data_path.rglob('*.wav'))
    
    if not wav_files:
        raise FileNotFoundError(f"Nie znaleziono żadnych plików .wav w katalogu '{data_dir}'")
    
    print(f"🔍 Znaleziono {len(wav_files)} plików .wav")
    
    # Wczytanie wszystkich plików
    audio_data = []
    successful = 0
    failed = 0
    
    for wav_file in sorted(wav_files):
        result = load_audio_file(str(wav_file))
        
        if result is not None:
            audio_data.append(result)
            successful += 1
        else:
            failed += 1
    
    print(f"✅ Pomyślnie wczytano: {successful} plików")
    
    if failed > 0:
        print(f"❌ Nie udało się wczytać: {failed} plików")
    
    return audio_data
