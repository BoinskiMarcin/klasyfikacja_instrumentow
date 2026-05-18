# Główny plik programu

import os
import sys

# Import modułów projektu
from audio_loader import load_all_audio_files
from feature_extractor import (extract_features_from_dataset, 
                               save_features_to_csv)
from stats import calculate_statistics, save_statistics_to_csv, save_statistics_to_excel
from classification import (train_and_evaluate_knn,
                            display_results,
                            plot_confusion_matrix,
                            save_results_to_txt)


# KONFIGURACJA PROJEKTU
# Ścieżki
DATA_DIR = '../data'                                    # Folder z plikami .wav
RESULTS_DIR = '../results'                              # Folder na wyniki

# Parametry ekstrakcji cech
WINDOW_LENGTH_ZCR = 1000                             # Długość okna dla ZCR [próbki]
SPECTRAL_WINDOW_MS = 20.0                            # Długość okna dla cech spektralnych [ms]

# Parametry klasyfikacji
KNN_NEIGHBORS = 5                                     # k dla k-NN
CV_FOLDS = 10                                        # Liczba foldów dla cross-validation
NORMALIZE_FEATURES = True                            # Czy normalizować cechy przed klasyfikacją

# GŁÓWNA FUNKCJA PROGRAMU
def main():
    print("PROJEKT: ANALIZA SYGNAŁÓW AUDIO I KLASYFIKACJA INSTRUMENTÓW")
    print()

    try:
        audio_data = load_all_audio_files(DATA_DIR)
    except Exception as e:
        print(f"\n❌ BŁĄD podczas wczytywania plików: {e}")
        sys.exit(1)
    
    if len(audio_data) == 0:
        print("\n❌ BŁĄD: Nie udało się wczytać żadnych plików")
        sys.exit(1)
    
    try:
        features_df = extract_features_from_dataset(
            audio_data,
            window_length_zcr=WINDOW_LENGTH_ZCR,
            spectral_window_ms=SPECTRAL_WINDOW_MS,
            show_progress=True
        )
    except Exception as e:
        print(f"\n❌ BŁĄD podczas ekstrakcji cech: {e}")
        sys.exit(1)
    
    print(f"\n✅ Wyekstrahowano cechy dla {len(features_df)} plików")
    
    # Zapis cech do CSV
    features_csv = os.path.join(RESULTS_DIR, 'features.csv')
    save_features_to_csv(features_df, features_csv)
    
    try:
        stats_df = calculate_statistics(features_df)
    except Exception as e:
        print(f"\n❌ BŁĄD podczas obliczania statystyk: {e}")
        sys.exit(1)
    
    print(f"\n✅ Obliczono statystyki dla {len(stats_df['feature'].unique())} cech")
    print(f"   Liczba klas: {len(stats_df['class'].unique())}")
    
    # Zapis statystyk
    stats_csv = os.path.join(RESULTS_DIR, 'statistics.csv')
    stats_excel = os.path.join(RESULTS_DIR, 'statistics.xlsx')
    
    save_statistics_to_csv(stats_df, stats_csv)
    save_statistics_to_excel(stats_df, stats_excel)
    
    # klasyfikacja k-NN z cross-validation
    try:
        results = train_and_evaluate_knn(
            features_df,
            n_neighbors=KNN_NEIGHBORS,
            cv_folds=CV_FOLDS,
            normalize=NORMALIZE_FEATURES
        )
    except Exception as e:
        print(f"\n❌ BŁĄD podczas klasyfikacji: {e}")
        sys.exit(1)
    
    # Wyświetl wyniki
    display_results(results)
    
    # Macierz pomyłek (obrazek)
    cm_png = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'],
                         results['class_names'],
                         cm_png)
    
    # Raport tekstowy
    results_txt = os.path.join(RESULTS_DIR, 'classification_results.txt')
    save_results_to_txt(results, results_txt)
    
    print("\nANALIZA ZAKOŃCZONA POMYŚLNIE")
    
    print("\n📁 Wygenerowane pliki w folderze 'results/':")
    print(f"   1. features.csv                  - ekstrahowane cechy dla wszystkich plików")
    print(f"   2. statistics.csv                - statystyki (min, max, mean, std, median) per klasa")
    print(f"   3. statistics.xlsx               - statystyki w formacie Excel")
    print(f"   4. confusion_matrix.png          - wizualizacja macierzy pomyłek")
    print(f"   5. classification_results.txt    - pełny raport klasyfikacji")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program przerwany przez użytkownika (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ NIEOCZEKIWANY BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)