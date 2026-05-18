# Moduł do klasyfikacji z użyciem k-NN i cross-validation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# Przygotowanie danych do klasyfikacji
def prepare_data(df: pd.DataFrame, class_column: str = 'class', 
                normalize: bool = True):
    # Pobierz nazwy cech (bez metadanych)
    feature_names = [col for col in df.columns 
                    if col not in ['filename', 'class']]
    
    # Macierz cech X
    X = df[feature_names].values
    
    # Wektor etykiet y
    y = df[class_column].values
    
    # Nazwy klas
    class_names = sorted(df[class_column].unique())
    
    # Normalizacja
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, feature_names, class_names, scaler

# Trenowanie i ewaluacja klasyfikatora k-NN z cross-validation
def train_and_evaluate_knn(df: pd.DataFrame, 
                          class_column: str = 'class',
                          n_neighbors: int = 5, 
                          cv_folds: int = 10,
                          normalize: bool = True):
    print("\n" + "=" * 80)
    print(f"KLASYFIKACJA k-NN (k={n_neighbors}, CV={cv_folds}-fold)")
    print("=" * 80)
    
    # Przygotowanie danych
    X, y, feature_names, class_names, scaler = prepare_data(df, class_column, normalize)
    
    print(f"\nRozmiar danych: {X.shape[0]} próbek, {X.shape[1]} cech")
    print(f"Klasy: {class_names}")
    print(f"Normalizacja: {'TAK' if normalize else 'NIE'}")
    
    # Utworzenie klasyfikatora
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Cross-validation - accuracy
    print(f"\n🔄 Trwa cross-validation ({cv_folds}-fold)...")
    cv_scores = cross_val_score(knn, X, y, cv=cv_folds, scoring='accuracy')
    
    # Cross-validation - predykcje (do confusion matrix)
    y_pred = cross_val_predict(knn, X, y, cv=cv_folds)
    
    # Oblicz metryki
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=class_names)
    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
    
    # Wyniki
    results = {
        'cv_scores': cv_scores,
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_names': class_names,
        'feature_names': feature_names,
        'n_neighbors': n_neighbors,
        'cv_folds': cv_folds
    }
    
    return results

# Wyświetlanie wyników klasyfikacji
def display_results(results: dict):
    print("\n" + "=" * 80)
    print("WYNIKI KLASYFIKACJI")
    print("=" * 80)
    
    # Accuracy z cross-validation
    print(f"\n📊 Cross-validation accuracy:")
    print(f"   Średnia: {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
    print(f"   Odchylenie standardowe: {results['std_accuracy']:.4f}")
    print(f"   Min: {results['cv_scores'].min():.4f}")
    print(f"   Max: {results['cv_scores'].max():.4f}")
    
    print(f"\n   Wyniki dla każdego foldu:")
    for i, score in enumerate(results['cv_scores'], 1):
        print(f"   Fold {i:2d}: {score:.4f} ({score*100:.2f}%)")
    
    # Confusion Matrix
    print(f"\n📋 Macierz pomyłek (Confusion Matrix):")
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    # Nagłówek
    print(f"\n{'Rzeczywista →':<18}", end='')
    for class_name in class_names:
        print(f"{class_name[:15]:>15}", end='')
    print()
    print("Predykcja ↓")
    print("-" * (18 + 15 * len(class_names)))
    
    # Wiersze macierzy
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<18}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>15}", end='')
        print()
    
    # Classification Report
    print(f"\n📈 Szczegółowy raport klasyfikacji:")
    print(f"\n{'Klasa':<18} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 80)
    
    report = results['classification_report']
    for class_name in class_names:
        class_metrics = report[class_name]
        print(f"{class_name:<18} {class_metrics['precision']:>10.4f} "
              f"{class_metrics['recall']:>10.4f} {class_metrics['f1-score']:>10.4f} "
              f"{int(class_metrics['support']):>10}")
    
    # Średnie
    print("-" * 80)
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    
    print(f"{'Macro avg':<18} {macro_avg['precision']:>10.4f} "
          f"{macro_avg['recall']:>10.4f} {macro_avg['f1-score']:>10.4f} "
          f"{int(macro_avg['support']):>10}")
    
    print(f"{'Weighted avg':<18} {weighted_avg['precision']:>10.4f} "
          f"{weighted_avg['recall']:>10.4f} {weighted_avg['f1-score']:>10.4f} "
          f"{int(weighted_avg['support']):>10}")

# Rysowanie i zapisywanie macierzy pomyłek
def plot_confusion_matrix(cm, class_names, filepath='../results/confusion_matrix.png',
                         normalize=False):
    # Stwórz folder jeśli nie istnieje
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Normalizacja opcjonalna
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Utwórz figurę
    plt.figure(figsize=(10, 8))
    
    # Heatmapa
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Rzeczywista klasa', fontsize=12)
    plt.xlabel('Predykcja', fontsize=12)
    plt.tight_layout()
    
    # Zapisz
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✅ Macierz pomyłek zapisana do: {filepath}")
    
    plt.close()

# Zapis wyników do pliku tekstowego
def save_results_to_txt(results: dict, filepath: str = '../results/classification_results.txt'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("WYNIKI KLASYFIKACJI k-NN\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Parametry:\n")
        f.write(f"  k (liczba sąsiadów): {results['n_neighbors']}\n")
        f.write(f"  Cross-validation: {results['cv_folds']}-fold\n")
        f.write(f"  Liczba cech: {len(results['feature_names'])}\n")
        f.write(f"  Cechy: {', '.join(results['feature_names'])}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ACCURACY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Średnia accuracy (CV): {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)\n")
        f.write(f"Odchylenie standardowe: {results['std_accuracy']:.4f}\n\n")
        
        f.write("Wyniki dla każdego foldu:\n")
        for i, score in enumerate(results['cv_scores'], 1):
            f.write(f"  Fold {i:2d}: {score:.4f} ({score*100:.2f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("MACIERZ POMYŁEK\n")
        f.write("=" * 80 + "\n\n")
        
        cm = results['confusion_matrix']
        class_names = results['class_names']
        
        # Nagłówek
        f.write(f"{'Rzeczywista →':<18}")
        for class_name in class_names:
            f.write(f"{class_name[:15]:>15}")
        f.write("\n")
        f.write("Predykcja ↓\n")
        f.write("-" * (18 + 15 * len(class_names)) + "\n")
        
        # Wiersze
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<18}")
            for j in range(len(class_names)):
                f.write(f"{cm[i][j]:>15}")
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RAPORT KLASYFIKACJI\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Klasa':<18} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 80 + "\n")
        
        report = results['classification_report']
        for class_name in class_names:
            class_metrics = report[class_name]
            f.write(f"{class_name:<18} {class_metrics['precision']:>10.4f} "
                   f"{class_metrics['recall']:>10.4f} {class_metrics['f1-score']:>10.4f} "
                   f"{int(class_metrics['support']):>10}\n")
    
    print(f"✅ Wyniki zapisane do: {filepath}")
