# Moduł do obliczania statystyk dla cech audio

import pandas as pd
import numpy as np

# Zliczanie statystyk dla cech w DataFrame
def calculate_statistics(df: pd.DataFrame, class_column: str = 'class') -> pd.DataFrame:
    # Pobierz listę cech numerycznych (bez metadanych)
    numeric_features = [col for col in df.columns 
                       if col not in ['filename', 'class'] and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    # Lista do przechowywania statystyk
    stats_list = []
    
    # Dla każdej klasy
    for class_name in sorted(df[class_column].unique()):
        # Filtruj dane dla danej klasy
        class_data = df[df[class_column] == class_name]
        
        # Dla każdej cechy
        for feature in numeric_features:
            feature_values = class_data[feature]
            
            stats = {
                'class': class_name,
                'feature': feature,
                'min': feature_values.min(),
                'max': feature_values.max(),
                'mean': feature_values.mean(),
                'std': feature_values.std(),
                'median': feature_values.median(),
                'count': len(feature_values)
            }
            
            stats_list.append(stats)
    
    # Konwersja do DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    return stats_df

# Zapis statystyk do pliku Excel
def save_statistics_to_excel(stats_df: pd.DataFrame, 
                             filepath: str = '../results/statistics.xlsx') -> None:
    import os
    
    # Stwórz folder results jeśli nie istnieje
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Zapis do Excel
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Jedna tabela ze wszystkim
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        # Formatowanie
        workbook = writer.book
        worksheet = writer.sheets['Statistics']
        
        # Szerokość kolumn
        worksheet.column_dimensions['A'].width = 18  # class
        worksheet.column_dimensions['B'].width = 25  # feature
        worksheet.column_dimensions['C'].width = 15  # min
        worksheet.column_dimensions['D'].width = 15  # max
        worksheet.column_dimensions['E'].width = 15  # mean
        worksheet.column_dimensions['F'].width = 15  # std
        worksheet.column_dimensions['G'].width = 15  # median
        worksheet.column_dimensions['H'].width = 10  # count
    
    print(f"✅ Statystyki zapisane do: {filepath}")

# Zapis statystyk do pliku CSV
def save_statistics_to_csv(stats_df: pd.DataFrame, 
                           filepath: str = '../results/statistics.csv') -> None:
    import os
    
    # Stwórz folder results jeśli nie istnieje
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Zapis do CSV
    stats_df.to_csv(filepath, index=False, float_format='%.6f')
    print(f"✅ Statystyki zapisane do: {filepath}")
