import pandas as pd
import numpy as np
from pathlib import Path

# Percorso della cartella dei risultati
BASE_DIR = Path("/home/ludovico/workspace/AA-STAL/evaluation/VidOR_use_case/TrackEval_Workspace/data/trackers/mot_challenge/VidOR-eval-train/DINO_Tracker")

def aggregate_trackeval_results():
    all_summaries = []
    headers = None

    # Cerca tutti i file che terminano con _summary.txt, anche nelle sottocartelle
    for file_path in BASE_DIR.rglob("*_summary.txt"):
        if "COMBINED" in file_path.name:
            continue
            
        # Estrae il vero nome della classe gestendo l'anomalia di "screen/monitor"
        rel_path = str(file_path.relative_to(BASE_DIR))
        class_name = rel_path.replace("_summary.txt", "").replace("/", "_")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                continue
                
            # TrackEval separa i valori con spazi variabili; .split() senza argomenti gestisce tutto
            current_headers = lines[0].split()
            current_values = lines[1].split()
            
            if headers is None:
                headers = current_headers
                
            # Converte i valori in float (gestendo eventuali NaN per classi con 0 predizioni)
            numeric_vals = []
            for v in current_values:
                try:
                    numeric_vals.append(float(v))
                except ValueError:
                    numeric_vals.append(np.nan)
                    
            all_summaries.append([class_name] + numeric_vals)

    if not all_summaries:
        print("Nessun file _summary.txt trovato.")
        return

# Crea un DataFrame completo
    df = pd.DataFrame(all_summaries, columns=["Class"] + headers)
    
    # IGNORA le classi che non hanno Ground Truth (GT_Dets == 0) per non sballare le medie
    if 'GT_Dets' in df.columns:
        df.loc[df['GT_Dets'] == 0, df.columns != 'Class'] = np.nan
    
    # Salva il dettaglio completo in CSV
    out_csv = BASE_DIR / "COMBINED_metrics_detailed.csv"
    df.to_csv(out_csv, index=False)
    print(f"[+] Dettaglio per classe salvato in: {out_csv}")

    # Calcola le medie ignorando automaticamente i NaN
    mean_vals = df.select_dtypes(include=[np.number]).mean()

    print("\n" + "="*40)
    print("METRICHE MULTI-CLASSE MEDIE (mHOTA, mMOTA, mIDF1)")
    print("="*40)
    
    # Stampa le metriche principali se presenti
    core_metrics = ['HOTA', 'MOTA', 'IDF1']
    for m in core_metrics:
        if m in mean_vals:
            print(f"{m:<10}: {mean_vals[m]:.3f}%")
            
    print("="*40)

if __name__ == "__main__":
    aggregate_trackeval_results()