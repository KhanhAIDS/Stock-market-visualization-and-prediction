import pandas as pd
import os
from glob import glob

def clean_or_remove(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(lines) <= 4:
            print(f"Removing {os.path.basename(file_path)} (insufficient data)")
            os.remove(file_path)
            return False
        
        cleaned_lines = [lines[0]] + lines[4:]
        
        temp_path = file_path + ".temp"
        with open(temp_path, 'w') as f:
            f.write('\n'.join(cleaned_lines))
        
        df = pd.read_csv(temp_path)
        
        if 'Price' in df.columns:
            df = df.rename(columns={'Price': 'Date'})
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        df = df.dropna(subset=['Date'])
        if df.empty:
            print(f"Removing {os.path.basename(file_path)} (no valid dates)")
            os.remove(file_path)
            return False
        
        df.to_csv(file_path, index=False)
        os.remove(temp_path)
        return True
        
    except Exception as e:
        print(f"Removing {os.path.basename(file_path)} (processing failed: {str(e)})")
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def clean_folder(folder_path):
    csv_files = glob(os.path.join(folder_path, "*.csv"))
    print(f"\nProcessing {len(csv_files)} files in {folder_path}...")
    
    results = {"cleaned": 0, "removed": 0}
    for file in csv_files:
        if clean_or_remove(file):
            results["cleaned"] += 1
        else:
            results["removed"] += 1
    
    print(f"\nResults for {folder_path}:")
    print(f"Successfully cleaned: {results['cleaned']}")
    print(f"Removed: {results['removed']}")
    print(f"Cleaned/removed ratio: {results['cleaned']/len(csv_files):.1%}")
    return results

total = {"cleaned": 0, "removed": 0, "total": 0}

for folder in ["etfs", "stocks"]:
    if os.path.exists(folder):
        res = clean_folder(folder)
        total["cleaned"] += res["cleaned"]
        total["removed"] += res["removed"]
        total["total"] += res["cleaned"] + res["removed"]

print(f"\nFINAL TOTALS:")
print(f"Total files processed: {total['total']}")
print(f"Files kept and cleaned: {total['cleaned']}")
print(f"Files removed: {total['removed']}")
print(f"Overall kept ratio: {total['cleaned']/total['total']:.1%}")