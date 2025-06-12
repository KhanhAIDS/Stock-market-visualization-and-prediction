import pandas as pd
import yfinance as yf
import os
import shutil
from os.path import isfile, join
import contextlib

def download_nasdaq_data(offset=0, limit=None, period='5y'):

    print("Loading NASDAQ symbol list...")
    data = pd.read_csv("http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep='|')
    data_clean = data[data['Test Issue'] == 'N']
    symbols = data_clean['NASDAQ Symbol'].tolist()
    
    limit = limit if limit else len(symbols)
    end = min(offset + limit, len(symbols))
    
    print(f"Processing symbols {offset+1} to {end} of {len(symbols)} total symbols")
    
    os.makedirs('hist', exist_ok=True)
    os.makedirs('stocks', exist_ok=True)
    os.makedirs('etfs', exist_ok=True)
    
    is_valid = [False] * len(symbols)
    download_errors = []
    
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            for i in range(offset, end):
                s = symbols[i]
                try:
                    if not isinstance(s, str):
                        download_errors.append(f"Skipped non-string symbol at index {i}")
                        continue
                        
                    data = yf.download(s, period=period)
                    
                    if len(data.index) == 0:
                        download_errors.append(f"No data for {s}")
                        continue
                    
                    data.to_csv(f'hist/{s}.csv')
                    is_valid[i] = True
                    print(f"Successfully downloaded {s} ({i-offset+1}/{end-offset})")
                    
                except Exception as e:
                    download_errors.append(f"Failed {s}: {str(e)}")
                    continue
    
    valid_data = data_clean[is_valid]
    valid_data.to_csv('symbols_valid_meta.csv', index=False)
    
    etfs = valid_data[valid_data['ETF'] == 'Y']['NASDAQ Symbol'].tolist()
    stocks = valid_data[valid_data['ETF'] == 'N']['NASDAQ Symbol'].tolist()
    
    def move_symbols(symbol_list, dest_folder):
        for s in symbol_list:
            try:
                src = join('hist', f'{s}.csv')
                if isfile(src):
                    shutil.move(src, join(dest_folder, f'{s}.csv'))
            except Exception as e:
                download_errors.append(f"File move failed for {s}: {str(e)}")
    
    move_symbols(etfs, "etfs")
    move_symbols(stocks, "stocks")
    
    try:
        os.rmdir('hist')
    except OSError:
        pass
    
    print(f"\nCompleted processing. Successfully downloaded {sum(is_valid)} symbols.")
    if download_errors:
        print("\nEncountered some errors (saved to download_errors.log):")
        with open('download_errors.log', 'w') as f:
            for error in download_errors:
                f.write(error + '\n')
        print(f"See {len(download_errors)} errors in download_errors.log")

download_nasdaq_data(offset=0, limit=None, period='5y')