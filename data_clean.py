import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def build_master_dataset():
    print("🚀 Starting Data Engineering Pipeline...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    today = datetime.today().strftime('%Y-%m-%d')

    # ==========================================
    # 1. LOAD HISTORICAL DATA (The "Messy" CSVs)
    # ==========================================
    print("📊 Processing Historical CSVs...")
    
    # --- A. Sugar (1960-2022) ---
    comm_df = pd.read_csv('data/commodity_prices.csv', parse_dates=['date'])
    comm_df.set_index('date', inplace=True)
    # The historical CSV is in Dollars (0.06), but Yahoo is in Cents (6.00). Multiply by 100 to align.
    sugar_hist = comm_df[['sugar_world']].rename(columns={'sugar_world': 'Sugar_Nominal'}) * 100
    sugar_hist.index = sugar_hist.index.tz_localize(None)

    # --- B. Gold (1978-2020s) ---
    gold_csv = pd.read_csv('data/gold.csv')
    # Rename first column to Date and parse the European DD/MM/YYYY format
    gold_csv.rename(columns={gold_csv.columns[0]: 'Date'}, inplace=True)
    gold_csv['Date'] = pd.to_datetime(gold_csv['Date'], format='%d/%m/%Y', errors='coerce')
    gold_csv.set_index('Date', inplace=True)
    # Extract the first 'US dollar' column
    gold_hist = gold_csv.iloc[:, 0].to_frame(name='Gold_Nominal')
    gold_hist['Gold_Nominal'] = pd.to_numeric(gold_hist['Gold_Nominal'], errors='coerce')
    gold_hist.index = gold_hist.index.tz_localize(None)

    # --- C. Bitcoin (2010-2024) ---
    btc_hist = pd.read_csv('data/bitcoin.csv', parse_dates=['Date'], index_col='Date')
    btc_hist = btc_hist[['Close']].rename(columns={'Close': 'BTC_Nominal'})
    btc_hist.index = btc_hist.index.tz_localize(None)

    # ==========================================
    # 2. FETCH MODERN PATCHES (Filling the Gaps to 2026)
    # ==========================================
    print("🌐 Fetching live modern data to patch gaps (2020-2026)...")
    
    # We download from 2020 to ensure plenty of overlap for stitching
    tickers = {'SB=F': 'Sugar_Nominal', 'GC=F': 'Gold_Nominal', 'BTC-USD': 'BTC_Nominal'}
    modern_data = yf.download(list(tickers.keys()), start='2020-01-01', end=today)['Close']
    modern_data.rename(columns=tickers, inplace=True)
    modern_data.index = modern_data.index.tz_localize(None)

    # Download perfectly up-to-date CPI from the Federal Reserve (FRED)
    cpi_df = web.DataReader('CPIAUCSL', 'fred', start='1960-01-01', end=today)
    cpi_df.rename(columns={'CPIAUCSL': 'CPI'}, inplace=True)
    cpi_df.index = cpi_df.index.tz_localize(None)

    # ==========================================
    # 3. STANDARDIZE FREQUENCY (Downsample to Monthly)
    # ==========================================
    print("⏳ Standardizing all frequencies to Monthly...")
    
    # Resample everything to the last business day of the month ('ME' or 'M')
    sugar_hist_m = sugar_hist.resample('ME').last()
    gold_hist_m = gold_hist.resample('ME').last()
    btc_hist_m = btc_hist.resample('ME').last()
    modern_data_m = modern_data.resample('ME').last()
    cpi_m = cpi_df.resample('ME').last()

    # ==========================================
    # 4. STITCHING AND INFLATION ADJUSTMENT
    # ==========================================
    print("🪡 Stitching epochs together and applying CPI adjustment...")

    # Combine historical and modern data. 'combine_first' prioritizes the modern Yahoo data 
    # for recent years, but falls back to your historical CSVs for the deep past!
    master_sugar = modern_data_m[['Sugar_Nominal']].combine_first(sugar_hist_m)
    master_gold = modern_data_m[['Gold_Nominal']].combine_first(gold_hist_m)
    master_btc = modern_data_m[['BTC_Nominal']].combine_first(btc_hist_m)

    # Merge everything into one beautiful master table
    df_master = master_sugar.join([master_gold, master_btc, cpi_m], how='outer')
    
    # Clean up empty rows and forward-fill missing single months
    df_master.dropna(subset=['CPI', 'Sugar_Nominal'], how='all', inplace=True)
    df_master.ffill(inplace=True)

    # Calculate REAL Prices (Adjusted to Today's USD)
    cpi_today = df_master['CPI'].iloc[-1]
    for asset in ['Sugar', 'Gold', 'BTC']:
        df_master[f'{asset}_Real'] = df_master[f'{asset}_Nominal'] * (cpi_today / df_master['CPI'])

    # ==========================================
    # 5. EXPORT
    # ==========================================
    output_file = 'data/master_macro_dataset.csv'
    df_master.to_csv(output_file)
    
    print(f"✅ SUCCESS! Clean dataset saved to '{output_file}'")
    print(f"   -> Total Months: {len(df_master)}")
    print(f"   -> Date Range: {df_master.index.min().strftime('%Y-%m')} to {df_master.index.max().strftime('%Y-%m')}")
    
    return df_master

# Run the pipeline
if __name__ == "__main__":
    df = build_master_dataset()
    print(df.head())