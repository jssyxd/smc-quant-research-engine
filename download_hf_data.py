import urllib.request, zipfile, io, os, gc

SYMBOLS = {
    'BTC': 'BTCUSDTraw_data.zip',
    'ETH': 'ETHUSDTraw_data.zip',
    'SOL': 'SOLUSDTraw_data.zip',
    'BNB': 'BNBUSDTraw_data.zip'
}

base_url = 'https://github.com/JasonleeQAQ/multi-asset-ohlcv/releases/download/1.0.0/'
os.makedirs('data/raw_extracted', exist_ok=True)

for sym, zname in SYMBOLS.items():
    dest_dir = f'data/raw_extracted/{sym}'
    os.makedirs(dest_dir, exist_ok=True)
    existing_5m = [f for f in os.listdir(dest_dir) if '_5m_' in f]
    if len(existing_5m) >= 40:
        print(f"{sym} already has {len(existing_5m)} 5m files, skipping download.")
        continue
    print(f"Downloading 5m bars for {sym} from {zname}...")
    url = base_url + zname
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        # target 5m files for 2021-2025
        target_files = [f for f in zf.namelist() if '/5m/' in f or '_5m_' in f]
        # filter for recent years 2021-2025 to save memory/disk
        target_files = [f for f in target_files if any(str(y) in f for y in range(2021, 2026))]
        print(f"  Extracting {len(target_files)} 5m parquet files for {sym}...")
        for tf in target_files:
            fname = os.path.basename(tf)
            if not fname:
                continue
            with open(os.path.join(dest_dir, fname), 'wb') as fp:
                fp.write(zf.read(tf))
    gc.collect()
    print(f"  Finished {sym} successfully.")
