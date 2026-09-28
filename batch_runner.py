import sys, os, gc, json, time
from datetime import datetime
import pandas as pd
import numpy as np
from backtest_core import load_symbol_timeframe_data, SMCExecutionEngine, get_strategy_config_and_funcs
from src.anti_overfit_suite import deflated_sharpe_ratio, haircut_sharpe

def run_batch(symbols, timeframes, strategies, engines, start_year=2021, end_year=2025, batch_id="batch1", use_credal=False):
    print(f"[{batch_id}] Starting run for symbols: {symbols}, timeframes: {timeframes}, Credal={use_credal}")
    t0 = time.time()
    
    os.makedirs(f"results/{batch_id}/trades", exist_ok=True)
    os.makedirs(f"results/{batch_id}/reports", exist_ok=True)
    
    summary_records = []
    
    for sym in symbols:
        for tf in timeframes:
            print(f"[{batch_id}] Loading data for {sym} {tf}...")
            try:
                df = load_symbol_timeframe_data(sym, tf)
            except Exception as e:
                print(f"[{batch_id}] ERROR loading {sym} {tf}: {e}")
                continue
                
            for strat in strategies:
                config, compute_ind, get_sig = get_strategy_config_and_funcs(strat)
                # Compute indicators on full continuous history to ensure complete warmup!
                data_full = compute_ind(df, config)
                data_full = get_sig(data_full, config)

                for year in range(start_year, end_year + 1):
                    # Extract year data with fully pre-warmed indicators
                    mask_year = data_full.index.year == year
                    data_year = data_full[mask_year]
                    if len(data_year) < 100:
                        continue
                        
                    split_idx = int(len(data_year) * 0.8)
                    data_is = data_year.iloc[:split_idx]
                    data_oos = data_year.iloc[split_idx:]
                    
                    for eng in engines:
                        engine_exec = SMCExecutionEngine(
                            engine_type=eng,
                            initial_cash=1000.0,
                            maker_fee=0.0002,
                            taker_fee=0.0005,
                            slippage=0.0005,
                            risk_pct=2.0,
                            use_credal=use_credal
                        )
                        
                        # Run In-Sample with pre-warmed indicators
                        res_is = engine_exec.run(data_is, strat)
                        # Run Out-of-Sample with pre-warmed indicators
                        res_oos = engine_exec.run(data_oos, strat)
                        
                        # Save trade logs
                        trade_prefix = f"results/{batch_id}/trades/{sym}_{tf}_{strat}_{eng}_{year}"
                        with open(f"{trade_prefix}_IS_trades.json", "w", encoding="utf-8") as fp:
                            json.dump(res_is["trades"], fp, indent=2)
                        with open(f"{trade_prefix}_OOS_trades.json", "w", encoding="utf-8") as fp:
                            json.dump(res_oos["trades"], fp, indent=2)
                            
                        m_is = res_is["metrics"]
                        m_oos = res_oos["metrics"]
                        
                        # Calculate Deflated Sharpe Ratio (DSR) & Haircut Sharpe (Harvey & Liu)
                        # N = 480 trials tested across the matrix
                        is_sr = m_is["sharpe_ratio"]
                        oos_sr = m_oos["sharpe_ratio"]
                        
                        try:
                            dsr_oos = deflated_sharpe_ratio(sr=max(oos_sr, -5.0), t=len(data_oos), n_trials=480).dsr
                            haircut_oos = haircut_sharpe(sr=max(oos_sr, 0.0), n_trials=480, t=len(data_oos)).haircut_sr
                        except Exception:
                            dsr_oos = 0.0
                            haircut_oos = 0.0
                        
                        rec = {
                            "symbol": sym,
                            "timeframe": tf,
                            "strategy": strat,
                            "engine": eng,
                            "year": year,
                            "is_bars": len(data_is),
                            "oos_bars": len(data_oos),
                            "is_return_pct": m_is["total_return_pct"],
                            "is_trades": m_is["total_trades"],
                            "is_win_rate_pct": m_is["win_rate_pct"],
                            "is_profit_factor": m_is["profit_factor"],
                            "is_max_dd_pct": m_is["max_drawdown_pct"],
                            "is_sharpe": m_is["sharpe_ratio"],
                            "is_pnl": m_is["total_pnl"],
                            "is_fees": m_is["total_fees"],
                            "is_slippage": m_is["total_slippage_cost"],
                            "oos_return_pct": m_oos["total_return_pct"],
                            "oos_trades": m_oos["total_trades"],
                            "oos_win_rate_pct": m_oos["win_rate_pct"],
                            "oos_profit_factor": m_oos["profit_factor"],
                            "oos_max_dd_pct": m_oos["max_drawdown_pct"],
                            "oos_sharpe": m_oos["sharpe_ratio"],
                            "oos_dsr": float(dsr_oos),
                            "oos_haircut_sharpe": float(haircut_oos),
                            "oos_pnl": m_oos["total_pnl"],
                            "oos_fees": m_oos["total_fees"],
                            "oos_slippage": m_oos["total_slippage_cost"],
                        }
                        summary_records.append(rec)
                
                del data_full
                gc.collect()
            
            del df
            gc.collect()
            
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(f"results/{batch_id}/summary.csv", index=False)
    print(f"[{batch_id}] Finished {len(summary_records)} backtest runs in {time.time()-t0:.2f}s!")

if __name__ == '__main__':
    batch_name = sys.argv[1]
    symbols = sys.argv[2].split(',')
    timeframes = sys.argv[3].split(',')
    strats = ['v1_internal_factors', 'v2_volatility_regime', 'v3_trend_strength']
    engines = ['QuantCell', 'NautilusTrader']
    use_cred = '--credal' in sys.argv
    run_batch(symbols, timeframes, strats, engines, start_year=2021, end_year=2025, batch_id=batch_name, use_credal=use_cred)
