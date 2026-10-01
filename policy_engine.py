"""
Trade Plan Generator & Entry State Evaluator.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

ENGINE_VERSION = "v11.0-policy"
HORIZONS = ["short", "mid", "long"]

def generate_trade_plan(df: pd.DataFrame, horizon: str) -> dict | None:
    if df.empty or len(df) < 20:
        return None
    
    p_close = float(df["Close"].iloc[-1])
    high_20 = float(df["High"].iloc[-20:].max())
    low_20 = float(df["Low"].iloc[-20:].min())
    atr = float((df["High"] - df["Low"]).iloc[-14:].mean()) if len(df) >= 14 else p_close * 0.03
    
    if horizon == "short":
        trigger = high_20 * 1.005
        zone_low = p_close * 0.98
        zone_high = p_close * 1.01
        chase_limit = p_close * 1.03
        invalidation = max(low_20, p_close - 2.0 * atr)
        entry_mode = "breakout_confirmed"
    elif horizon == "mid":
        trigger = high_20 * 1.01
        zone_low = p_close * 0.96
        zone_high = p_close * 1.015
        chase_limit = p_close * 1.04
        invalidation = p_close - 2.5 * atr
        entry_mode = "zone_confirmed"
    else:  # long
        trigger = high_20 * 1.02
        zone_low = p_close * 0.94
        zone_high = p_close * 1.02
        chase_limit = p_close * 1.05
        invalidation = p_close - 3.0 * atr
        entry_mode = "zone_confirmed"
        
    return {
        "trigger": round(trigger, 2),
        "zone_low": round(zone_low, 2),
        "zone_high": round(zone_high, 2),
        "chase_limit": round(chase_limit, 2),
        "invalidation": round(invalidation, 2),
        "entry_mode": entry_mode
    }

def evaluate_entry_state(df: pd.DataFrame, plan: dict | None) -> str:
    if not plan or df.empty:
        return "NO_RETURN_ESTIMATE"
    
    p_close = float(df["Close"].iloc[-1])
    if p_close <= plan["invalidation"]:
        return "INVALIDATED"
    elif p_close > plan["chase_limit"]:
        return "DO_NOT_CHASE"
    elif plan["zone_low"] <= p_close <= plan["zone_high"]:
        return "CONDITIONS_MET_NOT_FILLED"
    elif p_close < plan["zone_low"]:
        return "WAIT_ENTRY_ZONE"
    else:
        return "WAIT_BREAKOUT"