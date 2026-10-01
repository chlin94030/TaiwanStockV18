"""
Text formatting and plain summary generators.
"""
from __future__ import annotations

def plain_summary(forecast: dict) -> str:
    if not forecast or not forecast.get("estimate_available"):
        return "數據或歷史對照樣本不足，建議謹慎觀察。"
    
    score = forecast.get("composite_factor_score", 0.0)
    ev = forecast.get("strategy", {}).get("mean", 0.0)
    alpha = forecast.get("alpha_mean", 0.0)
    
    if score >= 70.0 and ev > 0.02:
        return f"多因子強勢指標（{score:.1f}分），具備顯著 Alpha 超額收益 ({alpha*100:+.2f}%)，多頭排列結構完整。"
    elif ev > 0:
        return f"多因子動能評分（{score:.1f}分），策略淨預期報酬 ({ev*100:+.2f}%) 轉正，適合跟隨趨勢布局。"
    else:
        return f"當前多因子得分（{score:.1f}分），大盤相對動能偏弱，建議等待回測布局區。"