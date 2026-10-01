"""
Taiwan Alpha Radar V11.0 Unified Research App.
Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import html
import json
import math
import os
import traceback
import gc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from market_data import DailyPriceStore, _taipei_timestamp, provider_runtime_status
from operational_tools import ScanBusyError, safe_error_text
from trading_calendar import calendar_reference, daily_freshness, entry_review_allowed, save_closure_notice
from input_validation import publish_inputs
from presentation import plain_summary
from policy_engine import ENGINE_VERSION, HORIZONS
from return_first_model import holding_review, finite_scalar, ModelDataError, timestamp

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))
VIEW_LABELS = ["⭐ 極選", "⚡ 短線", "📈 中線", "🧭 長波段", "🔎 個股"]
HORIZON_LABELS = {"short": "短線 · 10交易日", "mid": "中線 · 40交易日", "long": "長波段 · 120交易日"}
FAMILY_LABELS = {
    "價量研究｜無財報／法人代理": "price_only",
    "基本面確認｜需要歷史公告資料": "business_confirmed",
    "法人確認｜需要歷史籌碼資料": "flow_confirmed",
    "完整證據｜基本面＋法人": "full",
}
SETUP_LABELS = {
    "BREAKOUT": "平台突破", "PULLBACK": "趨勢回測", "RECLAIM": "重新站回",
    "TREND": "趨勢延續觀察", "BASE": "整理觀察", "DRYUP": "低量新低觀察",
}
STATE_LABELS = {
    "CONDITIONS_MET_NOT_FILLED": "收盤條件符合，次日開盤再確認",
    "WAIT_ENTRY_ZONE": "等待回到布局區",
    "WAIT_BREAKOUT": "等待突破確認",
    "WAIT_CONFIRMATION": "量價確認尚未成立",
    "DO_NOT_CHASE": "超出追價上限，等待新機會",
    "INVALIDATED": "原結構已失效",
    "DATA_UNVERIFIED": "資料待核對",
    "NO_RETURN_ESTIMATE": "沒有可用的報酬估計",
}
HOLD_LABELS = {
    "DATA_UNVERIFIED": "價格資料未核對，暫不作留／賣判斷",
    "ORIGINAL_STRUCTURE_INVALIDATED": "原始結構失效：優先檢視退出與可成交條件",
    "ORIGINAL_THESIS_INVALIDATED": "原始買進理由失效：應重新檢視持有決策",
    "PROTECTION_TRIGGER_REVIEW_EXECUTION": "已觸及保護價：核對報價及實際執行條件",
    "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW": "缺少原始失效點或持有理由，無法完整判斷",
    "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "原規則尚未被破壞；不是續抱收益保證",
}

CSS = """
<style>
:root{--ink:#17243b;--muted:#66748b;--line:#e5eaf2;--blue:#1d4ed8;}
.stApp{background:#f5f7fb;color:var(--ink)}
.block-container{max-width:1020px;padding-top:1.1rem;padding-bottom:4rem;}
.hero{padding:24px;border-radius:24px;background:linear-gradient(125deg,#102139,#173c70 65%,#126975);color:white;margin-bottom:18px;}
.hero .eyebrow{letter-spacing:.12em;font-size:.75rem;opacity:.8}
.hero h1{font-size:1.9rem;line-height:1.25;color:white;margin:.35rem 0;}
.hero p{font-size:1rem;opacity:.85;margin:.45rem 0 0;line-height:1.6}
.topnote{padding:14px 17px;border:1px solid #d7e1f3;background:#eef4ff;border-radius:14px;margin:12px 0;color:#334a70;font-size:.94rem;line-height:1.7}
.statusline{font-size:.89rem;color:#556580;line-height:1.65;margin:8px 0 14px}
.card{background:white;border:1px solid var(--line);border-radius:21px;padding:22px;margin:14px 0 10px;box-shadow:0 6px 24px rgba(19,42,80,.04)}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;}
.card-title{font-size:1.3rem;font-weight:850;line-height:1.4}
.card-code{font-size:.83rem;color:var(--muted);margin-top:4px}
.card-price{font-size:1.22rem;font-weight:800;text-align:right;white-space:nowrap}
.badge{display:inline-block;font-size:.78rem;font-weight:700;padding:5px 9px;border-radius:8px;background:#f0f3f8;color:#52617a;margin:10px 5px 7px 0}
.badge-blue{color:#234aaa;background:#eaf0ff}
.return-box{background:#f6f9ff;border:1px solid #e4ebf9;border-radius:15px;padding:15px;margin:12px 0;}
.return-k{font-size:.9rem;color:#4c5c75}.return-v{font-size:2.15rem;font-weight:900;letter-spacing:-.04em;line-height:1.4;color:#1e46a3;}
.return-desc{font-size:.82rem;color:#708098;line-height:1.5}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:12px 0;}
.stat{border:1px solid #edf0f6;background:#fcfdff;padding:12px 10px;border-radius:12px;}
.stat .k{font-size:.77rem;color:#6a7890;line-height:1.4}.stat .v{font-size:1.1rem;font-weight:800;margin-top:5px;}
.decision{padding:13px 15px;border-radius:12px;background:#fff5df;border:1px solid #f4dfb1;color:#88621a;font-weight:750;font-size:.99rem;line-height:1.6}
.decision-ok{background:#eaf6f0;border-color:#d1e9dd;color:#225e44}
.evidence{font-size:.87rem;line-height:1.7;color:#69758c;margin-top:13px}
.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;}
.level{background:#f8fafc;border:1px solid #edf0f5;border-radius:11px;padding:10px}
.level .k{font-size:.8rem;color:#62718a}.level .v{font-weight:800;font-size:1rem;margin-top:5px;}
.subnote{font-size:.85rem;color:#758198;line-height:1.65;margin:10px 0}
.stButton>button,.stFormSubmitButton>button{border-radius:12px;min-height:48px;font-weight:750;}
div[data-testid="stRadio"] div[role="radiogroup"]{gap:.4rem;flex-wrap:wrap}
div[data-testid="stRadio"] label{background:white;border:1px solid #e3e8f2;border-radius:10px;padding:9px 13px}
@media(max-width:650px){
 .block-container{padding:.8rem .8rem 3rem;}.hero{padding:19px 17px;border-radius:19px}.hero h1{font-size:1.5rem}.hero p{font-size:.9rem}
 .card{padding:17px 14px;border-radius:17px}.card-title{font-size:1.15rem}.card-price{font-size:1.13rem}.return-v{font-size:1.92rem}
 .stats{grid-template-columns:repeat(2,1fr)}.levels{grid-template-columns:repeat(2,1fr)}
 .stat .v{font-size:1.12rem}.stRadio label{font-size:.87rem}
}
</style>
"""

def esc(value): return html.escape(str(value))

def percent(value, signed=True):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v*100:{'+' if signed else ''}.2f}%"

def money(value):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v:,.2f}".rstrip("0").rstrip(".")

def render_chart(chart, plan, key):
    if not chart:
        st.caption("這檔 K 線未常駐記憶體；診斷時將動態讀取。")
        return
    d = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    dates = chart["dates"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.045, row_heights=[.74, .26])
    fig.add_trace(go.Candlestick(x=dates, open=d.Open, high=d.High, low=d.Low, close=d.Close,
                                increasing_line_color="#d85053", decreasing_line_color="#22917e",
                                name="原始價格"), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=d.Volume, name="成交量",
                        marker_color=np.where(d.Close >= d.Open, "#d85053", "#22917e")), row=2, col=1)
    if plan:
        fig.add_hrect(y0=plan.get("zone_low", 0), y1=plan.get("zone_high", 0), line_width=0,
                      fillcolor="rgba(43,102,210,.10)", row=1, col=1)
        fig.add_hline(y=plan.get("trigger", 0), line_color="#ca902b", line_dash="dot", row=1, col=1)
        fig.add_hline(y=plan.get("invalidation", 0), line_color="#ba5360", line_dash="dash", row=1, col=1)
    fig.update_layout(height=410, margin=dict(l=7, r=8, t=10, b=8), showlegend=False,
                      xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type="category", nticks=5, fixedrange=True, showgrid=False)
    fig.update_yaxes(fixedrange=True, gridcolor="#e9edf4")
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False})

def card(obj, h, snap, view, chart=None, calendar=None):
    block = obj["horizons"][h]
    f = block.get("forecast") or {}
    plan = block.get("plan")
    summary = f.get("strategy") or {}
    condition = block.get("entry_state", "NO_RETURN_ESTIMATE")
    state_label = STATE_LABELS.get(condition, condition)
    
    headline = f"TOP 多因子強勢推薦｜{state_label}"
    calendar = calendar or calendar_reference(DATA_DIR, now=_taipei_timestamp())
    fresh = daily_freshness(obj["price_date"], calendar)
    review_ok = entry_review_allowed(obj["price_date"], plan, calendar)
    
    if not fresh["current_daily"]:
        headline = "請更新日線行情後評估"
    
    ev = percent(summary.get("mean"))
    factor_score = f.get("composite_factor_score", 70.0)
    conf_score = f.get("confidence_score", 75.0)
    summary_sentence = plain_summary(f)
    setup = SETUP_LABELS.get(obj.get("setup"), obj.get("setup", ""))
    
    st.markdown(f"""
<div class="card">
 <div class="card-head"><div><div class="card-title">{esc(obj.get('name', obj['ticker']))}</div>
 <div class="card-code">{esc(obj['ticker'])} · {esc(obj.get('industry',''))}</div></div>
 <div class="card-price">{money(obj['price'])}<div class="card-code">{esc(obj['price_date'])} 日線</div></div></div>
 <span class="badge badge-blue">{esc(HORIZON_LABELS[h])}</span><span class="badge">{esc(setup)}</span><span class="badge">多因子得分：{factor_score:.1f} 分</span>
 <div class="topnote">{esc(summary_sentence)}</div>
 <div class="return-box"><div class="return-k">策略預期淨報酬 (EV)</div><div class="return-v">{ev}</div>
 <div class="return-desc">多因子綜合攻擊評分：<b>{factor_score:.1f} 分</b>｜模型信心度：<b>{conf_score:.1f}%</b></div></div>
 <div class="stats">
  <div class="stat"><div class="k">大盤 Alpha</div><div class="v">{percent(f.get('alpha_mean'))}</div></div>
  <div class="stat"><div class="k">中間情境 (P50)</div><div class="v">{percent(summary.get('median'))}</div></div>
  <div class="stat"><div class="k">偏佳情境 (P75)</div><div class="v">{percent(summary.get('p75'))}</div></div>
  <div class="stat"><div class="k">最差10%損失</div><div class="v">{percent(summary.get('expected_shortfall10_loss'), False)}</div></div>
 </div>
 <div class="decision {'decision-ok' if review_ok and condition=='CONDITIONS_MET_NOT_FILLED' else ''}">{esc(headline)}</div>
</div>""", unsafe_allow_html=True)
    
    with st.expander("進出場關鍵價位與 K 線", expanded=False):
        if plan:
            st.markdown(f"""<div class="levels">
<div class="level"><div class="k">布局區</div><div class="v">{money(plan.get('zone_low'))}–{money(plan.get('zone_high'))}</div></div>
<div class="level"><div class="k">突破價</div><div class="v">{money(plan.get('trigger'))}</div></div>
<div class="level"><div class="k">不追價上限</div><div class="v">{money(plan.get('chase_limit'))}</div></div>
<div class="level"><div class="k">結構失效</div><div class="v">{money(plan.get('invalidation'))}</div></div>
</div>""", unsafe_allow_html=True)
        chart = chart or snap.get("charts", {}).get(obj["ticker"])
        if chart is None:
            try: chart = service.chart_on_demand(snap, obj["ticker"], DATA_DIR, allow_fetch=False)
            except Exception: pass
        render_chart(chart, plan, f"chart_{view}_{h}_{obj['ticker']}_{snap['snapshot_id']}")

def render_horizon(snap, h, calendar=None):
    st.subheader(HORIZON_LABELS[h])
    picked = service.select_market_best(snap, h, n=3)
    if picked:
        st.caption(f"依據多因子量化矩陣（RS大盤強度＋多頭結構＋攻擊量）為您推薦 TOP {len(picked)} 精選標的：")
        for obj in picked:
            card(obj, h, snap, h, calendar=calendar)
    else:
        st.info("目前市場環境下無滿足條件之標的。")

def main():
    st.set_page_config(page_title="Alpha Radar · Unified V11.0", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero"><div class="eyebrow">TAIWAN ALPHA RADAR · V11.0 UNIFIED</div>
<h1>全台股收益導向量化選股與個股診斷</h1>
<p>2,000+ 檔上市櫃 Open Data 動態母池 × 多因子綜合動能打分 × 嚴格風控診斷</p></div>""", unsafe_allow_html=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if st.session_state.get("v8_version") != service.OPERATIONS_VERSION:
        for key in ("v8_snapshot", "v8_doctor", "v8_error"):
            st.session_state.pop(key, None)
        st.session_state["v8_version"] = service.OPERATIONS_VERSION
        try:
            previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json", include_features=False)
            if previous:
                st.session_state["v8_snapshot"] = service.compact_session_dashboard(previous)
        except Exception: pass

    with st.sidebar:
        st.markdown("### 資料與研究設定")
        family_label = st.selectbox("報酬模型的資料範圍", list(FAMILY_LABELS), key="family_v11")
        refs = st.selectbox("歷史參考股票數", [160, 300, 600], key="reference_v11")
        period = st.selectbox("歷史研究長度", ["5y", "8y", "10y", "3y"], key="period_v11")
        
        with st.expander("維護與快取", expanded=False):
            if st.button("強制清除行情與舊快照", key="clear_prices_v11"):
                DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
                service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
                st.session_state.pop("v8_snapshot", None)
                st.success("行情與舊快照已完全重置！")

    settings = service.RunSettings(
        reference_size=int(refs), candidate_size=300, history_period=period,
        model_family=FAMILY_LABELS[family_label]
    )

    if st.button("⚡ 更新市場與報酬研究（即時連線全台股 Open Data）", type="primary", use_container_width=True, key="run_scan_v11"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1., max(0., value)), text="連線抓取盤面與計算多因子：" + stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            compact = service.compact_session_dashboard(snap)
            st.session_state["v8_snapshot"] = compact
            st.session_state.pop("v8_doctor", None)
            st.session_state.pop("v8_error", None)
            del snap, compact
            gc.collect()
        except Exception as exc:
            st.session_state["v8_error"] = f"{type(exc).__name__}: {exc}"
            st.error("掃描未完成，已保留上一份成功快照。")
        finally:
            progress.empty()

    snap = st.session_state.get("v8_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())

    if snap and isinstance(snap, dict):
        st.markdown(f"""<div class="statusline">日線截至 <b>{esc(snap.get('price_date',''))}</b> · 即時母池 {snap.get('coverage',{}).get('requested',0):,} 檔 · 深度評估 {snap.get('candidate_n',0):,} 檔</div>""", unsafe_allow_html=True)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v11")

    if view == VIEW_LABELS[0]:
        st.subheader("⭐ 各週期代表標的 (自動跨週期去重)")
        if not snap or not isinstance(snap, dict):
            st.info("尚無收益快照，請點擊上方『⚡ 更新市場與報酬研究』進行連線掃描。")
        else:
            used_tickers = []
            for h in HORIZONS:
                picks = service.select_market_best(snap, h, n=3)
                valid_picks = [p for p in picks if p["ticker"] not in used_tickers]
                if valid_picks:
                    obj = valid_picks[0]
                    used_tickers.append(obj["ticker"])
                    card(obj, h, snap, "prime", calendar=calendar)
    elif view == VIEW_LABELS[4]:
        st.subheader("🔎 個股診斷與持股檢視")
        with st.form("doctor_form_v11"):
            code = st.text_input("股票代碼", value="2330", key="doctor_code_v11")
            horizon_text = st.selectbox("研究週期", list(HORIZON_LABELS.values()), index=1, key="doctor_h_v11")
            own = st.checkbox("已有持股", key="doctor_own_v11")
            c1, c2 = st.columns(2)
            with c1:
                cost = st.number_input("持股成本（選填）", min_value=0., value=0., key="doctor_cost_v11")
                invalid = st.number_input("原始結構失效價（未知填0）", min_value=0., value=0., key="doctor_stop_v11")
            with c2:
                shares = st.number_input("股數（選填）", min_value=0, value=0, step=100, key="doctor_qty_v11")
                trail = st.number_input("既有獲利保護價（未知填0）", min_value=0., value=0., key="doctor_trail_v11")
            thesis = st.selectbox("原買進理由是否仍成立？", ["尚未確認", "仍成立", "已不成立"], key="doctor_thesis_v11")
            submitted = st.form_submit_button("立即診斷", type="primary", use_container_width=True)
            
        if submitted and snap:
            dr = service.diagnose(code, snap, DATA_DIR)
            dr["h"] = next(k for k, v in HORIZON_LABELS.items() if v == horizon_text)
            st.session_state["v8_doctor"] = service.compact_doctor_result(dr)
            
        dr = st.session_state.get("v8_doctor")
        if dr and snap and "stock" in dr:
            card(dr["stock"], dr.get("h", "mid"), snap, "doctor", dr.get("chart"), calendar=calendar)
            if own:
                result = holding_review(dr["stock"]["price"], original_invalidation=invalid, trailing_protection=trail, thesis_broken=(thesis == "已不成立"))
                st.markdown("#### 既有持股原則檢查")
                st.info(HOLD_LABELS.get(result, result))
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)

if __name__ == "__main__":
    main()