"""
均線扣抵值分析工具 (MA Roll-off Projection)
台股 / FinMind
v5 — 新增批次摘要層（乖離率 / 收斂期數 / 確定上彎 / 箱體寬度 + 文字解讀）
"""

import io
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="均線扣抵值分析", page_icon="📉", layout="wide")

# ---------------- config ----------------
MA_OPTIONS = {
    "日線 MA20": ("D", 20),
    "日線 MA60": ("D", 60),
    "日線 MA120": ("D", 120),
    "日線 MA200": ("D", 200),
    "週線 W20": ("W", 20),
    "週線 W30": ("W", 30),
    "週線 W60": ("W", 60),
}


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def _token() -> str:
    return st.secrets.get("FINMIND_TOKEN", "").strip()


def _finmind(dataset: str, **params) -> pd.DataFrame:
    payload = {"dataset": dataset, "token": _token(), **params}
    r = requests.get(FINMIND_URL, params=payload, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != 200 and js.get("msg") not in (None, "success"):
        raise RuntimeError(js.get("msg", "FinMind error"))
    return pd.DataFrame(js.get("data", []))


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def fetch_daily(stock_id: str, years: int = 5) -> pd.DataFrame:
    start = (dt.date.today() - dt.timedelta(days=365 * years)).isoformat()
    df = _finmind("TaiwanStockPrice", data_id=stock_id, start_date=start)
    if df.empty:
        return pd.DataFrame()
    df = df[["date", "open", "max", "min", "close", "Trading_Volume"]].copy()
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["close"] > 0].sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def stock_name_map() -> dict:
    try:
        info = _finmind("TaiwanStockInfo")
        return dict(zip(info["stock_id"], info["stock_name"]))
    except Exception:
        return {}


TWSE_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def top_liquid_stocks(n: int = 120, include_etf: bool = False) -> list:
    """Rank 上市 stocks by latest-session turnover via TWSE open API (1 request)."""
    r = requests.get(TWSE_ALL_URL, timeout=30, headers={"accept": "application/json"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return []
    for col in ("TradeValue", "ClosingPrice"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    df = df.dropna(subset=["TradeValue", "ClosingPrice"])
    df = df[df["ClosingPrice"] > 0]
    mask = df["Code"].str.match(r"^\d{4,6}$") if include_etf else df["Code"].str.match(r"^[1-9]\d{3}$")
    df = df[mask]
    return df.sort_values("TradeValue", ascending=False).head(n)["Code"].tolist()


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = (
        df.set_index("date")
        .resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    return w


def analyse(bars: pd.DataFrame, period: int, horizon: int):
    """Return (table_df, ma_series, projected_ma) for the roll-off analysis."""
    close = bars["close"].to_numpy(dtype=float)
    dates = bars["date"]
    n = len(close)
    if n < period + 2:
        return None, None, None

    ma = pd.Series(close).rolling(period).mean()
    cur_price = close[-1]
    cur_ma = ma.iloc[-1]

    steps = min(horizon, period - 1)
    rows = []
    proj_window = list(close[-period:])  # window backing today's MA
    proj_ma = []
    for k in range(1, steps + 1):
        idx = n - period + k - 1          # bar that rolls off at step k
        roll_off = close[idx]
        roll_date = dates.iloc[idx].date()
        # project: assume price stays flat at current price
        proj_window.pop(0)
        proj_window.append(cur_price)
        pm = float(np.mean(proj_window))
        proj_ma.append(pm)
        gap = (cur_price - roll_off) / roll_off * 100
        rows.append({
            "步數": k,
            "扣抵日期": roll_date,
            "扣抵值": round(roll_off, 2),
            "現價": round(cur_price, 2),
            "現價-扣抵值 %": round(gap, 2),
            "均線方向": "🟢 上彎" if cur_price > roll_off else ("🔴 下彎" if cur_price < roll_off else "⚪ 持平"),
            "推估均線": round(pm, 2),
        })

    tbl = pd.DataFrame(rows)
    return tbl, ma, proj_ma


def verdict(tbl: pd.DataFrame, cur_price: float, cur_ma: float) -> str:
    up = (tbl["均線方向"] == "🟢 上彎").sum()
    total = len(tbl)
    stance = "站上" if cur_price > cur_ma else "跌破"
    # first flip point
    dirs = tbl["均線方向"].tolist()
    flip = None
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[0]:
            flip = i + 1
            break
    msg = f"現價 {cur_price:.2f}，{stance}均線 {cur_ma:.2f}。未來 {total} 期中 {up} 期扣抵偏低（均線上彎），{total - up} 期扣抵偏高（下彎壓力）。"
    if flip:
        msg += f" 第 {flip} 期起方向可能改變（扣抵值轉{'高' if dirs[0] == '🟢 上彎' else '低'}），屆時需觀察股價是否跟上。"
    else:
        msg += " 整段期間方向一致，趨勢延續性較高。"
    return msg


# ---------------- v5: 摘要層 ----------------

def summarize(bars: pd.DataFrame, tbl: pd.DataFrame, ma_series,
              period: int, look: int = 12, tol: float = 0.05) -> dict:
    """把單檔壓成可排序的純數字（float / bool，不含格式字串）。"""
    close = bars["close"].to_numpy(dtype=float)
    price = float(close[-1])
    ma_now = float(ma_series.iloc[-1])

    # 箱體：近 look 期的真實最高 / 最低（不是收盤價）
    look = min(look, len(bars))
    box_hi = float(bars["high"].iloc[-look:].max())
    box_lo = float(bars["low"].iloc[-look:].min())

    # 收斂期數：價格持平下，均線幾期後把乖離壓到 tol 以內 = 攤牌時間點
    # 兩個試過的爛定義，留著當警惕：
    #   「均線追上現價」→ 數學上恆等於 period（視窗填滿才相等），沒鑑別度
    #   「均線升到箱底」→ 實測 94% 的股票都是 1（均線本來就高於近期低點）
    target = price * (1 - tol)
    if ma_now >= target:
        conv = 0.0                      # 已經貼住了
    else:
        c, win = None, list(close[-period:])
        for k in range(1, period + 1):
            win.pop(0)
            win.append(price)
            if sum(win) / period >= target:
                c = k
                break
        conv = float(c) if c else float(period)

    return {
        "現價": round(price, 2),
        "均線": round(ma_now, 2),
        "乖離率": price / ma_now - 1,
        "收斂期數": conv,
        "確定上彎": bool(tbl["均線方向"].str.contains("上彎").all()),
        "箱體寬度": (box_hi - box_lo) / box_lo,
        "箱頂": round(box_hi, 2),
        "箱底": round(box_lo, 2),
    }


def explain(r) -> str:
    """把數字翻成中文。門檻值可自行調整。"""
    p = []

    d = r["乖離率"]
    if d > 0.15:
        p.append(f"乖離 +{d:.1%}（離均線太遠，追高風險大）")
    elif d > 0.05:
        p.append(f"乖離 +{d:.1%}（健康，有緩衝空間）")
    elif d > 0:
        p.append(f"乖離 +{d:.1%}（貼著均線，隨時可能跌破）")
    else:
        p.append(f"乖離 {d:.1%}（已在均線下方，空方格局）")

    w = r["收斂期數"]
    if w == 0:
        p.append("均線已經貼到腳下（乖離 5% 內，隨時攤牌）")
    elif w > 12:
        p.append(f"均線要 {w:.0f} 期才追到 5% 以內（乖離太大，還早）")
    elif w >= 6:
        p.append(f"均線約 {w:.0f} 期後追到 5% 以內（攤牌時間點）")
    else:
        p.append(f"均線 {w:.0f} 期內就貼上來（迫在眉睫）")

    p.append("均線確定上彎 ✅（未來這段支撐只會越墊越高）"
             if r["確定上彎"] else
             "均線可能翻下 ⚠️（有扣抵值高於現價，支撐會鬆動）")

    b = r["箱體寬度"]
    if b < 0.08:
        p.append(f"箱體 {b:.1%}（極窄，能量壓縮到極致，快突破了）")
    elif b < 0.15:
        p.append(f"箱體 {b:.1%}（收斂中，方向未明）")
    else:
        p.append(f"箱體 {b:.1%}（還在大幅震盪，沒整理完）")

    p.append(f"箱頂 {r['箱頂']} / 箱底 {r['箱底']}（站上箱頂才叫真突破）")

    if r["確定上彎"] and b < 0.10 and w < 8:
        p.append("→ 【重點觀察】均線快追上、價格又縮緊，最接近攤牌")
    elif r["確定上彎"] and d > 0.15:
        p.append("→ 【等回檔】方向對但位置太高，等乖離縮小")
    elif not r["確定上彎"]:
        p.append("→ 【避開】結構在轉弱")
    else:
        p.append("→ 【觀望】還沒到決勝點")

    return "｜".join(p)


# ---------------- UI ----------------
st.title("📉 均線扣抵值分析工具")
st.caption("扣抵值 = 未來即將被移出均線計算的舊價格。現價 > 扣抵值 → 均線上彎；現價 < 扣抵值 → 均線下彎。")

with st.sidebar:
    st.header("設定")
    mode = st.radio("標的來源", ["手動輸入", "熱門排行（自動）"], horizontal=True)
    if mode == "手動輸入":
        tickers_raw = st.text_area(
            "股票代號（逗號或換行分隔）",
            value="2330\n2317\n2454",
            height=140,
        )
        top_n, include_etf = 0, False
    else:
        tickers_raw = ""
        top_n = st.slider("取前 N 檔（上市當日成交值排行）", 20, 200, 120, step=10)
        include_etf = st.checkbox("含 ETF / 非四碼", value=False)
        st.caption("每檔各一次 API 呼叫，120 檔約需 1-2 分鐘，並可能觸及 FinMind 每小時額度。")
    ma_label = st.selectbox("均線", list(MA_OPTIONS.keys()), index=5)
    horizon = st.slider("往後推估期數", 4, 60, 12)
    show_charts = st.checkbox("顯示個股圖表", value=True)
    st.caption("超過 20 檔時自動關閉圖表，只輸出摘要，避免瀏覽器卡死。")

    st.divider()
    st.subheader("摘要篩選")
    st.caption("三個條件同時成立才會列入摘要：趨勢向上 + 價格縮在一起 + 快要見真章。")

    f_up = st.checkbox(
        "只看均線確定上彎", value=False,
        help="未來 12 期的扣抵值全部低於現價 → 均線這段期間不可能翻下。"
             "這是最嚴的一條，一個扣抵值高於現價就出局，勾了通常只剩個位數檔。",
    )
    f_box = st.slider(
        "箱體寬度上限", 0.05, 0.60, 0.25, 0.01,
        help="近 12 期真實最高與最低的差距。越小代表價格越收縮、能量壓得越緊。"
             "台股週線 15% 以內算很嚴，25% 起跳比較抓得到東西。",
    )
    f_conv = st.slider(
        "收斂期數上限", 0, 60, 20,
        help="價格持平下，均線還要幾期才把乖離壓到 5% 以內 = 攤牌時間點。"
             "0 代表現在就已經貼住。週線 12 期約三個月，20 期約五個月。"
             "數字越小越迫在眉睫；乖離越大的股票這個數字越大。",
    )
    st.caption("篩選只影響上方摘要表，下方個股區仍會列出全部。抓到 10~20 檔比較合理；"
               "如果只剩個位數就放寬，太多就收緊。")

    run = st.button("開始分析", type="primary", use_container_width=True)

if not run:
    st.info("左側輸入代號後按「開始分析」。週線模式下 1 期 = 1 週，日線模式 1 期 = 1 個交易日。")
    st.stop()

freq, period = MA_OPTIONS[ma_label]
if mode == "手動輸入":
    tickers = [t.strip() for t in tickers_raw.replace(",", "\n").replace("，", "\n").split("\n") if t.strip()]
else:
    with st.spinner("取得成交值排行..."):
        try:
            tickers = top_liquid_stocks(top_n, include_etf)
        except Exception as e:
            st.error(f"排行取得失敗：{e}")
            st.stop()
    st.success(f"已載入前 {len(tickers)} 檔：{', '.join(tickers[:15])}{' ...' if len(tickers) > 15 else ''}")
tickers = list(dict.fromkeys(tickers))

if not tickers:
    st.error("沒有輸入任何代號。")
    st.stop()

# 檔數多時強制關圖，否則 120 張 plotly 會把瀏覽器打死
detail = show_charts and len(tickers) <= 20

names = stock_name_map()
all_tables = []
summaries = []
summary_slot = st.container()   # 佔位：摘要表最後才填，但顯示在最上面
detail_slot = st.container()
progress = st.progress(0.0)

for i, tk in enumerate(tickers, 1):
    progress.progress(i / len(tickers), text=f"處理 {tk} ({i}/{len(tickers)})")
    try:
        daily = fetch_daily(tk, years=5 if freq == "W" else 2)
    except Exception as e:
        st.error(f"{tk}：抓取失敗 — {e}")
        continue
    if daily.empty:
        st.error(f"{tk}：查無資料")
        continue

    bars = to_weekly(daily) if freq == "W" else daily
    tbl, ma, proj = analyse(bars, period, horizon)
    if tbl is None:
        st.warning(f"{tk}：歷史資料不足以計算 {ma_label}")
        continue

    name = names.get(tk, "")
    cur_price = float(bars["close"].iloc[-1])
    cur_ma = float(ma.iloc[-1])

    # ── 摘要（不論是否顯示細節都算）──
    try:
        summaries.append({"代號": tk, "名稱": name,
                          **summarize(bars, tbl, ma, period, look=12)})
    except Exception as e:
        st.warning(f"{tk}：摘要計算失敗 — {e}")

    # ── CSV 明細（不論是否顯示細節都收）──
    out = tbl.copy()
    out.insert(0, "股票名稱", name)
    out.insert(0, "股票代號", tk)
    out.insert(2, "均線", ma_label)
    all_tables.append(out)

    if not detail:
        continue

    with detail_slot:
        st.subheader(f"{tk} {name} — {ma_label}")
        st.write(verdict(tbl, cur_price, cur_ma))

        hist_n = min(len(bars), period * 4)
        h = bars.tail(hist_n)
        hma = ma.tail(hist_n)
        step = pd.Timedelta(days=7) if freq == "W" else pd.Timedelta(days=1)
        future_dates = [bars["date"].iloc[-1] + step * k for k in range(1, len(proj) + 1)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["date"], y=h["close"], name="收盤價", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=h["date"], y=hma, name=ma_label, line=dict(width=2)))
        fig.add_trace(go.Scatter(x=future_dates, y=proj, name="推估均線(價格持平)",
                                 line=dict(width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=tbl["扣抵日期"], y=tbl["扣抵值"], name="扣抵值(來源價)",
                                 mode="markers", marker=dict(size=6, symbol="x")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # paste-to-AI block
        recent = bars.tail(12)[["date", "close"]]
        lines = [
            f"[{tk} {name}] {ma_label}",
            f"現價 {cur_price:.2f} / 現在均線 {cur_ma:.2f} / {'站上' if cur_price > cur_ma else '跌破'}",
            "近12期收盤: " + ", ".join(
                f"{d.date()}={c:.2f}" for d, c in zip(recent['date'], recent['close'])
            ),
            "扣抵值表 (步數|扣抵日|扣抵值|方向|推估均線):",
        ]
        for _, r in tbl.iterrows():
            lines.append(
                f"  {r['步數']}|{r['扣抵日期']}|{r['扣抵值']}|{r['均線方向'][-2:]}|{r['推估均線']}"
            )
        st.text_area("📋 複製給 AI 分析", value="\n".join(lines), height=160, key=f"ai_{tk}")
        st.divider()

progress.empty()

# ---------------- 摘要表（填回最上面的佔位）----------------
if summaries:
    sdf = pd.DataFrame(summaries)
    sdf["解讀"] = sdf.apply(explain, axis=1)

    m = (sdf["箱體寬度"] <= f_box) & (sdf["收斂期數"] <= f_conv)
    if f_up:
        m &= sdf["確定上彎"]
    hit = sdf[m].sort_values(["箱體寬度", "收斂期數"])

    with summary_slot:
        st.subheader(f"📊 摘要 — 符合條件 {len(hit)} / {len(sdf)} 檔")

        cond = []
        if f_up:
            cond.append("**確定上彎 ✅** — 未來 12 期扣抵值全部低於現價 → 均線這段期間不可能翻下")
        cond.append(f"**箱體寬度 ≤ {f_box:.0%}** — 近 12 期高低差在 {f_box:.0%} 以內 → 價格在收縮、沒亂噴")
        cond.append(f"**收斂期數 ≤ {f_conv:.0f}** — 均線 {f_conv:.0f} 期內把乖離壓到 5% 以內 → 攤牌時間點快到了")
        with st.expander("目前篩選的是什麼？（點開看說明）", expanded=False):
            st.markdown("以下條件**同時成立**才會列入：\n\n"
                        + "\n".join(f"{i}. {c}" for i, c in enumerate(cond, 1))
                        + "\n\n白話：趨勢向上、價格縮在一起、而且快要見真章。")

        if hit.empty:
            st.info("沒有符合條件的標的，可放寬左側篩選。")
        else:
            st.dataframe(
                hit.drop(columns=["解讀"]),
                use_container_width=True, hide_index=True,
                column_config={
                    "乖離率": st.column_config.NumberColumn(format="%.2f%%"),
                    "箱體寬度": st.column_config.NumberColumn(format="%.2f%%"),
                    "收斂期數": st.column_config.NumberColumn(format="%.0f"),
                },
            )
            st.caption("表格排序用，逐檔文字解讀請展開下方。")
            for _, r in hit.iterrows():
                with st.expander(f"{r['代號']} {r['名稱']}　{r['現價']}　乖離 {r['乖離率']:.1%}"):
                    st.write(r["解讀"].replace("｜", "\n\n"))

        st.download_button(
            "⬇️ 下載摘要 (CSV)",
            data=sdf.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"摘要_{ma_label}_{dt.date.today():%Y%m%d}_{len(sdf)}檔.csv",
            mime="text/csv",
            key="dl_summary",
        )
        st.divider()

# ---------------- 明細 CSV ----------------
if all_tables:
    combined = pd.concat(all_tables, ignore_index=True)
    buf = io.StringIO()
    combined.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ 下載全部扣抵明細 (CSV)",
        data=buf.getvalue().encode("utf-8-sig"),
        file_name=f"扣抵值_{ma_label}_{dt.date.today():%Y%m%d}_{len(all_tables)}檔.csv",
        mime="text/csv",
        type="primary",
        key="dl_detail",
    )
