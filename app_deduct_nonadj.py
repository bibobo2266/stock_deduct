"""
均線扣抵值分析工具 (MA Roll-off Projection)
台股 / parquet 資料層

v6 — 資料改讀 minervini_picks 的 data/prices.parquet（全市場 ~2000 檔，每日 15:30 自動更新）
     不再逐檔打 FinMind，掃描從數分鐘降到數秒，且不吃 API 額度。
     母體從「成交值前 N 名」改成流動性門檻——排行榜裝的都是今天最吵的股票，
     而這支工具要找的是安靜收縮的股票，用排行榜取樣等於在噪音裡找安靜。
"""
import io
import datetime as dt
import re

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

REPO = "bibobo2266/minervini_picks"
BRANCH = "main"
PRICES_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/prices.parquet"
UNIVERSE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/universe.parquet"


# ---------------- 資料層 ----------------
@st.cache_data(ttl=60 * 60 * 4, show_spinner="讀取行情資料…")
def load_prices_raw() -> pd.DataFrame:
    r = requests.get(PRICES_URL, timeout=120)
    r.raise_for_status()
    df = pd.read_parquet(io.BytesIO(r.content))
    df["date"] = pd.to_datetime(df["date"])
    # FinMind 對停牌日回傳整列 0，這些 0 會污染均線與箱體高低點
    df = df[df["close"] > 0]
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def load_prices(as_of=None) -> pd.DataFrame:
    """as_of 不為 None 時把資料截斷到該日，等同回到那天重跑一次。
    扣抵值不產生買賣訊號，所以不需要預先算好的歷史表——單檔即時重算
    不到 1 秒，加個日期就夠了。"""
    df = load_prices_raw()
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    return df


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_universe() -> pd.DataFrame:
    r = requests.get(UNIVERSE_URL, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def stock_name_map() -> dict:
    try:
        u = load_universe()
        return dict(zip(u["stock_id"], u["stock_name"]))
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def industry_map() -> dict:
    try:
        u = load_universe()
        return dict(zip(u["stock_id"], u["industry_category"]))
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def liquid_universe(min_wan: float = 5000, min_days: int = 250, as_of=None) -> list:
    """近 60 日均額超過門檻、且有足夠歷史的股票。
    流動性門檻而不是排行榜：正在收縮的股票天然不會出現在成交值前 N 名。"""
    df = load_prices(as_of)
    last60 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=95)]
    avg = last60.groupby("stock_id")["Trading_money"].mean()
    days = df.groupby("stock_id").size()
    keep = avg[(avg > min_wan * 1e4) & (days.reindex(avg.index) >= min_days)]
    return sorted(keep.index.tolist())


def get_daily(all_px: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    g = all_px[all_px["stock_id"] == stock_id]
    if g.empty:
        return pd.DataFrame()
    out = g[["date", "open", "max", "min", "close", "Trading_Volume"]].copy()
    out.columns = ["date", "open", "high", "low", "close", "volume"]
    return out.reset_index(drop=True)


def parse_tickers(raw: str) -> list:
    """空白、逗號（半形/全形）、換行、頓號都能當分隔符。
    這樣可以直接把 Minervini app 的空白分隔輸出整段貼進來。"""
    parts = re.split(r"[,\s、，;；]+", raw.strip())
    seen, out = set(), []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.set_index("date")
        .resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )


def analyse(bars: pd.DataFrame, period: int, horizon: int):
    """Return (table_df, ma_series, projected_ma) for the roll-off analysis."""
    close = bars["close"].to_numpy(dtype=float)
    dates = bars["date"]
    n = len(close)
    if n < period + 2:
        return None, None, None

    ma = pd.Series(close).rolling(period).mean()
    cur_price = close[-1]
    steps = min(horizon, period - 1)

    rows = []
    proj_window = list(close[-period:])
    proj_ma = []
    for k in range(1, steps + 1):
        idx = n - period + k - 1
        roll_off = close[idx]
        roll_date = dates.iloc[idx].date()
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
    return pd.DataFrame(rows), ma, proj_ma


def verdict(tbl: pd.DataFrame, cur_price: float, cur_ma: float) -> str:
    up = (tbl["均線方向"] == "🟢 上彎").sum()
    total = len(tbl)
    stance = "站上" if cur_price > cur_ma else "跌破"
    dirs = tbl["均線方向"].tolist()
    flip = None
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[0]:
            flip = i + 1
            break
    msg = (f"現價 {cur_price:.2f}，{stance}均線 {cur_ma:.2f}。未來 {total} 期中 {up} 期扣抵偏低"
           f"（均線上彎），{total - up} 期扣抵偏高（下彎壓力）。")
    if flip:
        msg += f" 第 {flip} 期起方向可能改變（扣抵值轉{'高' if dirs[0] == '🟢 上彎' else '低'}），屆時需觀察股價是否跟上。"
    else:
        msg += " 整段期間方向一致，趨勢延續性較高。"
    return msg


# ---------------- 摘要層 ----------------
def summarize(bars: pd.DataFrame, tbl: pd.DataFrame, ma_series,
              period: int, look: int = 12, tol: float = 0.05) -> dict:
    """把單檔壓成可排序的純數字（float / bool，不含格式字串）。"""
    close = bars["close"].to_numpy(dtype=float)
    price = float(close[-1])
    ma_now = float(ma_series.iloc[-1])

    look = min(look, len(bars))
    box_hi = float(bars["high"].iloc[-look:].max())
    box_lo = float(bars["low"].iloc[-look:].min())

    # 收斂期數：價格持平下，均線幾期後把乖離壓到 tol 以內 = 攤牌時間點
    # 兩個試過的爛定義，留著當警惕：
    #   「均線追上現價」→ 數學上恆等於 period（視窗填滿才相等），沒鑑別度
    #   「均線升到箱底」→ 實測 94% 的股票都是 1（均線本來就高於近期低點）
    target = price * (1 - tol)
    if ma_now >= target:
        conv = 0.0
    else:
        c, win = None, list(close[-period:])
        for k in range(1, period + 1):
            win.pop(0)
            win.append(price)
            if sum(win) / period >= target:
                c = k
                break
        conv = float(c) if c else float(period)

    width = (box_hi - box_lo) / box_lo
    pos = (price - box_lo) / (box_hi - box_lo) if box_hi > box_lo else 0.5

    # 狀態 = 寬度 × 位置的九宮格。
    # 只看寬度會把 31% 的亞德客和 99% 的華邦電歸成同一類；
    # 只看寬度也會讓「窄箱貼頂」（最接近突破）藏在「盤整中」裡面。
    def _grid(w, p_):
        hi, lo = p_ >= 0.70, p_ <= 0.35
        if w <= 0.30:
            return "窄箱貼頂" if hi else ("窄箱貼底" if lo else "盤整中")
        if w <= 0.60:
            return "剛噴出" if hi else ("噴完回落" if lo else "波動偏大")
        return "剛噴出" if hi else ("噴完回落" if lo else "劇烈震盪")

    state = _grid(width, pos)

    # 邊界註記：把位置與寬度各推 ±3pp，狀態真的會翻面才標。
    etol = 0.03
    edge = any(_grid(width + dw, pos + dp) != state
               for dw in (-etol, 0, etol) for dp in (-etol, 0, etol))

    return {
        "現價": round(price, 2),
        "均線": round(ma_now, 2),
        "乖離率": price / ma_now - 1,
        "收斂期數": conv,
        "確定上彎": bool(tbl["均線方向"].str.contains("上彎").all()),
        "箱體寬度": width,
        "箱內位置": pos,
        "狀態": state,
        "邊界": edge,
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

    b, pos, st_ = r["箱體寬度"], r["箱內位置"], r["狀態"]
    if st_ == "窄箱貼頂":
        p.append(f"箱體 {b:.1%} 且貼在箱頂（窄幅整理又頂在上緣，最接近突破的型態）")
    elif st_ == "窄箱貼底":
        p.append(f"箱體 {b:.1%} 但壓在箱底（窄幅整理但撐在下緣，要守住才有戲）")
    elif st_ == "盤整中":
        p.append(f"箱體 {b:.1%}，價格在箱子中段（收斂中，方向未明）")
    elif st_ == "剛噴出":
        p.append(f"箱體 {b:.1%} 且貼近箱頂（這不是箱型，是剛噴出的股票，追高風險極大）")
    elif st_ == "噴完回落":
        p.append(f"箱體 {b:.1%} 但只在箱子下緣（高點已過，現在是回落段，不是盤整）")
    elif st_ == "波動偏大":
        p.append(f"箱體 {b:.1%}，價格在中段（波動偏大，還沒整理乾淨）")
    else:
        p.append(f"箱體 {b:.1%} 上下劇烈甩動（沒有結構可言，扣抵值判讀意義低）")

    p.append(f"箱頂 {r['箱頂']} / 箱底 {r['箱底']}，現價在箱內 {pos:.0%} 位置"
             "（站上箱頂才叫真突破）")

    if r.get("邊界"):
        p.append("⚠️ 位置/寬度剛好卡在分類門檻上，下面的狀態只差幾個百分點就會翻面，"
                 "當參考不當訊號")

    if not r["確定上彎"]:
        p.append("→ 【避開】結構在轉弱")
    elif st_ == "噴完回落":
        p.append("→ 【避開】主升段已過，均線上彎只是舊帳，不代表還能漲")
    elif st_ == "剛噴出":
        p.append("→ 【不追】正在噴的段落，扣抵值幫不上忙，等它做出箱子再看")
    elif st_ == "窄箱貼頂":
        p.append("→ 【重點觀察】窄箱 + 貼頂 + 均線上彎，三個條件到齊，盯箱頂")
    elif st_ == "窄箱貼底":
        p.append("→ 【等訊號】結構還在但價格弱，等它站回箱子中段再談")
    elif d > 0.15:
        p.append("→ 【等回檔】方向對但位置太高，等乖離縮小")
    else:
        p.append("→ 【觀望】還沒到決勝點")
    return "｜".join(p)


# ---------------- UI ----------------
st.title("📉 均線扣抵值分析工具")
st.caption("扣抵值 = 未來即將被移出均線計算的舊價格。現價 > 扣抵值 → 均線上彎；現價 < 扣抵值 → 均線下彎。")

with st.sidebar:
    st.header("設定")
    mode = st.radio("標的來源", ["手動輸入", "流動性母體（全市場）"], horizontal=True)

    if mode == "手動輸入":
        tickers_raw = st.text_area(
            "股票代號",
            value="2330 2317 2454",
            height=140,
            help="空白、逗號、換行都可以。可直接貼 Minervini app 的代號清單。",
        )
        liq_min, max_n = 5000, 0
    else:
        tickers_raw = ""
        liq_min = st.number_input("60日均額門檻（萬元）", 500, 100000, 5000, 500,
                                  help="流動性過濾。不用排行榜——正在收縮的股票"
                                       "天然不會出現在成交值前 N 名。")
        max_n = st.slider("最多分析幾檔（0 = 全部）", 0, 800, 300, 50,
                          help="全市場約 700-800 檔，全跑約 1-2 分鐘。")

    if st.checkbox("自訂均線週期", value=False,
                   help="台股淺碟，週線 W20 常比 W30 貼近實況；也可以自己填任意期數。"):
        _f = st.radio("頻率", ["週線", "日線"], horizontal=True)
        _n = st.number_input("均線期數", 5, 250, 20 if _f == "週線" else 60)
        freq_override = "W" if _f == "週線" else "D"
        ma_label = f"{_f} {'W' if freq_override == 'W' else 'MA'}{_n}"
        period_override = int(_n)
    else:
        ma_label = st.selectbox("均線", list(MA_OPTIONS.keys()),
                                index=list(MA_OPTIONS).index("週線 W20"))
        freq_override = period_override = None

    st.divider()
    hist = st.checkbox("回到某一天重算", value=False,
                       help="把資料截斷到指定日期，等同回到那天跑一次。"
                            "用來回答「當時這檔的均線狀況如何」。")
    as_of = None
    if hist:
        as_of = st.date_input("資料截止日", value=dt.date.today(),
                              min_value=dt.date(2017, 1, 1),
                              max_value=dt.date.today())
        st.caption("目前 parquet 只到 2024-07 起。更早的日期要先回補還原股價。")
    st.divider()

    horizon = st.slider("往後推估期數（扣抵值要列幾期）", 4, 60, 12)
    look_back = st.slider("箱體回看期數", 4, 26, 10,
                          help="用近幾期的真實最高/最低算箱體。預設 10 是配 W20 抓的"
                               "（箱體約佔均線窗口一半）。切回 W30 時可以放到 12~15。")
    show_charts = st.checkbox("顯示個股圖表", value=True)
    st.caption("超過 20 檔時自動關閉圖表，只輸出摘要，避免瀏覽器卡死。")

    st.divider()
    st.subheader("摘要篩選")
    only_top = st.checkbox(
        "只看「窄箱貼頂」", value=False,
        help="這支工具唯一會給出【重點觀察】的狀態：窄幅整理 + 貼在上緣 + 均線上彎。"
             "勾了就直接看結論，不用自己在幾百檔裡翻。",
    )
    f_up = st.checkbox(
        "只看均線確定上彎", value=False,
        help="未來 12 期的扣抵值全部低於現價 → 均線這段期間不可能翻下。"
             "這是最嚴的一條，一個扣抵值高於現價就出局。",
    )
    f_box = st.slider(
        "箱體寬度上限", 0.05, 0.60, 0.25, 0.01,
        help="近 N 期真實最高與最低的差距。越小代表價格越收縮、能量壓得越緊。"
             "台股週線 15% 以內算很嚴，25% 起跳比較抓得到東西。",
    )
    # 收斂期數當篩選沒有鑑別度：窄箱 → 價格貼近均線 → 收斂期數必然小。
    # 它跟乖離率的相關係數 0.76，本質是同一件事的兩種說法 —— 所以改當排序鍵。
    sort_by = st.selectbox(
        "摘要表排序", ["收斂期數（攤牌最近的排前面）", "箱體寬度（最窄的排前面）",
                    "箱內位置（最貼箱頂的排前面）", "乖離率（最小的排前面）"],
        help="排序不篩掉任何標的，只決定名單裡誰排前面。",
    )
    st.caption("篩選只影響上方摘要表，下方個股區仍會列出全部。")

    run = st.button("開始分析", type="primary", use_container_width=True)

if not run:
    st.info("左側輸入代號或選流動性母體，按「開始分析」。"
            "週線模式下 1 期 = 1 週，日線模式 1 期 = 1 個交易日。")
    st.stop()

if period_override:
    freq, period = freq_override, period_override
else:
    freq, period = MA_OPTIONS[ma_label]

all_px = load_prices(as_of)
if all_px.empty:
    st.error("該日期之前沒有資料。")
    st.stop()
_d = all_px["date"].max()
st.caption(f"資料日 {_d:%Y-%m-%d} · 全市場 {all_px['stock_id'].nunique()} 檔"
           + (f" · ⏪ 回溯模式（截斷至 {as_of}）" if as_of else ""))
if as_of and (_d.date() - as_of).days < -3:
    st.warning(f"最後一筆資料是 {_d:%Y-%m-%d}，比你選的日期早。回補範圍不足。")

if mode == "手動輸入":
    tickers = parse_tickers(tickers_raw)
else:
    with st.spinner("篩選流動性母體…"):
        tickers = liquid_universe(liq_min, as_of=as_of)
    if max_n:
        tickers = tickers[:max_n]
    st.success(f"母體 {len(tickers)} 檔（60日均額 > {liq_min:,.0f} 萬）")

if not tickers:
    st.error("沒有輸入任何代號。")
    st.stop()

detail = show_charts and len(tickers) <= 20
names = stock_name_map()
inds = industry_map()

all_tables = []
summaries = []
missing = []

summary_slot = st.container()
detail_slot = st.container()
progress = st.progress(0.0)

for i, tk in enumerate(tickers, 1):
    if i % 10 == 0 or i == len(tickers):
        progress.progress(i / len(tickers), text=f"處理 {tk} ({i}/{len(tickers)})")
    daily = get_daily(all_px, tk)
    if daily.empty:
        missing.append(tk)
        continue

    bars = to_weekly(daily) if freq == "W" else daily
    tbl, ma, proj = analyse(bars, period, horizon)
    if tbl is None:
        missing.append(tk)
        continue

    name = names.get(tk, "")
    cur_price = float(bars["close"].iloc[-1])
    cur_ma = float(ma.iloc[-1])

    try:
        summaries.append({"代號": tk, "名稱": name, "產業": inds.get(tk, ""),
                          **summarize(bars, tbl, ma, period, look=look_back)})
    except Exception as e:
        st.warning(f"{tk}：摘要計算失敗 — {e}")

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
if missing:
    st.caption(f"略過 {len(missing)} 檔（無資料或歷史不足）：{', '.join(missing[:20])}"
               f"{' …' if len(missing) > 20 else ''}")

# ---------------- 摘要表 ----------------
if summaries:
    sdf = pd.DataFrame(summaries)
    sdf["解讀"] = sdf.apply(explain, axis=1)

    if only_top:
        m = sdf["狀態"] == "窄箱貼頂"
    else:
        m = sdf["箱體寬度"] <= f_box
        if f_up:
            m &= sdf["確定上彎"]

    _sort = {"收斂期數（攤牌最近的排前面）": ["收斂期數", "箱體寬度"],
             "箱體寬度（最窄的排前面）": ["箱體寬度", "收斂期數"],
             "箱內位置（最貼箱頂的排前面）": None,
             "乖離率（最小的排前面）": ["乖離率", "箱體寬度"]}[sort_by]
    hit = (sdf[m].sort_values("箱內位置", ascending=False) if _sort is None
           else sdf[m].sort_values(_sort))

    with summary_slot:
        st.subheader(f"📊 摘要 — 符合條件 {len(hit)} / {len(sdf)} 檔")

        vc = sdf["狀態"].value_counts()
        cols = st.columns(4)
        cols[0].metric("🎯 窄箱貼頂", int(vc.get("窄箱貼頂", 0)), help="唯一的【重點觀察】")
        cols[1].metric("盤整中", int(vc.get("盤整中", 0)))
        cols[2].metric("剛噴出", int(vc.get("剛噴出", 0)), help="【不追】")
        cols[3].metric("噴完回落", int(vc.get("噴完回落", 0)), help="【避開】")

        if only_top:
            st.caption("目前只顯示「窄箱貼頂」——窄幅整理 + 貼在上緣，最接近突破的型態。")
        else:
            cond = []
            if f_up:
                cond.append("**確定上彎 ✅** — 未來 12 期扣抵值全部低於現價 → 均線這段期間不可能翻下")
            cond.append(f"**箱體寬度 ≤ {f_box:.0%}** — 近 {look_back} 期高低差在 {f_box:.0%} 以內 → 價格在收縮、沒亂噴")
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
                    "箱內位置": st.column_config.NumberColumn(format="%.0f%%"),
                    "收斂期數": st.column_config.NumberColumn(format="%.0f"),
                },
            )
            st.text_area("複製代號清單", " ".join(hit["代號"].tolist()),
                         height=68, key="copy_hits")
            st.caption("表格排序用，逐檔文字解讀請展開下方。")
            for _, r in hit.head(40).iterrows():
                with st.expander(f"{r['代號']} {r['名稱']} {r['現價']} 乖離 {r['乖離率']:.1%}"):
                    st.write(r["解讀"].replace("｜", "\n\n"))

        st.download_button(
            "⬇️ 下載摘要 (CSV)",
            data=sdf.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"摘要_{ma_label}_{(as_of or dt.date.today()):%Y%m%d}_{len(sdf)}檔.csv",
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
        file_name=f"扣抵值_{ma_label}_{(as_of or dt.date.today()):%Y%m%d}_{len(all_tables)}檔.csv",
        mime="text/csv",
        type="primary",
        key="dl_detail",
    )
