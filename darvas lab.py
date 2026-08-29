"""
darvas_lab.py

達華斯箱型（Darvas Box）實驗頁 — 重疊率驗證

唯一目的：算出「Darvas 觸發但不在 Minervini 名單內」的比例，
決定要不要開第四個 repo。低於 20% 就別開，把箱體與一目目標價併成模組。

放置位置：bibobo2266/stock_deduct 根目錄
部署：Streamlit Cloud 另開一個 app，Main file path 填 darvas_lab.py
      （app_deduct.py 完全不受影響，不需要 pages/ 目錄）
requirements.txt 不用改。

判定邏輯的來源：
  Minervini 側 — build_matrices / scan / zigzag / vcp_foot / base_count /
                 stage_of / classify 全部照 app_minervini.py 原樣搬過來，
                 確保重疊率的分母跟你每天實際看的名單是同一個東西。
  扣抵值側   — 九宮格、確定上彎、收斂期數照 app_deduct.py 的 summarize 搬。
  Darvas 側  — detect_boxes 是新的，事件驅動狀態機。
  一目側     — ichimoku_targets 是新的，水準論四式。
"""

import datetime as dt
import io

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="達華斯箱型實驗", page_icon="▣", layout="wide")

REPO = "bibobo2266/minervini_picks"
BRANCH = "main"
PRICES_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/prices.parquet"
UNIVERSE_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data/universe.parquet"

MA_WEEKS = 30
DEFAULT_LIQ = 5000  # 萬元，60 日均額門檻


# ================================================================== 資料層

@st.cache_data(ttl=60 * 60 * 4, show_spinner="讀取行情資料…")
def load_parquet(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def build_matrices():
    """把長表轉成 date × stock_id 的寬矩陣。與 app_minervini 相同。"""
    df = load_parquet(PRICES_URL)
    df["date"] = pd.to_datetime(df["date"])
    m = {}
    for k, col in [("c", "close"), ("h", "max"), ("l", "min"),
                   ("v", "Trading_Volume"), ("mo", "Trading_money")]:
        m[k] = df.pivot(index="date", columns="stock_id", values=col).sort_index()

    # FinMind 對停牌日回傳整列 0。價格沿用前一日，量與額留白。
    bad = m["c"] <= 0
    for k in ("c", "h", "l"):
        m[k] = m[k].mask(bad).ffill()
    for k in ("v", "mo"):
        m[k] = m[k].mask(bad)
    return m


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_universe() -> pd.DataFrame:
    return load_parquet(UNIVERSE_URL).set_index("stock_id")


def px_of(m, sid) -> pd.DataFrame:
    """從矩陣取單檔日線 OHLCV（欄名沿用 app_minervini 慣例）。"""
    return pd.DataFrame({
        "close": m["c"][sid], "max": m["h"][sid],
        "min": m["l"][sid], "Trading_Volume": m["v"][sid],
    }).dropna()


# ========================================================= Minervini 側（照搬）

def zigzag(h, l, pct=6.0):
    n = len(h)
    if n < 10:
        return []
    piv, d, ei, ep = [], 0, 0, h[0]
    for i in range(1, n):
        if d >= 0 and h[i] > ep:
            ei, ep = i, h[i]
        if d < 0 and l[i] < ep:
            ei, ep = i, l[i]
        if d >= 0:
            if l[i] <= ep * (1 - pct / 100):
                piv.append((ei, ep, "H")); d = -1; ei, ep = i, l[i]
        else:
            if h[i] >= ep * (1 + pct / 100):
                piv.append((ei, ep, "L")); d = 1; ei, ep = i, h[i]
    piv.append((ei, ep, "H" if d >= 0 else "L"))
    return piv


def vcp_foot(px: pd.DataFrame, pct=6.0):
    out = dict(foot="", n=0, first=None, last=None, ok=False, pivot=None, near=False)
    b = px.tail(120)
    if len(b) < 40:
        return out
    h, l = b["max"].values, b["min"].values
    piv = zigzag(h, l, pct)
    depths, starts = [], []
    for i in range(len(piv) - 1):
        if piv[i][2] == "H" and piv[i + 1][2] == "L":
            depths.append(round((piv[i][1] - piv[i + 1][1]) / piv[i][1] * 100, 1))
            starts.append(piv[i][0])
    price = float(b["close"].iloc[-1])
    pivot = round(float(b["max"].tail(20).max()), 2)
    out.update(pivot=pivot, near=bool(price >= pivot * 0.97))
    if len(depths) < 2:
        out.update(n=len(depths))
        return out
    depths, starts = depths[-4:], starts[-4:]
    wk = max(2, (len(b) - min(starts)) // 5)
    a = ((b["max"] - b["min"]) / b["close"] * 100).tail(60).values
    seg = len(a) // 3
    s1, s2, s3 = a[:seg].mean(), a[seg:2 * seg].mean(), a[2 * seg:].mean()
    vol = b["Trading_Volume"].tail(60).values
    dry = vol[-10:].mean() <= vol[:-10].mean() * 1.05
    ok = bool(s3 <= s1 * 0.75 and s3 <= s2 * 1.05 and s3 <= 6.0 and dry)
    out.update(foot=f"{wk}W-{depths[0]:.0f}/{depths[-1]:.0f}-{len(depths)}T",
               n=len(depths), first=depths[0], last=depths[-1], ok=ok)
    return out


def base_count(px: pd.DataFrame, thr=12.0, min_bars=15, lookback=378):
    b = px.tail(lookback)
    c, h = b["close"].values, b["max"].values
    if len(c) < 60:
        return 0
    peak, pk_i, n, in_base, bs = h[0], 0, 0, False, 0
    for i in range(len(c)):
        if h[i] > peak:
            if in_base and (i - bs) >= min_bars:
                n += 1
            in_base, peak, pk_i = False, h[i], i
        elif not in_base and c[i] <= peak * (1 - thr / 100):
            in_base, bs = True, pk_i
    return min(6, n + (1 if in_base else 0))


def stage_of(px: pd.DataFrame):
    """Weinstein 階段（30 週線 + 斜率）。"""
    w = px["close"].resample("W-FRI").last().dropna()
    if len(w) < MA_WEEKS + 5:
        return None, None, None
    ma = w.rolling(MA_WEEKS).mean()
    price, ma_now = w.iloc[-1], ma.iloc[-1]
    slope = ma.diff(4).iloc[-1]
    above = price > ma_now
    stage = 2 if (above and slope > 0) else 4 if (not above and slope < 0) \
        else 3 if above else 1
    return stage, round(float(ma_now), 2), not above


def mv_scan(m, liq_wan: float, min_days: int = 250):
    """向量化趨勢模板 + RS。與 app_minervini.scan() 相同。"""
    c, h, l, v, mo = m["c"], m["h"], m["l"], m["v"], m["mo"]
    keep = (mo.tail(60).mean() > liq_wan * 1e4) & (c.notna().sum() >= min_days)
    ids = keep[keep].index
    if len(ids) == 0:
        return pd.DataFrame()
    c, h, l, v, mo = [x[ids] for x in (c, h, l, v, mo)]

    ma50, ma150, ma200 = (c.rolling(k).mean() for k in (50, 150, 200))
    px = c.iloc[-1]
    m50, m150, m200, m200_1mo = (ma50.iloc[-1], ma150.iloc[-1],
                                 ma200.iloc[-1], ma200.iloc[-22])
    lo52, hi52 = l.tail(252).min(), h.tail(252).max()
    r126 = c.iloc[-1] / c.iloc[-127] - 1
    r63 = c.iloc[-1] / c.iloc[-64] - 1
    rs = (0.6 * r126 + 0.4 * r63).rank(pct=True) * 100

    cond = pd.DataFrame({
        "c1": (px > m150) & (px > m200),
        "c2": m150 > m200,
        "c3": m200 > m200_1mo,
        "c4": (m50 > m150) & (m150 > m200),
        "c5": px > m50,
        "c6": px >= lo52 * 1.30,
        "c7": px >= hi52 * 0.75,
        "c8": rs >= 70,
    })
    v5, v10 = v.tail(5).mean(), v.tail(10).mean()
    return pd.DataFrame({
        "收盤": px.round(2),
        "RS": rs.round(0),
        "TT分": cond.sum(axis=1),
        "距高點": ((px / hi52 - 1) * 100).round(1),
        "量比": (v5 / v10).round(2),
        "量增": v5 > v10,
        "均額億": (mo.tail(60).mean() / 1e8).round(2),
    })


def classify(row) -> str:
    """五色分流。與 app_minervini.classify() 相同。"""
    if row["TT分"] < 6 or row["階段"] != 2:
        return "淘汰"
    near_hi = row["距高點"] >= -10
    if row["突破"] and row["量增"] and row["TT分"] >= 7 and near_hi:
        return "觸發"
    if row["近樞紐"] and near_hi and (row["VCP"] or row["TT分"] >= 7):
        return "準備"
    return "觀察"


# ========================================================= 扣抵值側（照搬）

def to_weekly(px: pd.DataFrame) -> pd.DataFrame:
    """日線 → 週線（W-FRI），欄名轉成 high/low/close。"""
    w = px.rename(columns={"max": "high", "min": "low"})
    return (w.resample("W-FRI")
             .agg({"high": "max", "low": "min", "close": "last"})
             .dropna(subset=["close"]))


def deduct_summary(bars: pd.DataFrame, period: int = 20,
                   horizon: int = 12, look: int = 10, tol: float = 0.05) -> dict:
    """app_deduct.summarize() 的等價實作（週線 W20 / 箱體回看 10 期）。"""
    close = bars["close"].to_numpy(dtype=float)
    n = len(close)
    if n < period + 2:
        return {}
    price = float(close[-1])
    ma_now = float(pd.Series(close).rolling(period).mean().iloc[-1])

    # 確定上彎：未來 horizon 期扣抵值全部低於現價
    steps = min(horizon, period - 1)
    ups = [price > close[n - period + k - 1] for k in range(1, steps + 1)]
    sure_up = bool(ups) and all(ups)

    # 收斂期數：價格持平下，均線幾期後把乖離壓到 tol 以內
    target = price * (1 - tol)
    if ma_now >= target:
        conv = 0.0
    else:
        conv, win = float(period), list(close[-period:])
        for k in range(1, period + 1):
            win.pop(0)
            win.append(price)
            if sum(win) / period >= target:
                conv = float(k)
                break

    look = min(look, len(bars))
    box_hi = float(bars["high"].iloc[-look:].max())
    box_lo = float(bars["low"].iloc[-look:].min())
    width = (box_hi - box_lo) / box_lo if box_lo > 0 else np.nan
    pos = (price - box_lo) / (box_hi - box_lo) if box_hi > box_lo else 0.5

    def _grid(w, p_):
        hi, lo = p_ >= 0.70, p_ <= 0.35
        if w <= 0.30:
            return "窄箱貼頂" if hi else ("窄箱貼底" if lo else "盤整中")
        if w <= 0.60:
            return "剛噴出" if hi else ("噴完回落" if lo else "波動偏大")
        return "剛噴出" if hi else ("噴完回落" if lo else "劇烈震盪")

    state = _grid(width, pos)
    etol = 0.03
    edge = any(_grid(width + dw, pos + dp) != state
               for dw in (-etol, 0, etol) for dp in (-etol, 0, etol))

    return {
        "週線乖離": round(price / ma_now - 1, 4),
        "收斂期數": conv,
        "確定上彎": sure_up,
        "扣抵箱寬": round(width, 4),
        "扣抵箱位": round(pos, 3),
        "扣抵狀態": state,
        "邊界": edge,
    }


# ========================================================= Darvas 側（新）

def tick_size(price: float) -> float:
    """台股最小升降單位。"""
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.10
    if price < 500:
        return 0.50
    if price < 1000:
        return 1.00
    return 5.00


def detect_boxes(high: np.ndarray, low: np.ndarray, confirm: int = 3):
    """
    達華斯箱型，事件驅動狀態機。

      SEEK_TOP  找新高。新高後連續 confirm 根沒被超過 → 箱頂確立
      SEEK_BOT  找箱底。低點後連續 confirm 根沒被跌破 → 箱體成立
      IN_BOX    箱體有效。向上突破 → 疊下一個箱；跌破箱底 → 箱序歸零

    這跟扣抵值 app 的箱體不是同一件事：那裡是固定回看 N 期的高低點
    （隨窗口滑動而漂移），這裡的箱體一旦確立就固定到被突破為止。
    """
    n = len(high)
    boxes = []
    if n < confirm + 2:
        return boxes

    state = "SEEK_TOP"
    top, top_i = float(high[0]), 0
    bot, bot_i = np.nan, -1
    seq = 0
    cur = None

    for i in range(1, n):
        if state == "SEEK_TOP":
            if high[i] > top:
                top, top_i = float(high[i]), i
            elif i - top_i >= confirm:
                seg = low[top_i + 1: i + 1]
                if len(seg) == 0:
                    continue
                k = int(np.argmin(seg))
                bot, bot_i = float(seg[k]), top_i + 1 + k
                state = "SEEK_BOT"

        elif state == "SEEK_BOT":
            if high[i] > top:
                top, top_i = float(high[i]), i
                state = "SEEK_TOP"
            elif low[i] < bot:
                bot, bot_i = float(low[i]), i
            elif i - bot_i >= confirm:
                seq += 1
                cur = {"top": top, "top_i": top_i, "bot": bot, "bot_i": bot_i,
                       "confirm_i": i, "break_i": None, "fail_i": None, "seq": seq}
                boxes.append(cur)
                state = "IN_BOX"

        elif state == "IN_BOX":
            if high[i] > top:
                cur["break_i"] = i
                top, top_i = float(high[i]), i
                state = "SEEK_TOP"
            elif low[i] < bot:
                cur["fail_i"] = i
                seq = 0
                top, top_i = float(high[i]), i
                state = "SEEK_TOP"

    return boxes


def ichimoku_targets(a: float, b: float, c: float) -> dict:
    """一目均衡表 水準論。A = 起漲低點, B = 箱頂, C = 箱底。"""
    return {"NT值": c + (c - a), "N值": c + (b - a),
            "V值": b + (b - c), "E值": b + (b - a)}


# ================================================================== 主流程

def run(liq_wan, min_tt, confirm, fresh_days, box_width_max, box_bars):
    m = build_matrices()
    uni = load_universe()

    base = mv_scan(m, liq_wan)
    if base.empty:
        return None, None, None

    rows = []
    ids = list(base.index)
    prog = st.progress(0.0, text="箱型偵測中…")
    for i, sid in enumerate(ids, 1):
        if i % 25 == 0 or i == len(ids):
            prog.progress(i / len(ids), text=f"箱型偵測中… {sid} ({i}/{len(ids)})")

        px = px_of(m, sid)
        if len(px) < 260:
            continue

        seg = px.tail(box_bars)
        boxes = detect_boxes(seg["max"].values.astype(float),
                             seg["min"].values.astype(float), confirm)
        if not boxes:
            continue

        last = boxes[-1]
        anchor = last["break_i"] if last["break_i"] is not None else last["confirm_i"]
        if len(seg) - 1 - anchor > fresh_days:
            continue

        top, bot = last["top"], last["bot"]
        if bot <= 0 or top <= bot:
            continue
        width = (top - bot) / bot
        if width > box_width_max:
            continue

        b = base.loc[sid]
        price = float(px["close"].iloc[-1])
        tk = tick_size(top)
        trigger, stop = top + tk, bot - tk

        # 一目 A 點：前一個箱底；只有單一箱體時退回箱頂前 120 根的最低
        if len(boxes) >= 2:
            a_pt = boxes[-2]["bot"]
        else:
            s0 = max(0, last["top_i"] - 120)
            a_pt = float(np.min(seg["min"].values[s0: last["top_i"] + 1]))
        tgt = ichimoku_targets(a_pt, top, bot)

        # --- Minervini 側完整判定 ---
        stg, _, _ = stage_of(px)
        f = vcp_foot(px)
        # app_minervini 的 ① 選股只把 TT分 >= min_tt 送進型態分析，
        # 沒進去的那批在他的名單裡根本不存在 → 這裡等同淘汰
        if int(b["TT分"]) < min_tt:
            mv_state = "淘汰"
        else:
            mv_state = classify({
                "TT分": int(b["TT分"]), "階段": stg or 0, "距高點": b["距高點"],
                "突破": bool(b["收盤"] >= (f["pivot"] or 1e9)),
                "量增": bool(b["量增"]), "近樞紐": f["near"], "VCP": f["ok"],
            })

        ded = deduct_summary(to_weekly(px))

        rows.append({
            "代號": sid,
            "名稱": uni.loc[sid, "stock_name"] if sid in uni.index else "?",
            "產業": uni.loc[sid, "industry_category"] if sid in uni.index else "",
            "收盤": round(price, 2),
            "箱頂": round(top, 2), "箱底": round(bot, 2),
            "觸發價": round(trigger, 2), "停損價": round(stop, 2),
            "單筆風險%": round((trigger - stop) / trigger * 100, 2),
            "箱寬%": round(width * 100, 2),
            "箱內位置": round((price - bot) / (top - bot), 3),
            "箱序": last["seq"],
            "箱齡": int(len(seg) - 1 - last["confirm_i"]),
            "已突破": last["break_i"] is not None,
            "NT值": round(tgt["NT值"], 2), "N值": round(tgt["N值"], 2),
            "V值": round(tgt["V值"], 2), "E值": round(tgt["E值"], 2),
            "Minervini狀態": mv_state,
            "TT分": int(b["TT分"]),
            "RS": int(b["RS"]) if pd.notna(b["RS"]) else None,
            "階段": stg or 0, "距高點": b["距高點"], "均額億": b["均額億"],
            "底部序": base_count(px),
            **ded,
        })
    prog.empty()
    return pd.DataFrame(rows), len(base), f"{m['c'].index[-1]:%Y-%m-%d}"


# ================================================================== UI

st.title("▣ 達華斯箱型 × 扣抵值 × 一目水準論")
st.caption("重疊率實驗。Minervini 判定完整複製 app_minervini 的 scan → stage → VCP → classify，"
           "扣抵值欄位複製 app_deduct 的九宮格與確定上彎。")

with st.sidebar:
    st.header("Darvas 參數")
    confirm = st.number_input("箱體確認根數（原著 3）", 2, 6, 3)
    fresh_days = st.number_input("只看最近 N 根內的箱體事件", 1, 60, 10)
    box_width_max = st.slider("箱寬上限 %", 5, 60, 25) / 100
    box_bars = st.number_input("箱型偵測回看根數", 120, 500, 300, 20)
    st.divider()
    st.header("母體 / Minervini")
    liq_wan = st.number_input("60日均額門檻（萬元）", 500, 100000, DEFAULT_LIQ, 500)
    min_tt = st.slider("最低 TT 分（對應 ① 選股的滑桿，預設 8）", 4, 8, 8)
    st.caption("這個值必須跟你平常在 Minervini app 用的一致，否則分母不對。")
    st.divider()
    go = st.button("執行掃描", type="primary", use_container_width=True)

if not go:
    st.info("設定完按「執行掃描」。首次要下載 15MB parquet，全程約 1–2 分鐘。")
    st.stop()

df, pool, day = run(liq_wan, min_tt, confirm, fresh_days, box_width_max, box_bars)

if df is None or df.empty:
    st.warning("這組參數下沒有箱體事件。放寬箱寬上限或拉長 fresh_days。")
    st.stop()

st.caption(f"母體 {pool} 檔 · 資料日 {day} · Darvas 事件 {len(df)} 檔")

total = len(df)
inlist = int((df["Minervini狀態"] != "淘汰").sum())
unique = total - inlist
rate = unique / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Darvas 事件", total)
c2.metric("已在 Minervini 名單", inlist)
c3.metric("Darvas 獨有", unique)
c4.metric("獨有比例", f"{rate:.1f}%")

if rate >= 20:
    st.success(f"{rate:.1f}% ≥ 20% — 有獨立價值，可考慮開新 repo（先看下面的分佈）。")
else:
    st.error(f"{rate:.1f}% < 20% — 不要開新 repo，把箱體與水準論併成 minervini_picks 的模組。")

st.divider()

t1, t2, t3, t4 = st.tabs(["Darvas 獨有", "全部事件", "交叉分佈", "欄位說明"])

with t1:
    sub = df[df["Minervini狀態"] == "淘汰"].sort_values(["箱序", "箱寬%"])
    st.caption("這批是 Darvas 抓到、Minervini 漏掉的。逐檔看它們為什麼被淘汰"
               "（TT分不足？階段不對？距高點太遠？）才知道差異是不是真的有價值。")
    st.dataframe(sub, hide_index=True, use_container_width=True, height=480)
    st.download_button("下載 CSV", sub.to_csv(index=False).encode("utf-8-sig"),
                       f"darvas_unique_{dt.date.today():%Y%m%d}.csv", "text/csv")
    st.text_area("複製代號（可直接貼進扣抵值 app）",
                 " ".join(sub["代號"].tolist()), height=68)

with t2:
    st.dataframe(df.sort_values(["Minervini狀態", "箱寬%"]),
                 hide_index=True, use_container_width=True, height=480)
    st.download_button("下載 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       f"darvas_all_{dt.date.today():%Y%m%d}.csv", "text/csv")

with t3:
    st.write("**Minervini 狀態 × 扣抵值九宮格**")
    st.dataframe(pd.crosstab(df["Minervini狀態"], df["扣抵狀態"]),
                 use_container_width=True)
    sub = df[df["Minervini狀態"] == "淘汰"]
    if not sub.empty:
        st.write("**Darvas 獨有的淘汰原因分佈**")
        st.dataframe(pd.DataFrame({
            "TT分不足": [(sub["TT分"] < min_tt).sum()],
            "階段非2": [(sub["階段"] != 2).sum()],
            "距高點<-10%": [(sub["距高點"] < -10).sum()],
        }, index=["檔數"]), use_container_width=True)
    st.write("**箱序分佈**（達華斯：1–2 最佳，3 以上轉危險）")
    st.dataframe(df["箱序"].value_counts().sort_index().rename("檔數").to_frame().T,
                 use_container_width=True)

with t4:
    st.markdown("""
| 欄位 | 意義 |
|---|---|
| 箱頂 / 箱底 | 事件驅動確認的箱體，一旦確立就固定，不隨窗口漂移 |
| 觸發價 / 停損價 | 箱頂 +1 檔 / 箱底 −1 檔，台股檔位已依價格分級 |
| 單筆風險% | (觸發−停損)/觸發，直接餵 Minervini app ③ 的 1.25% 部位計算 |
| 箱序 | 自最近一次跌破箱底算起的第幾個箱 |
| NT/N/V/E 值 | 一目水準論，A=前一箱底 B=箱頂 C=箱底 |
| 扣抵箱寬 / 扣抵箱位 / 扣抵狀態 | 週線 W20、回看 10 期，與 app_deduct 同一套九宮格 |
| 確定上彎 | 未來 12 期扣抵值全部低於現價 |
| Minervini狀態 | 觸發/準備/觀察/淘汰，判定流程與 app_minervini 相同 |

**「箱寬%」與「扣抵箱寬」是兩個不同的東西**：前者是 Darvas 箱體（日線、事件驅動），
後者是扣抵值 app 的固定回看箱體（週線、10 期）。兩者不一致本身就是資訊——
日線箱子很窄但週線箱子很寬，代表這是大波動裡的短暫喘息，不是真收縮。

**已知限制**
- 台股漲停 10% 會讓突破當天封死，觸發價實務上掛不到。「已突破」只供統計。
- Minervini app ① 選股的滑桿若不是 8，這裡的 min_tt 要跟著改，否則分母不對。
- 一目 A 點在只有單一箱體時退回前 120 根最低，目標價會偏高。
- 大盤環境（follow-through / 分佈日）未納入，那一層屬於 Stan_stages。
""")
