"""
A.R.E.S. — Automated Reconnaissance & Entry System
Crypto Futures Signal Bot for Binance

Strategy:
  - Scans 24 major altcoin pairs on Binance Futures every hour
  - Uses 1H EMA crossover + RSI + volume filters for entry signals
  - Uses BTC 4H trend as a market direction filter (only trades with the trend)
  - Ranks signals by quality score and sends them via Telegram

Usage:
  1. Copy .env.example to .env and fill in your credentials
  2. pip install -r requirements.txt
  3. python ares.py
"""

import os
import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import time
import logging
import gc
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env file.")

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        RotatingFileHandler("ares.log", maxBytes=5*1024*1024, backupCount=2, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────
WATCHLIST = [
    'ETH/USDT',  'SOL/USDT',  'BNB/USDT',  'AVAX/USDT', 'LINK/USDT',
    'XRP/USDT',  'ADA/USDT',  'DOGE/USDT', 'DOT/USDT',  'INJ/USDT',
    'FET/USDT',  'APT/USDT',  'SUI/USDT',  'ARB/USDT',  'OP/USDT',
    'NEAR/USDT', 'RNDR/USDT', 'ATOM/USDT', 'FTM/USDT',  'WLD/USDT',
    'TIA/USDT',  'SEI/USDT',  'MATIC/USDT','LDO/USDT'
]

# Lower liquidity coins get a relaxed volume threshold
LOW_LIQUIDITY_COINS = {'TIA/USDT','SEI/USDT','FET/USDT','WLD/USDT','RNDR/USDT','LDO/USDT'}

OHLCV_LIMIT  = 100
SIGNAL_TF    = '1h'   # Timeframe for signal detection
BTC_TREND_TF = '4h'   # Timeframe for BTC market direction filter

# Daily scan statistics (reset at 22:00 UTC)
daily_stats = {"scans": 0, "signals": 0, "long": 0, "short": 0}


# ══════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """Sends a formatted HTML message to Telegram. Returns True on success."""
    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        log.error(f"Telegram send failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# EXCHANGE CONNECTION
# ══════════════════════════════════════════════════════════════

def get_exchange() -> ccxt.binance:
    """Creates and returns a Binance Futures connection."""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })
    exchange.load_markets()
    log.info("Binance Futures connection established.")
    return exchange


def safe_fetch_ohlcv(exchange: ccxt.binance, symbol: str,
                     timeframe: str, limit: int, max_retry: int = 3):
    """
    Fetches OHLCV data with retry logic.
    Returns a list of candles or None on failure.
    """
    for attempt in range(1, max_retry + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            log.warning(f"[{symbol}] Network error (attempt {attempt}/{max_retry}): {e}")
            if attempt < max_retry:
                time.sleep(5 * attempt)
            else:
                return None
        except ccxt.RateLimitExceeded as e:
            log.warning(f"[{symbol}] Rate limit exceeded, waiting 30s: {e}")
            time.sleep(30)
        except ccxt.BadSymbol:
            log.error(f"[{symbol}] Invalid symbol, skipping.")
            return None
        except Exception as e:
            log.error(f"[{symbol}] Unexpected OHLCV error: {e}")
            return None
    return None


def reconnect_exchange(exchange: ccxt.binance) -> ccxt.binance:
    """
    Attempts to reconnect to the exchange up to 5 times.
    Returns the new connection or the original if all attempts fail.
    """
    log.warning("Reconnecting to exchange...")
    for attempt in range(1, 6):
        try:
            new_exchange = get_exchange()
            log.info("Reconnection successful.")
            send_telegram(
                "🔌 <b>CONNECTION RESTORED</b>\n"
                "Exchange connection was lost but the bot reconnected successfully."
            )
            return new_exchange
        except Exception as e:
            log.error(f"Reconnect attempt {attempt}/5 failed: {e}")
            time.sleep(10 * attempt)

    send_telegram("🔴 <b>CRITICAL:</b> Failed to reconnect to Binance after 5 attempts.")
    return exchange


# ══════════════════════════════════════════════════════════════
# BTC TREND FILTER
# ══════════════════════════════════════════════════════════════

def get_btc_trend(exchange: ccxt.binance) -> str:
    """""
    Determines BTC market direction using 4H candles.

    Uses a triple filter:
      1. EMA crossover (EMA20 vs EMA50)
      2. RSI momentum confirmation
      3. Volume confirmation

    Why 4H instead of 1H?
      1H EMAs react too quickly and produce false trend signals in sideways markets.
      4H EMAs are slower and more reliable for determining the true market direction.

    Returns: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    """""
    bars = safe_fetch_ohlcv(exchange, 'BTC/USDT', timeframe=BTC_TREND_TF, limit=60)
    if bars is None or len(bars) < 55:
        log.warning("Insufficient BTC 4H data, returning NEUTRAL")
        return "NEUTRAL"

    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
    df['EMA_20']    = ta.ema(df['close'], length=20)
    df['EMA_50']    = ta.ema(df['close'], length=50)
    df['RSI_14']    = ta.rsi(df['close'], length=14)
    df['Vol_SMA20'] = ta.sma(df['volume'], length=20)

    last = df.iloc[-2]   # Last closed candle
    if any(pd.isna(last[c]) for c in ['EMA_20','EMA_50','RSI_14','Vol_SMA20']):
        return "NEUTRAL"

    volume_ok = last['volume'] > last['Vol_SMA20'] * 0.9

    if last['EMA_20'] > last['EMA_50'] and last['RSI_14'] > 52 and volume_ok:
        return "BULLISH"
    if last['EMA_20'] < last['EMA_50'] and last['RSI_14'] < 48 and volume_ok:
        return "BEARISH"
    return "NEUTRAL"


# ══════════════════════════════════════════════════════════════
# SIGNAL QUALITY SCORE
# ══════════════════════════════════════════════════════════════

def signal_score(vol_mult: float, rr: float, rsi: float, direction: str) -> tuple[int, str]:
    """
    Calculates a quality score (0-100) for a signal based on:
      - Volume strength   (max 40 pts)
      - Risk/Reward ratio (max 35 pts)
      - RSI in ideal zone (max 25 pts)

    Returns: (score, label)
    """
    score = 0

    # Volume contribution
    if vol_mult >= 3.0:   score += 40
    elif vol_mult >= 2.0: score += 30
    elif vol_mult >= 1.5: score += 20
    else:                 score += 10

    # Risk/Reward contribution
    if rr >= 3.0:   score += 35
    elif rr >= 2.5: score += 28
    elif rr >= 2.0: score += 20
    else:           score += 12

    # RSI ideal zone contribution
    if direction == "LONG":
        if 45 <= rsi <= 60:   score += 25
        elif 40 <= rsi < 45:  score += 15
        else:                  score += 8
    else:  # SHORT
        if 40 <= rsi <= 55:   score += 25
        elif 55 < rsi <= 60:  score += 15
        else:                  score += 8

    if score >= 75:   label = "STRONG"
    elif score >= 50: label = "MODERATE"
    else:             label = "WEAK"

    return score, label


def position_size_suggestion(score: int) -> str:
    """
    Suggests a position size based on signal quality.
    Assumes ~$300-400 balance with 2x leverage and max 5% risk per trade.
    """
    if score >= 75:
        return "💰 <b>Suggestion:</b> 4-5% of balance (~$15-20) | Leverage: 2x"
    elif score >= 50:
        return "💰 <b>Suggestion:</b> 2-3% of balance (~$8-12) | Leverage: 2x"
    else:
        return "💰 <b>Suggestion:</b> 1-2% of balance (~$4-8) | Leverage: 2x"


# ══════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════

def check_signal(symbol: str, btc_trend: str, exchange: ccxt.binance) -> dict | None:
    """
    Checks a symbol for a valid 1H hybrid signal.

    Entry conditions (LONG):
      - EMA20 > EMA50 (uptrend structure)
      - Bullish candle body
      - BTC 4H trend is BULLISH
      - RSI < 70 (not overbought)
      - Price within 4% of EMA50 and 2% of EMA20 (avoid late entries)
      - Volume > 1.3x average (1.2x for low liquidity coins)
      - Candle body < 3x ATR (filters out extreme spike candles)
      - R/R >= 1.5

    SHORT conditions are the mirror image of LONG.
    SL and TP are calculated dynamically using ATR.

    Returns a signal dict or None if no valid signal.
    """
    bars = safe_fetch_ohlcv(exchange, symbol, timeframe=SIGNAL_TF, limit=OHLCV_LIMIT)
    if bars is None or len(bars) < 60:
        return None

    df = pd.DataFrame(bars, columns=['timestamp','open','high','low','close','volume'])
    df['Vol_SMA_20'] = ta.sma(df['volume'], length=20)
    df['ATR']        = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['EMA_20']     = ta.ema(df['close'], length=20)
    df['EMA_50']     = ta.ema(df['close'], length=50)
    df['RSI_14']     = ta.rsi(df['close'], length=14)

    last = df.iloc[-2]   # Last fully closed candle

    required = ['ATR','EMA_20','EMA_50','RSI_14','Vol_SMA_20']
    if any(pd.isna(last[c]) for c in required):
        return None

    # Volume filter
    vol_sma = last['Vol_SMA_20']
    if vol_sma == 0:
        return None
    vol_mult = last['volume'] / vol_sma

    vol_threshold = 1.2 if symbol in LOW_LIQUIDITY_COINS else 1.3
    if vol_mult < vol_threshold:
        return None

    # Reject extreme volatility candles
    candle_body = abs(last['close'] - last['open'])
    if candle_body > (last['ATR'] * 3):
        return None

    atr = last['ATR']

    # ── LONG ───────────────────────────────────────────────────
    if (last['EMA_20'] > last['EMA_50']) and (last['close'] > last['open']):
        if btc_trend != "BULLISH":  return None
        if last['RSI_14'] >= 70:    return None

        dist_50 = (last['close'] - last['EMA_50']) / last['EMA_50']
        dist_20 = (last['close'] - last['EMA_20']) / last['EMA_20']
        if dist_50 >= 0.04 or dist_20 >= 0.02:
            return None

        sl     = last['close'] - (atr * 1.5)
        tp     = last['close'] + (atr * 2.5)
        risk   = last['close'] - sl
        reward = tp - last['close']
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < 1.5:
            return None

        return {
            "price": last['close'], "sl": sl, "tp": tp,
            "rr": rr, "vol": vol_mult,
            "direction": "LONG", "rsi": last['RSI_14']
        }

    # ── SHORT ──────────────────────────────────────────────────
    elif (last['EMA_20'] < last['EMA_50']) and (last['close'] < last['open']):
        if btc_trend != "BEARISH":  return None
        if last['RSI_14'] <= 30:    return None

        dist_50 = (last['EMA_50'] - last['close']) / last['EMA_50']
        dist_20 = (last['EMA_20'] - last['close']) / last['EMA_20']
        if dist_50 >= 0.04 or dist_20 >= 0.02:
            return None

        sl     = last['close'] + (atr * 1.5)
        tp     = last['close'] - (atr * 2.5)
        risk   = sl - last['close']
        reward = last['close'] - tp
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < 1.5:
            return None

        return {
            "price": last['close'], "sl": sl, "tp": tp,
            "rr": rr, "vol": vol_mult,
            "direction": "SHORT", "rsi": last['RSI_14']
        }

    return None


# ══════════════════════════════════════════════════════════════
# SCAN ENGINE
# ══════════════════════════════════════════════════════════════

def run_scan(exchange: ccxt.binance, btc_trend: str) -> ccxt.binance:
    """
    Scans all coins in WATCHLIST for valid signals.
    Signals are collected, sorted by quality score, then sent best-first.
    btc_trend is passed in to avoid a duplicate API call with the pulse check.
    """
    global daily_stats
    log.info(f"Scan started | BTC trend: {btc_trend}")
    daily_stats["scans"] += 1

    now_utc      = datetime.now(timezone.utc)
    low_vol_hour = 0 <= now_utc.hour < 6

    if low_vol_hour:
        log.info("Low volume hours (00-06 UTC) — signals may be less reliable.")

    # Step 1: Scan all coins and collect signals
    found_signals = []

    for coin in WATCHLIST:
        try:
            signal = check_signal(coin, btc_trend, exchange)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            log.error(f"[{coin}] Connection error during scan: {e}")
            exchange = reconnect_exchange(exchange)
            time.sleep(3)
            continue
        except Exception as e:
            log.error(f"[{coin}] Unexpected error during scan: {e}")
            continue

        if signal:
            score, label = signal_score(
                signal['vol'], signal['rr'], signal['rsi'], signal['direction']
            )
            found_signals.append({
                "coin":   coin,
                "signal": signal,
                "score":  score,
                "label":  label
            })
            log.info(
                f"Signal found: {coin} {signal['direction']} | "
                f"{label} ({score}/100) | R/R: {signal['rr']:.2f}"
            )

        time.sleep(0.5)

    # Step 2: Sort by score (highest first)
    found_signals.sort(key=lambda x: x["score"], reverse=True)
    total = len(found_signals)
    log.info(f"Scan complete | {total} signal(s) found")

    # Step 3: Send ranked summary if multiple signals
    if total > 1:
        rows = []
        for rank, item in enumerate(found_signals, 1):
            icon = '🟢' if item['signal']['direction'] == 'LONG' else '🔴'
            rows.append(
                f"{rank}. {icon} <b>#{item['coin'].split('/')[0]}</b>"
                f" — {item['label']} ({item['score']}/100)"
                f" | R/R 1:{item['signal']['rr']:.2f}"
            )

        send_telegram(
            f"📋 <b>SCAN COMPLETE — {total} SIGNALS FOUND</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Ranked best to worst:\n\n"
            + "\n".join(rows) +
            f"\n━━━━━━━━━━━━━━━━━━\n"
            f"<i>Details below. It is recommended to only enter #1.</i>"
        )
        time.sleep(1)

    # Step 4: Send detailed signal messages in ranked order
    for rank, item in enumerate(found_signals, 1):
        coin      = item["coin"]
        signal    = item["signal"]
        score     = item["score"]
        label     = item["label"]

        daily_stats["signals"] += 1
        if signal['direction'] == "LONG": daily_stats["long"]  += 1
        else:                             daily_stats["short"] += 1

        size_tip     = position_size_suggestion(score)
        dir_icon     = '🟢' if signal['direction'] == 'LONG' else '🔴'
        low_vol_note = "\n⚠️ <i>Note: Low volume hours (00-06 UTC) — trade with caution</i>" if low_vol_hour else ""

        if rank == 1:   rank_label = "🥇 <b>BEST SIGNAL</b>"
        elif rank == 2: rank_label = "🥈 <b>2ND SIGNAL</b>"
        elif rank == 3: rank_label = "🥉 <b>3RD SIGNAL</b>"
        else:           rank_label = f"#{rank} SIGNAL"

        send_telegram(
            f"{dir_icon} <b>A.R.E.S. SIGNAL</b> — {rank_label} {dir_icon}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💎 <b>Coin:</b> #{coin.split('/')[0]}\n"
            f"🧭 <b>Direction:</b> <b>{signal['direction']}</b>\n"
            f"👑 <b>BTC Trend:</b> {btc_trend}\n"
            f"🏆 <b>Signal Strength:</b> {label} ({score}/100)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Entry:</b> {signal['price']:.4f}\n"
            f"💸 <b>Take Profit:</b> {signal['tp']:.4f}\n"
            f"🛡️ <b>Stop Loss:</b> {signal['sl']:.4f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚖️ <b>Risk/Reward:</b> 1:{signal['rr']:.2f}\n"
            f"🌊 <b>Volume:</b> {signal['vol']:.1f}x\n"
            f"📈 <b>RSI:</b> {signal['rsi']:.1f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{size_tip}\n"
            f"⏱️ <b>Valid for:</b> ~30 minutes{low_vol_note}"
        )
        time.sleep(1)

    return exchange


# ══════════════════════════════════════════════════════════════
# DAILY SUMMARY
# ══════════════════════════════════════════════════════════════

def send_daily_summary():
    """Sends a daily performance summary and resets counters."""
    global daily_stats
    send_telegram(
        f"📊 <b>DAILY SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🔄 <b>Total Scans:</b> {daily_stats['scans']}\n"
        f"📡 <b>Signals Generated:</b> {daily_stats['signals']}\n"
        f"🟢 <b>Long Signals:</b> {daily_stats['long']}\n"
        f"🔴 <b>Short Signals:</b> {daily_stats['short']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<i>See you tomorrow.</i>"
    )
    daily_stats = {"scans": 0, "signals": 0, "long": 0, "short": 0}


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

def main():
    log.info("A.R.E.S. starting...")
    exchange = get_exchange()

    send_telegram(
        "🦅 <b>A.R.E.S. ONLINE</b> 🦅\n\n"
        "✅ <b>Signal Ranking:</b> Best to worst\n"
        "✅ <b>BTC Filter:</b> 4H trend (reduces false signals)\n"
        "✅ <b>Signal Score:</b> Strong / Moderate / Weak\n"
        "✅ <b>Position Sizing:</b> Auto-suggested per signal\n"
        "✅ <b>Daily Summary:</b> Every day at 22:00 UTC\n\n"
        "⏱️ <b>Scan interval:</b> Every hour at XX:00:15 UTC\n"
        "<i>Bot is live. Waiting for signals.</i>"
    )

    last_scan_hour   = -1
    last_pulse_hour  = -1
    last_summary_day = -1

    while True:
        try:
            now = datetime.now(timezone.utc)

            pulse_due = (now.hour % 4 == 0 and
                         now.minute == 0 and
                         last_pulse_hour != now.hour)

            scan_due  = (now.minute == 0 and
                         now.second >= 15 and
                         last_scan_hour != now.hour)

            if scan_due:
                last_scan_hour = now.hour
                btc_trend      = get_btc_trend(exchange)

                # Send status update if pulse also falls on this hour
                if pulse_due:
                    last_pulse_hour = now.hour
                    status_icon = "🟢" if btc_trend == "BULLISH" else "🔴" if btc_trend == "BEARISH" else "⚪"
                    send_telegram(
                        f"📡 <b>STATUS REPORT</b>\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"⚙️ <b>Server:</b> Running\n"
                        f"👑 <b>BTC Trend:</b> {status_icon} <b>{btc_trend}</b>\n"
                        f"⏰ <b>Time:</b> {now.strftime('%H:%M')} UTC\n"
                        f"<i>No confirmation, no trade.</i>"
                    )

                exchange = run_scan(exchange, btc_trend)
                gc.collect()

                remaining = 60 - datetime.now(timezone.utc).second
                time.sleep(max(remaining, 5))

            elif pulse_due:
                last_pulse_hour = now.hour
                btc_trend       = get_btc_trend(exchange)
                status_icon     = "🟢" if btc_trend == "BULLISH" else "🔴" if btc_trend == "BEARISH" else "⚪"
                send_telegram(
                    f"📡 <b>STATUS REPORT</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"⚙️ <b>Server:</b> Running\n"
                    f"👑 <b>BTC Trend:</b> {status_icon} <b>{btc_trend}</b>\n"
                    f"⏰ <b>Time:</b> {now.strftime('%H:%M')} UTC\n"
                    f"<i>No confirmation, no trade.</i>"
                )

            elif (now.hour == 22 and now.minute == 0 and last_summary_day != now.day):
                last_summary_day = now.day
                send_daily_summary()

            else:
                time.sleep(2)

        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            log.error(f"Main loop network error: {e}")
            exchange = reconnect_exchange(exchange)

        except KeyboardInterrupt:
            log.info("Stopped by user.")
            send_telegram("🛑 <b>A.R.E.S. stopped.</b>")
            break

        except Exception as e:
            log.error(f"Main loop unexpected error: {e}")
            send_telegram(f"⚠️ <b>System Error:</b> {str(e)[:200]}")
            time.sleep(10)


if __name__ == "__main__":
    main()
