# A.R.E.S. — Automated Reconnaissance & Entry System

A crypto futures signal bot for Binance that scans 24 major altcoin pairs every hour and sends ranked trading signals via Telegram.

## Strategy

- Scans 24 altcoin/USDT pairs on Binance Futures every hour
- Entry logic: 1H EMA crossover + RSI + volume confirmation
- Market filter: BTC 4H trend (only takes signals aligned with BTC direction)
- Signals are scored (0–100) and ranked best to worst before being sent

## Signal Conditions

**LONG:**
- EMA20 > EMA50 on 1H
- Bullish candle close
- BTC 4H trend is BULLISH
- RSI < 70
- Price within 4% of EMA50 and 2% of EMA20
- Volume > 1.3x 20-period average
- R/R ≥ 1.5

**SHORT:** Mirror conditions with BEARISH BTC filter.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/metehanulger/A.R.E.S-crypto-analysis-bot.git
cd A.R.E.S-crypto-analysis-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and fill in your Telegram bot token and chat ID.

### 4. Run
```bash
python ares.py
```

### Running on a server (AWS EC2)
```bash
screen -S ares
python ares.py
# Press Ctrl+A, D to detach
```

## Signal Output

Each signal includes:
- Coin and direction (LONG/SHORT)
- Entry price, Take Profit, Stop Loss (ATR-based)
- Risk/Reward ratio
- Signal quality score (Strong / Moderate / Weak)
- Position size suggestion
- BTC trend at signal time

## Disclaimer

This bot is for educational purposes only. Crypto trading involves significant risk. Past performance does not guarantee future results. Always use a stop loss and never risk more than you can afford to lose.
