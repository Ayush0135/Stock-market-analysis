from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import yfinance as yf
import requests
import os
import numpy as np
import pandas as pd
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime, timedelta
from backtest import Backtest, sma_crossover_strategy, rsi_strategy

# Load environment variables FIRST (before any os.getenv calls)
load_dotenv()

# Alpaca SDK import and client configuration
from alpaca_trade_api import REST

# Load Alpaca credentials from environment
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

# Lazy-load Alpaca REST client to avoid startup errors when credentials are missing
_alpaca_client = None

def get_alpaca_client():
    """Get or create the Alpaca REST client (lazy initialization)."""
    global _alpaca_client
    if _alpaca_client is None:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("Alpaca API credentials not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        _alpaca_client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL, api_version='v2')
    return _alpaca_client


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


def place_order(symbol: str, qty: int, side: str, order_type: str = "market", time_in_force: str = "gtc"):
    """Submit an order to Alpaca.
    Returns the raw JSON response from Alpaca.
    """
    try:
        alpaca = get_alpaca_client()
        order = alpaca.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
        )
        return order._raw
    except Exception as e:
        print(f"[Alpaca] Order error: {e}")
        raise

@app.route("/api/alpaca_order", methods=["POST"])
def api_alpaca_order():
    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip().upper()
    qty = data.get("qty")
    side = data.get("side", "buy").strip().lower()
    order_type = data.get("type", "market")
    time_in_force = data.get("time_in_force", "gtc")
    if not symbol or not qty:
        return jsonify({"error": "symbol and qty required"}), 400
    try:
        resp = place_order(symbol, int(qty), side, order_type, time_in_force)
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Helper: Convert numpy types to native Python types ----------
def to_native(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    return x


# ---------- Backtesting API ----------
from backtest import Backtest, sma_crossover_strategy, rsi_strategy

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """
    Run a backtest on historical data.
    
    JSON payload:
        symbol: Stock symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: Strategy name ('sma_crossover' or 'rsi')
        initial_capital: Starting capital (default 10000)
        params: Optional strategy parameters
    """
    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip().upper()
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    strategy_name = data.get("strategy", "sma_crossover")
    initial_capital = float(data.get("initial_capital", 10000))
    params = data.get("params", {})
    
    if not symbol or not start_date or not end_date:
        return jsonify({"error": "symbol, start_date, and end_date are required"}), 400
    
    try:
        # Pass global credentials explicitly to ensure they are used
        bt = Backtest(symbol, start_date, end_date, api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
        
        # Select strategy
        if strategy_name == "sma_crossover":
            short_period = params.get("short_period", 10)
            long_period = params.get("long_period", 30)
            strategy = sma_crossover_strategy(short_period, long_period)
        elif strategy_name == "rsi":
            period = params.get("period", 14)
            oversold = params.get("oversold", 30)
            overbought = params.get("overbought", 70)
            strategy = rsi_strategy(period, oversold, overbought)
        else:
            return jsonify({"error": f"Unknown strategy: {strategy_name}"}), 400
        
        results = bt.run(strategy, initial_capital=initial_capital)
        
        # Convert timestamps for JSON serialization
        for trade in results.get('trades', []):
            if 'timestamp' in trade:
                trade['timestamp'] = str(trade['timestamp'])
        
        # Summarize portfolio history (too large to return fully)
        # Summarize portfolio history (too large to return fully)
        # But we do want to return simpler series for plotting
        history = results.get('portfolio_history', [])
        # Sample or just return as is (assuming it's daily data for 1 year ~250 points, which is fine)
        results['portfolio_history'] = history
        
        results['portfolio_summary'] = {
            'data_points': len(history),
            'start_value': history[0]['portfolio_value'] if history else 0,
            'end_value': history[-1]['portfolio_value'] if history else 0,
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/execute_strategy", methods=["POST"])
def api_execute_strategy():
    """
    Execute a strategy on live/latest data and place orders if signal detected.
    Paper Trading Environment recommended.
    """
    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip().upper()
    strategy_name = data.get("strategy", "sma_crossover")
    params = data.get("params", {})
    
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400

    try:
        # Load Strategy
        if strategy_name == "sma_crossover":
            short_period = params.get("short_period", 10)
            long_period = params.get("long_period", 30)
            strategy = sma_crossover_strategy(short_period, long_period)
            lookback_days = long_period * 3 # sufficient history
        elif strategy_name == "rsi":
            period = params.get("period", 14)
            oversold = params.get("oversold", 30)
            overbought = params.get("overbought", 70)
            strategy = rsi_strategy(period, oversold, overbought)
            lookback_days = period * 4
        else:
            return jsonify({"error": f"Unknown strategy: {strategy_name}"}), 400

        # Fetch recent data
        # Using Backtest class just to fetch data is cleaner than re-implementing get_bars
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Format dates as strings
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Initialize backtest helper with explicit credentials to fetch data
        # We won't run full backtest, just use it to get data
        bt = Backtest(symbol, start_str, end_str, api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)
        df = bt.fetch_data()
        
        if df.empty:
            return jsonify({"message": "No data available", "action": "none"}), 200

        # Check latest signal
        # Strategies typically need 'positions' list, but for single execution we can check 
        # current Alpaca position for this symbol
        
        alpaca = get_alpaca_client()
        current_position_qty = 0
        try:
            pos = alpaca.get_position(symbol)
            current_position_qty = float(pos.qty)
        except:
            # likely 404 if no position
            current_position_qty = 0

        # Mock positions list for strategy function: [(qty, avg_entry_price)]
        # We don't have exact entry price for all lots easily, but for signal generation 
        # (check if we hold it) the strategy mostly cares if list is empty or not
        # strategies implemented: 'if not positions' (buy) or 'if positions' (sell)
        fake_positions = [(current_position_qty, 0)] if current_position_qty > 0 else []
        
        # Run strategy on the LAST complete bar
        # Use -1 index
        i = len(df) - 1
        signal = strategy(df, i, fake_positions)
        
        if signal:
            action, qty = signal
            
            # Place Order!
            # Use 'qty' from strategy or override
            # For simplicity, let's fix quantity to 1 or 10 if not specified to avoid draining account
            # The strategy returns qty, let's use it but cap it if needed
            
            # Execute
            order = alpaca.submit_order(
                symbol=symbol,
                qty=qty,
                side=action,
                type='market',
                time_in_force='gtc'
            )
            
            return jsonify({
                "message": f"Signal detected! {action.upper()} order placed.",
                "signal": action,
                "order_id": order.id,
                "price": df['close'].iloc[-1]
            })
        else:
            return jsonify({
                "message": "No signal detected at this time.",
                "signal": "none",
                "price": df['close'].iloc[-1]
            })

    except Exception as e:
        print(f"Live Execution Error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- Login Decorator ----------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first!", "warning")
            return redirect(url_for("home_page"))
        return f(*args, **kwargs)
    return wrapper

# ---------- Search Ticker ----------
@app.route("/api/search_ticker")
def api_search_ticker():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query required"}), 400

    try:
        
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        
        candidates = []
        if 'quotes' in data:
            for q in data['quotes']:
                sym = q.get('symbol')
                name = q.get('shortname') or q.get('longname') or sym
                exch = q.get('exchange')
                # Prioritize NSE/BSE for Indian context if user implies that
                score = 0
                if '.NS' in sym or 'NSE' in exch: score += 2
                if '.BO' in sym or 'BSE' in exch: score += 1
                
                candidates.append({
                    "symbol": sym,
                    "name": name,
                    "exchange": exch,
                    "score": score
                })
        
        # FALLBACK: If API returns nothing, check if Query + .NS is valid
        if not candidates:
            try:
                test_sym = f"{query.upper()}.NS"
                t = yf.Ticker(test_sym)
                # Fast check using fast_info
                if t.fast_info.last_price:
                    candidates.append({
                        "symbol": test_sym,
                        "name": f"{query.upper()} (NSE)",
                        "exchange": "NSE",
                        "score": 10
                    })
            except:
                pass

        # Sort by relevance to India context
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({"candidates": candidates})
        
    except Exception as e:
        print("Search error:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Intraday Chart ----------
@app.route("/api/intraday_data")
def api_intraday_data():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m")
        if df.empty:
            return jsonify({"error": "No intraday data found"}), 404

        df = df.reset_index()
        
        # Format timestamps for better display
        timestamps = df["Datetime"].dt.strftime('%H:%M').tolist()
        
        # OHLC data for candlestick charts
        opens = [to_native(x) for x in df["Open"].tolist()]
        highs = [to_native(x) for x in df["High"].tolist()]
        lows = [to_native(x) for x in df["Low"].tolist()]
        closes = [to_native(x) for x in df["Close"].tolist()]
        volumes = [to_native(x) for x in df["Volume"].tolist()]
        
        # Moving averages
        sma20 = df["Close"].rolling(20, min_periods=1).mean().tolist()
        sma50 = df["Close"].rolling(50, min_periods=1).mean().tolist()

        return jsonify({
            "timestamps": timestamps,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes,
            "sma20": [to_native(x) for x in sma20],
            "sma50": [to_native(x) for x in sma50],
        })
    except Exception as e:
        print("Error in /api/intraday_data:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Live Price (native currency, no INR conversion) ----------
@app.route("/api/live_price")
def api_live_price():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev_close = info.get("previousClose")

        if price is None:
            hist = t.history(period="1d", interval="1m")
            if hist.empty:
                return jsonify({"error": "No live data"}), 404
            price = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[0]) if len(hist) > 1 else price

        change = round(price - (prev_close or price), 2)
        change_percent = round(((price - (prev_close or price)) / (prev_close or price)) * 100, 2)

        return jsonify({
            "ticker": ticker,
            "price": round(price, 2),
            "change": change,
            "change_percent": change_percent,
            "currency": info.get("currency", "USD")
        })
    except Exception as e:
        print("Error in /api/live_price:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Company Info ----------
@app.route("/api/info")
def api_info():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Fallback for missing info
        if not info:
            fast = t.fast_info
            info = {
                "symbol": ticker,
                "currency": fast.get("currency"),
                "last_price": fast.get("last_price"),
                "year_high": fast.get("year_high"),
                "year_low": fast.get("year_low"),
            }

        data = {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "symbol": info.get("symbol", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "No company overview available."),
            "market_cap": to_native(info.get("marketCap")) if info.get("marketCap") else None,
            "pe_ratio": info.get("trailingPE"),
            "website": info.get("website", ""),
            "currency": info.get("currency", "USD"),
        }
        return jsonify(data)

    except Exception as e:
        print("Error in /api/info:", e)
        return jsonify({"error": str(e)}), 500



# ---------- Batch Price API ----------
@app.route("/api/batch_price")
def api_batch_price():
    tickers_param = request.args.get("tickers", "").strip()
    if not tickers_param:
        return jsonify({"data": []})
    
    ticker_list = [t.strip().upper() for t in tickers_param.split(',') if t.strip()]
    if not ticker_list:
        return jsonify({"data": []})

    results = []
    try:
        # Use yf.Tickers for batch fetching
        tickers_obj = yf.Tickers(" ".join(ticker_list))
        
        for sym in ticker_list:
            try:
            
                t = tickers_obj.tickers[sym]
                fast = t.fast_info
                price = fast.last_price
                prev = fast.previous_close
                if price:
                    change_pct = ((price - prev)/prev)*100 if prev else 0.0
                    results.append({
                        "symbol": sym,
                        "price": round(price, 2),
                        "change": round(price - prev, 2) if prev else 0,
                        "change_percent": round(change_pct, 2)
                    })
                else:
                    # Fallback if fast_info fails (e.g. invalid ticker)
                    results.append({"symbol": sym, "error": "No data"})
            except:
                results.append({"symbol": sym, "error": "Fetch failed"})
                
        return jsonify({"data": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- 5-Year Analysis ----------
@app.route("/api/stock_analysis")
def stock_analysis():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker missing"}), 400

    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return jsonify({"error": "No historical data found"}), 404

        # Handle MultiIndex columns (common in newer yfinance versions)
        # logic: if df["Close"] returns a DataFrame (because of MultiIndex with Ticker), extract the single column
        close_data = df["Close"]
        if isinstance(close_data, pd.DataFrame):
            close_prices = close_data.iloc[:, 0]
        else:
            close_prices = close_data

        close_prices = close_prices.dropna().astype(float)

        avg_price = float(np.mean(close_prices))
        min_price = float(np.min(close_prices))
        max_price = float(np.max(close_prices))

        # Handle potential numpy types or Series for single item access
        first_price = float(close_prices.iloc[0])
        last_price = float(close_prices.iloc[-1])

        change_percent = float(((last_price - first_price) / first_price) * 100)

        t_obj = yf.Ticker(ticker)
        info = t_obj.info or {}

        # Calculate Returns & Performance (1W, 1M, 3M, 6M, 1Y)
        current_price = float(close_prices.iloc[-1])
        
        def safe_return(days):
            if len(close_prices) > days:
                past = float(close_prices.iloc[-days])
                return ((current_price - past) / past) * 100
            return 0.0

        performance = {
            "1W": round(safe_return(5), 2),
            "1M": round(safe_return(21), 2),
            "3M": round(safe_return(63), 2),
            "6M": round(safe_return(126), 2),
            "1Y": round(safe_return(252), 2),
            "YTD": info.get('ytdReturn') or 0.0 
        }

        # Technical Indicators calculation
        # Simple RSI (14) using close_prices (Series)
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = round(rsi.iloc[-1], 2) if not rsi.empty else 50.0

        # Moving Averages
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
        sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
        trend = "Bullish" if float(sma_50) > float(sma_200) else "Bearish"
        
        technicals = {
            "rsi": current_rsi,
            "sma_50": round(float(sma_50), 2) if not pd.isna(sma_50) else "N/A",
            "sma_200": round(float(sma_200), 2) if not pd.isna(sma_200) else "N/A",
            "trend": trend,
            "signal": "Buy" if current_rsi < 30 else "Sell" if current_rsi > 70 else "Hold"
        }
        
        # Financials / Analyst Targets
        fundamentals = {
            "target_mean": info.get('targetMeanPrice'),
            "recommendation": info.get('recommendationKey', 'none').replace('_', ' ').title(),
            "pe_ratio": info.get('trailingPE'),
            "pb_ratio": info.get('priceToBook'),
            "earnings_growth": info.get('earningsQuarterlyGrowth'),
            "dividend_yield": info.get('dividendYield')
        }

        data = {
            "symbol": ticker,
            "currency": info.get("currency", "USD"),
            "dates": [d.strftime("%Y-%m-%d") for d in close_prices.index],
            "prices": [round(val, 2) for val in close_prices.tolist()],
            "avg_price": round(float(np.mean(close_prices)), 2),
            "min_price": round(float(np.min(close_prices)), 2),
            "max_price": round(float(np.max(close_prices)), 2),
            "first_price": round(float(close_prices.iloc[0]), 2),
            "last_price": round(float(close_prices.iloc[-1]), 2),
            "performance": performance,
            "technicals": technicals,
            "fundamentals": fundamentals
        }

        return jsonify(data)
    except Exception as e:
        print("Error in /api/stock_analysis:", e)
        return jsonify({"error": str(e)}), 500



# ---------- AI Recommendation ----------
@app.route("/api/ai_analysis", methods=["POST"])
def api_ai_analysis():
    data = request.get_json() or {}
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400
    try:
        info = yf.Ticker(ticker).info
        prompt = f"""
        Analyze {info.get('longName', ticker)} ({ticker}):
        - Current price: {info.get('regularMarketPrice')}
        - 52W High: {info.get('fiftyTwoWeekHigh')}
        - 52W Low: {info.get('fiftyTwoWeekLow')} 
        - Market Cap: {info.get('marketCap')}
        - PE Ratio: {info.get('trailingPE')}
        - Sector: {info.get('sector')}
        
        Provide a structured analysis with the following sections:
        
        1. Market Sentiment: State if Bullish, Bearish, or Neutral
        2. Short-term Outlook: Brief outlook for next 1-3 months
        3. Long-term Outlook: Brief outlook for next 6-12 months
        4. Recommendation: State BUY, HOLD, or SELL with confidence percentage (e.g., BUY with 75% confidence)
        5. Key Points: List 2-3 important considerations
        
        Keep each section concise and clear.
        """

        if not GROQ_API_KEY:
            return jsonify({"ai_analysis": "AI key missing. Add GROQ_API_KEY to .env."}), 200

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a professional financial analyst providing structured stock analysis. Be concise and use clear section headers."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 400
        }
        r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload)
        text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return jsonify({"ai_analysis": text})
    except Exception as e:
        print("AI Error:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Stock News ----------
@app.route("/api/stock_news")
def api_stock_news():
    ticker = request.args.get("ticker", "").strip().upper()
    
    try:
        # If no ticker, return general market news
        if not ticker:
            articles = []
            if MARKETAUX_API_KEY:
                # General market news
                url = (
                    "https://api.marketaux.com/v1/news/all"
                    f"?language=en&limit=10&api_token={MARKETAUX_API_KEY}"
                )
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        data = r.json()
                        items = data.get("data", [])
                        for it in items:
                            articles.append({
                                "title": it.get("title"),
                                "snippet": it.get("description") or it.get("snippet", ""),
                                "url": it.get("url"),
                                "source": it.get("source", "Unknown"),
                                "published_at": it.get("published_at", "")
                            })
                except Exception as e:
                    print(f"MarketAux error: {e}")

            elif GOOGLE_API_KEY and GOOGLE_CSE_ID:
                q = "stock market news"
                url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={requests.utils.quote(q)}"
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        data = r.json()
                        items = data.get("items", [])
                        for i in items:
                            articles.append({
                                "title": i.get("title"),
                                "snippet": i.get("snippet"),
                                "url": i.get("link"),
                                "source": i.get("displayLink", "Unknown"),
                                "published_at": ""
                            })
                except Exception as e:
                    print(f"Google CSE error: {e}")

            if not articles:
                # Fallback to yfinance generally popular tickers news
                tickers = ["SPY", "AAPL", "MSFT", "GOOGL"]
                for t_sym in tickers:
                    try:
                        t = yf.Ticker(t_sym)
                        yf_news = t.news if hasattr(t, "news") else []
                        for it in yf_news[:2]:
                             if isinstance(it, dict):
                                articles.append({
                                    "title": it.get("title"),
                                    "snippet": it.get("summary"),
                                    "url": it.get("link"),
                                    "source": str(it.get("publisher", "Unknown")),
                                    "published_at": it.get("providerPublishTime", "")
                                })
                    except: continue
            
            return jsonify({"articles": articles})

        # Get company info to use company name in search
        company_info = yf.Ticker(ticker).info or {}
        company_name = company_info.get("longName") or company_info.get("shortName") or ticker
        
        # Prefer MarketAux if key available - it filters by ticker symbol
        if MARKETAUX_API_KEY:
            url = (
                "https://api.marketaux.com/v1/news/all"
                f"?symbols={ticker}&filter_entities=true&language=en&limit=10&api_token={MARKETAUX_API_KEY}"
            )
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                items = data.get("data", [])
                articles = []
                for it in items:
                    src = it.get("source")
                    if isinstance(src, dict):
                        source_name = src.get("domain") or src.get("name") or "Unknown"
                    elif isinstance(src, str):
                        source_name = src
                    else:
                        source_name = "Unknown"
                    
                    articles.append({
                        "title": it.get("title"),
                        "snippet": it.get("description") or it.get("snippet", ""),
                        "url": it.get("url"),
                        "source": source_name,
                        "published_at": it.get("published_at", "")
                    })
                
                if articles:
                    return jsonify({"articles": articles})
            else:
                print("MarketAux returned", r.status_code, r.text[:200])

        # Fallback to Google Custom Search with company name
        if GOOGLE_API_KEY and GOOGLE_CSE_ID:
            # Use company name for better results
            q = f'"{company_name}" stock news'
            url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={requests.utils.quote(q)}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                items = data.get("items", [])
                articles = []
                for i in items:
                    articles.append({
                        "title": i.get("title"),
                        "snippet": i.get("snippet"),
                        "url": i.get("link"),
                        "source": i.get("displayLink", "Unknown"),
                        "published_at": ""
                    })
                
                if articles:
                    return jsonify({"articles": articles})
            else:
                print("Google CSE returned", r.status_code, r.text[:200])

        # Last-resort: use yfinance .news if present
        try:
            t = yf.Ticker(ticker)
            yf_news = t.news if hasattr(t, "news") else []
            articles = []
            for it in yf_news[:10]:
                if isinstance(it, dict):
                    articles.append({
                        "title": it.get("title"),
                        "snippet": it.get("summary") or it.get("description", ""),
                        "url": it.get("link"),
                        "source": str(it.get("publisher", "Unknown")),
                        "published_at": it.get("providerPublishTime", "")
                    })
            
            if articles:
                return jsonify({"articles": articles})
        except Exception as e:
            print("yfinance news error:", e)

        return jsonify({"articles": []})
    except Exception as e:
        print("Error in stock_news:", e)
        return jsonify({"error": str(e)}), 500



# ---------- ETF Info ----------
# ---------- ETF Info ----------
@app.route("/api/etf")
def api_etf():
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400

    try:
        etf = yf.Ticker(symbol)
        fast = getattr(etf, "fast_info", {}) or {}
        info = getattr(etf, "info", {}) or {}

        hist = etf.history(period="6mo", interval="1d")
        latest_price = None
        if not hist.empty:
            latest_price = float(hist["Close"].iloc[-1])

        price = info.get("regularMarketPrice") or fast.get("last_price") or latest_price
        prev_close = info.get("previousClose") or fast.get("previous_close") or (
            float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        )

        if not price:
            return jsonify({"error": "No ETF data found"}), 404

        change = round(price - (prev_close or price), 2)
        change_percent = round(((price - (prev_close or price)) / (prev_close or price)) * 100, 2)

        data = {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "price": round(price, 2),
            "change": change,
            "change_percent": change_percent,
            "category": info.get("category", "-"),
            "expense_ratio": info.get("annualReportExpenseRatio"),
            "total_assets": info.get("totalAssets"),
            "currency": info.get("currency", fast.get("currency", "USD")),
            "ytd_return": info.get("ytdReturn") or "-",
            "three_year_return": info.get("threeYearAverageReturn") or "-",
            "five_year_return": info.get("fiveYearAverageReturn") or "-",
            "website": info.get("website"),
        }

        # Generate AI Analysis for ETF
        if GROQ_API_KEY:
            prompt = f"""
            Analyze ETF {data['name']} ({symbol}):
            - Price: {data['price']}
            - Expense Ratio: {data['expense_ratio']}
            - Category: {data['category']}
            - 3Y Return: {data['three_year_return']}
            
            Provide a concise 3-sentence investment insight (Risk, Performance, Recommendation).
            """
            try:
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": "You are a financial expert."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 150
                }
                r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=5)
                data["ai_analysis"] = r.json().get("choices", [{}])[0].get("message", {}).get("content", "AI analysis unavailable.")
            except:
                data["ai_analysis"] = "AI unavailable at the moment."
        else:
            data["ai_analysis"] = "Configure GROQ_API_KEY for AI insights."

        return jsonify(data)
    except Exception as e:
        print("Error in /api/etf:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Crypto Info ----------
@app.route("/api/crypto")
def api_crypto():
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    
    # Auto-append -USD if missing (common for crypto in yfinance)
    if not symbol.endswith("-USD") and "-" not in symbol:
        symbol += "-USD"

    try:
        coin = yf.Ticker(symbol)
        fast = getattr(coin, "fast_info", {}) or {}
        
        price = fast.get("last_price")
        if not price:
             # Fallback to history
             hist = coin.history(period="1d")
             if not hist.empty:
                 price = hist["Close"].iloc[-1]
             else:
                 return jsonify({"error": "Crypto data not found"}), 404

        prev_close = fast.get("previous_close") or price
        change = price - prev_close
        change_pct = (change / prev_close) * 100
        
        data = {
            "symbol": symbol,
            "name": coin.info.get("name") or symbol,
            "price": round(price, 4) if price < 1 else round(price, 2),
            "change": round(change, 4) if price < 1 else round(change, 2),
            "change_percent": round(change_pct, 2),
            "market_cap": coin.info.get("marketCap", "N/A"),
            "volume": coin.info.get("volume24Hr", "N/A"),
            "high_24h": fast.get("day_high", "N/A"),
            "low_24h": fast.get("day_low", "N/A"),
        }
        
        # AI Analysis for Crypto
        if GROQ_API_KEY:
            prompt = f"""
            Analyze Crypto {data['name']} ({symbol}):
            - Price: {data['price']}
            - 24h Change: {data['change_percent']}%
            
            Provide a concise, caution-aware investment insight (Volatility, Trend, Outlook).
            """
            try:
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": "You are a crypto analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.6,
                    "max_tokens": 150
                }
                r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=5)
                data["ai_analysis"] = r.json().get("choices", [{}])[0].get("message", {}).get("content", "AI analysis unavailable.")
            except:
                data["ai_analysis"] = "AI unavailable at the moment."
        else:
            data["ai_analysis"] = "Configure GROQ_API_KEY for AI insights."
            
        return jsonify(data)

    except Exception as e:
        print("Error in /api/crypto:", e)
        return jsonify({"error": str(e)}), 500

# ---------- Top ETFs List ----------
@app.route('/markets')
def markets_page():
    return render_template("markets.html")

# ---------- MARKET DASHBOARD UTILS ----------
# A representative list of Nifty 50 stocks for Top Movers calculation
NIFTY_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LICI.NS", "LT.NS", "TATAMOTORS.NS", "AXISBANK.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "MARUTI.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "WIPRO.NS", "HCLTECH.NS", "ADANIENT.NS", "M&M.NS", "TATASTEEL.NS",
    "NTPC.NS", "POWERGRID.NS", "INDUSINDBK.NS", "ONGC.NS", "NESTLEIND.NS",
    "TECHM.NS", "GRASIM.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "HINDALCO.NS",
    "TATACONSUM.NS", "SBILIFE.NS", "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS",
    "COALINDIA.NS", "APOLLOHOSP.NS", "EICHERMOT.NS", "DIVISLAB.NS", "BAJAJ-AUTO.NS"
]

@app.route('/api/market_overview')
def api_market_overview():
    # Major Indices
    indices = ["^NSEI", "^BSESN", "^NSEBANK", "^GSPC", "^IXIC", "^DJI", "^FTSE", "^N225"]
    # INDIAVIX often needs special handling or different ticker like ^INDIAVIX
    # yfinance sometimes struggles with VIX, but we try ^INDIAVIX
    
    results = []
    for symbol in indices + ["^INDIAVIX"]:
        try:
            t = yf.Ticker(symbol)
            fast = t.fast_info
            
            # fast_info keys occasionally vary, safely get price
            price = fast.last_price
            prev_close = fast.previous_close
            
            if price and prev_close:
                change = price - prev_close
                pct = (change / prev_close) * 100
                results.append({
                    "symbol": symbol,
                    "price": price,
                    "change": change,
                    "percent": pct
                })
        except Exception as e:
            print(f"Index fetch error {symbol}: {e}")
            
    return jsonify({"indices": results})

@app.route('/api/top_movers')
def api_top_movers():
    # Efficiently fetch batch data
    try:
        data = []
        # We process stocks one by one or in small batches if download is too heavy
        # For responsiveness, looping with fast_info is often faster than 
        # downloading 2 days of history for 50 stocks sequentially if connection is good,
        # but yf.Tickers is better.
        
        tickers = yf.Tickers(" ".join(NIFTY_STOCKS))
        
        for symbol in NIFTY_STOCKS:
            try:
                # Access ticker from the Tickers object
                t = tickers.tickers[symbol]
                fast = t.fast_info
                price = fast.last_price
                prev = fast.previous_close
                
                if price and prev:
                    pct = ((price - prev) / prev) * 100
                    data.append({
                        "symbol": symbol.replace('.NS', ''),
                        "price": price,
                        "change": price - prev,
                        "percent": pct
                    })
            except: continue

        # Sort by percent change
        data.sort(key=lambda x: x['percent'], reverse=True)
        
        top_gainers = data[:5]
        top_losers = data[-5:]
        top_losers.reverse() # Show worst first
        
        return jsonify({
            "gainers": top_gainers,
            "losers": top_losers
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/sector_performance')
def api_sector_performance():
    # Simulated approximation using representative stocks/indices 
    # (Since direct sector indices are tricky)
    sectors = [
        {"name": "Bank Nifty", "ticker": "^NSEBANK"},
        {"name": "Nifty IT", "ticker": "TCS.NS"}, # Proxy
        {"name": "Auto", "ticker": "MARUTI.NS"}, # Proxy
        {"name": "Pharma", "ticker": "SUNPHARMA.NS"}, # Proxy
        {"name": "FMCG", "ticker": "ITC.NS"}, # Proxy
        {"name": "Metals", "ticker": "TATASTEEL.NS"}, # Proxy
        {"name": "Realty", "ticker": "DLF.NS"}, 
        {"name": "Energy", "ticker": "RELIANCE.NS"} # Proxy
    ]
    
    results = []
    for s in sectors:
        try:
            t = yf.Ticker(s['ticker'])
            fast = t.fast_info
            pct = ((fast.last_price - fast.previous_close) / fast.previous_close) * 100
            results.append({
                "name": s['name'],
                "percent": pct
            })
        except:
             results.append({"name": s['name'], "percent": 0.0})
             
    # Sort worst to best or arbitrary? Let's sort best to worst
    results.sort(key=lambda x: x['percent'], reverse=True)
    return jsonify({"sectors": results})

# ---------- Top ETFs List ----------
@app.route("/api/etf_list")
def api_etf_list():
    top_etfs = [
        "VOO", "SPY", "QQQ", "IWM", "VTI", 
        "XLK", "XLF", "ARKK", "EFA", "VWO"
    ]

    results = []
    for sym in top_etfs:
        try:
            etf = yf.Ticker(sym)
            fast = getattr(etf, "fast_info", {}) or {}
            price = fast.get("last_price")
            prev = fast.get("previous_close") or price
            if not price:
                continue

            change = round(price - prev, 2)
            change_percent = round(((price - prev) / prev) * 100, 2)

            results.append({
                "symbol": sym,
                "name": etf.info.get("shortName") or sym,
                "price": round(price, 2),
                "change": change,
                "change_percent": change_percent,
                "currency": fast.get("currency", "USD"),
            })
        except Exception as e:
            print(f"ETF list error for {sym}:", e)
            continue

    return jsonify({"etfs": results})




# ---------- Frontend Pages ----------
@app.route('/')
def home_page():
    return render_template("home.html")

@app.route('/analytics')
def analytics_page():
    ticker = request.args.get("ticker", "")
    return render_template("analytics.html", ticker=ticker)

@app.route('/etfs')
def etfs_page():
    return render_template("etfs.html")

@app.route('/crypto')
def crypto_page():
    return render_template("crypto.html")

@app.route('/news')
def news_page():
    return render_template("news.html")

@app.route('/settings')
def settings_page():
    return render_template("settings.html")

@app.route('/backtest')
def backtest_page():
    return render_template("backtest.html")

if __name__ == "__main__":
    app.run(debug=True)
