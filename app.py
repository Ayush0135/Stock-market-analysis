from flask import Flask, render_template, request, jsonify, redirect, url_for
import yfinance as yf
import requests
from langchain_groq import ChatGroq
import sqlite3
import os
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from pydantic import SecretStr

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
DB = "database.db"

load_dotenv()


# ------------------ AI Model Setup ------------------
llm = ChatGroq(
    temperature=0.4,
    api_key=SecretStr(os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None,
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
)
# ------------------ Portfolio ------------------
portfolio = []

# ------------------ Helper Functions ------------------# ============ DATABASE SETUP ============


# Call this function when the app starts     

def get_intraday_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period='1d', interval='1m')
    if df.empty:
        return {'error': 'No intraday data found'}

    df.reset_index(inplace=True)
    df['Time'] = df['Datetime'].dt.strftime('%H:%M')

    closes = df['Close'].tolist()
    times = df['Time'].tolist()
    sma50 = df['Close'].rolling(window=50, min_periods=1).mean().tolist()
    sma200 = df['Close'].rolling(window=200, min_periods=1).mean().tolist()

    ref_price = closes[0]
    pnl = [round(c - ref_price, 2) for c in closes]
    pnl_pct = [round((c - ref_price)/ref_price*100, 2) for c in closes]

    return {
        'timestamps': times,
        'closes': closes,
        'sma50': sma50,
        'sma200': sma200,
        'pnl': pnl,
        'pnl_pct': pnl_pct
    }

def get_company_info(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    if not info:
        return {'error': 'No company info found'}

    hist = ticker.history(period='6mo', interval='1d')
    sma50 = hist['Close'].rolling(window=50).mean().iloc[-1] if not hist.empty else None
    sma200 = hist['Close'].rolling(window=200).mean().iloc[-1] if not hist.empty else None

    day_change_pct = None
    if 'previousClose' in info and 'currentPrice' in info and info['previousClose']:
        day_change_pct = round((info['currentPrice'] - info['previousClose']) / info['previousClose'] * 100, 2)

    return {
        'symbol': info.get('symbol', ticker_symbol),
        'shortName': info.get('shortName'),
        'longName': info.get('longName'),
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'marketCap': info.get('marketCap'),
        'currentPrice': info.get('currentPrice'),
        'trailingPE': info.get('trailingPE'),
        'sma50': round(sma50, 2) if sma50 else None,
        'sma200': round(sma200, 2) if sma200 else None,
        'dayChangePct': day_change_pct,
        'website': info.get('website'),
        'longBusinessSummary': info.get('longBusinessSummary')
    }

def get_ai_recommendation(info):
    prompt = f"""
    You are a professional stock analyst.
    Analyze this company's data and suggest whether it is a BUY, HOLD, or SELL:
    {info}

    Include:
    - A clear decision (BUY / SELL / HOLD)
    - 2-3 sentences explaining why
    - Confidence level as a percentage
    """
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

# ============ LOGIN REQUIRED DECORATOR ============
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first!", "warning")
            return redirect(url_for("auth"))
        return f(*args, **kwargs)
    return decorated



# ------------------ Portfolio API ------------------
@app.route('/api/portfolio', methods=['GET'])
def api_get_portfolio():
    return jsonify(portfolio)

@app.route('/api/portfolio/add', methods=['POST'])
def api_add_portfolio():
    data = request.get_json()
    ticker = data.get('ticker')
    if ticker and ticker.upper() not in portfolio:
        portfolio.append(ticker.upper())
    return jsonify(portfolio)

@app.route('/api/portfolio/remove', methods=['POST'])
def api_remove_portfolio():
    data = request.get_json()
    ticker = data.get('ticker')
    if ticker and ticker.upper() in portfolio:
        portfolio.remove(ticker.upper())
    return jsonify(portfolio)

# ------------------ Stock Data API ------------------
@app.route('/api/data')
def api_data():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker required'})
    return jsonify(get_intraday_data(ticker.upper()))

@app.route("/api/info")
def api_info():
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker not specified"})

    try:
        stock = yf.Ticker(ticker)
        info = stock.info  # This can fail if ticker invalid
        if not info or "shortName" not in info:
            return jsonify({"error": f"No data found for ticker: {ticker}"})
        
        return jsonify({
            "symbol": info.get("symbol"),
            "shortName": info.get("shortName"),
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "longBusinessSummary": info.get("longBusinessSummary"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "website": info.get("website")
        })

    except Exception as e:
        return jsonify({"error": f"Failed to fetch data: {str(e)}"})

@app.route('/api/ai_analysis', methods=['POST'])
def api_ai_analysis():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({'error': 'Ticker required'}), 400
    info = get_company_info(ticker.upper())
    if 'error' in info:
        return jsonify(info), 404
    analysis = get_ai_recommendation(info)
    return jsonify({'ai_analysis': analysis})

# ------------------ Stock News via Google CSE ------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")         

@app.route('/api/stock_news')
def stock_news():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error':'Ticker required'}), 400

    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CSE_ID}&q={ticker}+stock"
    res = requests.get(url)
    articles = []
    if res.status_code == 200:
        data = res.json()
        for item in data.get('items', []):
            articles.append({
                'title': item.get('title'),
                'link': item.get('link'),
                'snippet': item.get('snippet')
            })
    return jsonify({'articles': articles})

# ------------------ Recent Searches ------------------
DB_PATH = "database.db"

def add_search(ticker):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO searches (ticker) VALUES (?)", (ticker.upper(),))
        conn.commit()
    except Exception as e:
        print("DB insert error:", e)
    finally:
        conn.close()

def get_recent_searches(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, timestamp FROM searches ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{"ticker": row[0], "timestamp": row[1]} for row in rows]

# API endpoint to add a search
@app.route("/api/search/add", methods=["POST"])
def api_add_search():
    data = request.get_json()
    ticker = data.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    add_search(ticker)
    return jsonify({"success": True, "ticker": ticker.upper()})

# API endpoint to get recent searches
@app.route("/api/search/recent")
def api_recent_searches():
    limit = int(request.args.get("limit", 10))
    return jsonify(get_recent_searches(limit))




@app.route("/api/live_price")
def live_price():
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker required"}), 400

    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    if data.empty:
        return jsonify({"error": "No data found"}), 404

    last_price = float(data["Close"].iloc[-1])

    return jsonify({
        "ticker": ticker.upper(),
        "price_in_inr": last_price,
        "currency": "INR"
    })



@app.route("/api/stock_analysis")
def stock_analysis():
    ticker_symbol = request.args.get("ticker")
    if not ticker_symbol:
        return jsonify({"error": "Ticker required"}), 400

    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="5y")  # 5-year historical data

        if hist.empty:
            return jsonify({"error": "No data found"}), 404

        avg_price = hist['Close'].mean()
        low_price = hist['Low'].min()
        high_price = hist['High'].max()
        five_year_price = hist['Close'].iloc[0]
        ltp = hist['Close'].iloc[-1]
        ftp = hist['Open'].iloc[0]

        return jsonify({
            "avgPrice": round(avg_price, 2),
            "lowPrice": round(low_price, 2),
            "highPrice": round(high_price, 2),
            "fiveYearPrice": round(five_year_price, 2),
            "ltp": round(ltp, 2),
            "ftp": round(ftp, 2)
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Failed to fetch stock data"}), 500






@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/analytics')
def analytics_page():
    ticker = request.args.get('ticker')
    if not ticker:
        return redirect(url_for('home_page'))
    return render_template('analytics.html', ticker=ticker)

@app.route('/features')
def features_page():
    return render_template('features.html')

@app.route('/portfolio')
def portfolio_page():
    return render_template('portfolio.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
