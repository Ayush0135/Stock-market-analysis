from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# ------------------ Helper functions ------------------

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
        'website': info.get('website'),
        'longBusinessSummary': info.get('longBusinessSummary')
    }

# ------------------ Portfolio ------------------
portfolio = []

@app.route('/api/portfolio')
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

# ------------------ API Endpoints ------------------

@app.route('/api/data')
def api_data():  # Intraday data
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error':'Ticker required'})
    return jsonify(get_intraday_data(ticker.upper()))

@app.route('/api/info')
def api_info():  # Company info
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'error':'Ticker required'})
    return jsonify(get_company_info(ticker.upper()))

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '').upper()
    results = [{'symbol': t, 'name': t} for t in portfolio if query in t]
    if query and query not in [r['symbol'] for r in results]:
        results.append({'symbol': query, 'name': query})
    return jsonify(results)

# ------------------ Frontend Routes ------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/features')
def features_page():
    return render_template('features.html')

@app.route('/portfolio')
def portfolio_page():
    return render_template('portfolio.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

# ------------------ Run App ------------------

if __name__ == '__main__':
    app.run(debug=True)
