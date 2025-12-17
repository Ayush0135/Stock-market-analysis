# 📈 Stock Market Analysis Dashboard

A comprehensive Flask-based web application for stock market analysis, portfolio management, backtesting trading strategies, and paper trading integration with Alpaca Markets.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.1.2-green.svg)

## 🌟 Features

### 📊 Real-Time Stock Analysis
- **Live Price Tracking**: Get real-time stock prices and market data
- **Intraday Charts**: Visualize minute-by-minute price movements with candlestick charts
- **Technical Indicators**: SMA (20, 50, 200), RSI, Moving Averages
- **5-Year Historical Analysis**: Comprehensive historical price data and trends
- **Performance Metrics**: 1W, 1M, 3M, 6M, 1Y, YTD returns

### 🤖 AI-Powered Insights
- **AI Market Analysis**: Powered by Groq API (LLaMA 3.1)
- **Sentiment Analysis**: Market sentiment detection (Bullish/Bearish/Neutral)
- **Smart Recommendations**: Buy/Hold/Sell suggestions with confidence levels
- **Outlook Predictions**: Short-term (1-3 months) and long-term (6-12 months) forecasts

### 📰 News & Market Intelligence
- **Stock-Specific News**: Real-time news feed for individual stocks
- **Market News**: General market updates and trends
- **Multiple News Sources**: Integration with MarketAux API, Google Custom Search, and yfinance
- **Company Information**: Detailed company profiles, sector, industry, and business summaries

### 💼 Portfolio Management
- **Track Holdings**: Monitor your stock portfolio in real-time
- **Performance Analytics**: Track gains/losses and portfolio value
- **Diversification Analysis**: View portfolio composition and allocation
- **Paper Trading**: Practice trading with virtual money through Alpaca Markets

### 🧪 Strategy Backtesting
- **Historical Backtesting**: Test trading strategies on historical data
- **Pre-built Strategies**:
  - **SMA Crossover**: Golden cross and death cross signals
  - **RSI Strategy**: Oversold/overbought momentum trading
- **Custom Strategies**: Build and test your own trading algorithms
- **Performance Metrics**: Win rate, total return, number of trades, portfolio history
- **Live Strategy Execution**: Execute tested strategies in paper trading environment

### 🔄 Live Trading (Paper Trading)
- **Alpaca Integration**: Seamless connection to Alpaca Markets paper trading
- **Order Placement**: Market orders for buying and selling stocks
- **Position Tracking**: Monitor current positions and holdings
- **Strategy Automation**: Auto-execute strategies based on market signals

### 📈 Market Sections
- **Stocks**: In-depth stock analysis and research
- **Crypto**: Cryptocurrency market tracking
- **ETFs**: Exchange-traded fund analysis
- **Markets**: Overall market trends and indices

### 🎨 Modern UI/UX
- Responsive design for desktop and mobile
- Interactive charts powered by Chart.js
- Clean and intuitive interface
- Real-time data updates

---

## 🛠️ Tech Stack

### Backend
- **Flask 3.1.2**: Web framework
- **Python 3.8+**: Core language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### APIs & Data Sources
- **yfinance**: Yahoo Finance data integration
- **Alpaca Trade API**: Paper trading and historical market data
- **Groq API**: AI-powered market analysis (LLaMA 3.1)
- **MarketAux API**: Financial news aggregation
- **Google Custom Search API**: Additional news sources

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript**: Dynamic interactions
- **Chart.js**: Interactive charting library
- **Bootstrap**: Responsive UI components

### Database
- **SQLite**: User management and session storage

### Additional Libraries
- `langchain-groq`: LLM integration
- `requests`: HTTP requests
- `python-dotenv`: Environment variable management
- `Flask-Mail`: Email notifications

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- API Keys (see Configuration section)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ayush0135/Stock-market-analysis.git
cd Stock-market-analysis
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
# Flask Configuration
SECRET_KEY=your_secret_key_here

# Alpaca Markets (Paper Trading)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

# Groq API (AI Analysis)
GROQ_API_KEY=your_groq_api_key

# MarketAux API (News)
MARKETAUX_API_KEY=your_marketaux_api_key

# Google Custom Search (Optional - for additional news)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

---

## 🔑 API Keys Setup

### Alpaca Markets (Required for Trading & Backtesting)
1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Navigate to Paper Trading account
3. Generate API keys from the dashboard
4. Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to `.env`

### Groq API (Required for AI Analysis)
1. Sign up at [Groq](https://console.groq.com/)
2. Generate an API key
3. Add `GROQ_API_KEY` to `.env`

### MarketAux (Optional - for News)
1. Sign up at [MarketAux](https://www.marketaux.com/)
2. Get your API token
3. Add `MARKETAUX_API_KEY` to `.env`

### Google Custom Search (Optional - for Additional News)
1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Custom Search API
3. Create a Custom Search Engine at [Programmable Search](https://programmablesearchengine.google.com/)
4. Add `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` to `.env`

---

## 📖 Usage

### Stock Analysis
1. Navigate to the **Analytics** page
2. Enter a stock ticker (e.g., AAPL, GOOGL, TSLA)
3. View real-time price, charts, and technical indicators
4. Get AI-powered analysis and recommendations
5. Read latest news about the stock

### Portfolio Management
1. Go to the **Portfolio** page
2. Add stocks to track your holdings
3. Monitor performance and gains/losses
4. View portfolio composition

### Backtesting Strategies
1. Visit the **Backtest** page
2. Select a stock ticker and date range
3. Choose a strategy:
   - **SMA Crossover**: Short/Long period configuration
   - **RSI Strategy**: Oversold/Overbought thresholds
4. Run backtest and analyze results
5. View trade history and portfolio performance chart

### Live Strategy Execution
1. After backtesting, navigate to strategy execution
2. Select the tested strategy
3. Execute on live market data (paper trading)
4. Monitor positions in your Alpaca dashboard

### News & Market Insights
1. Go to the **News** page
2. View general market news or search for specific stocks
3. Click on articles to read full stories

---

## 🏗️ Project Structure

```
Stock-market-analysis/
├── app.py                      # Main Flask application
├── backtest.py                 # Backtesting engine
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in repo)
├── .gitignore                  # Git ignore rules
├── users.db                    # SQLite database
│
├── templates/                  # HTML templates
│   ├── base.html              # Base template with navigation
│   ├── home.html              # Landing page
│   ├── analytics.html         # Stock analysis dashboard
│   ├── portfolio.html         # Portfolio management
│   ├── backtest.html          # Strategy backtesting
│   ├── news.html              # News feed
│   ├── markets.html           # Market overview
│   ├── crypto.html            # Cryptocurrency section
│   ├── etfs.html              # ETF analysis
│   ├── settings.html          # User settings
│   ├── about.html             # About page
│   ├── contact.html           # Contact page
│   └── features.html          # Features page
│
├── static/                     # Static assets
│   └── style.css              # Custom CSS styles
│
├── data/                       # Cached historical data
│   └── *.json                 # Stock data cache files
│
└── tests/                      # Test files
    └── test_alpaca.py         # Alpaca integration tests
```

---

## 🔌 API Endpoints

### Stock Data
- `GET /api/live_price?ticker=AAPL` - Get live price for a ticker
- `GET /api/info?ticker=AAPL` - Get company information
- `GET /api/stock_analysis?ticker=AAPL` - Get 5-year analysis
- `GET /api/intraday_data?ticker=AAPL` - Get intraday chart data
- `GET /api/batch_price?tickers=AAPL,GOOGL,TSLA` - Get multiple stock prices

### Search
- `GET /api/search_ticker?query=apple` - Search for stock tickers

### News
- `GET /api/stock_news?ticker=AAPL` - Get news for a specific stock
- `GET /api/stock_news` - Get general market news

### AI Analysis
- `POST /api/ai_analysis` - Get AI-powered stock analysis
  ```json
  {
    "ticker": "AAPL"
  }
  ```

### Backtesting
- `POST /api/backtest` - Run a backtest
  ```json
  {
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "strategy": "sma_crossover",
    "initial_capital": 10000,
    "params": {
      "short_period": 10,
      "long_period": 30
    }
  }
  ```

### Trading
- `POST /api/execute_strategy` - Execute strategy on live data
  ```json
  {
    "symbol": "AAPL",
    "strategy": "rsi",
    "params": {
      "period": 14,
      "oversold": 30,
      "overbought": 70
    }
  }
  ```

- `POST /api/alpaca_order` - Place a paper trading order
  ```json
  {
    "symbol": "AAPL",
    "qty": 10,
    "side": "buy",
    "type": "market",
    "time_in_force": "gtc"
  }
  ```

---

## 🧪 Testing

Run tests using pytest:
```bash
pip install pytest
pytest tests/
```

Test Alpaca connection:
```bash
python tests/test_alpaca.py
```

---

## 🎯 Backtesting Strategies

### SMA Crossover Strategy
```python
from backtest import Backtest, sma_crossover_strategy

bt = Backtest('AAPL', '2023-01-01', '2023-12-31')
strategy = sma_crossover_strategy(short_period=10, long_period=30)
results = bt.run(strategy, initial_capital=10000)
print(results)
```

**How it works:**
- Buys when the short-term SMA crosses above the long-term SMA (Golden Cross)
- Sells when the short-term SMA crosses below the long-term SMA (Death Cross)

### RSI Strategy
```python
from backtest import Backtest, rsi_strategy

bt = Backtest('AAPL', '2023-01-01', '2023-12-31')
strategy = rsi_strategy(period=14, oversold=30, overbought=70)
results = bt.run(strategy, initial_capital=10000)
print(results)
```

**How it works:**
- Buys when RSI drops below the oversold threshold (default: 30)
- Sells when RSI rises above the overbought threshold (default: 70)

### Custom Strategy
```python
def my_custom_strategy(df, i, positions):
    """
    Create your own trading strategy.
    
    Args:
        df: DataFrame with OHLCV data
        i: Current index in the dataframe
        positions: List of current positions [(qty, buy_price), ...]
    
    Returns:
        None or ('buy'/'sell', quantity)
    """
    # Your strategy logic here
    if i < 20:
        return None
    
    # Example: Simple price threshold
    current_price = df['close'].iloc[i]
    if current_price < 150 and not positions:
        return ('buy', 10)
    elif current_price > 200 and positions:
        return ('sell', 10)
    
    return None

bt = Backtest('AAPL', '2023-01-01', '2023-12-31')
results = bt.run(my_custom_strategy, initial_capital=10000)
```

---

## 🔒 Security Best Practices

1. **Never commit `.env` file** - It's already in `.gitignore`
2. **Use paper trading** - Test strategies before using real money
3. **Rotate API keys** - Regularly update your API credentials
4. **Secure your SECRET_KEY** - Use a strong, random string
5. **HTTPS in Production** - Always use SSL/TLS for production deployments

---

## 🚧 Troubleshooting

### Issue: "Module not found" errors
**Solution:** Ensure virtual environment is activated and dependencies are installed
```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Alpaca API credentials not configured"
**Solution:** Check that `.env` file exists and contains valid Alpaca credentials

### Issue: "No data available" for backtesting
**Solution:** Verify the date range and ensure Alpaca has data for that period

### Issue: AI analysis returns "AI key missing"
**Solution:** Add a valid `GROQ_API_KEY` to your `.env` file

### Issue: News not loading
**Solution:** 
- Check if `MARKETAUX_API_KEY` or `GOOGLE_API_KEY` is configured
- The app falls back to yfinance news if external APIs aren't available

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Update README for new features
- Test thoroughly before submitting PR

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Ayush**
- GitHub: [@Ayush0135](https://github.com/Ayush0135)
- Repository: [Stock-market-analysis](https://github.com/Ayush0135/Stock-market-analysis)

---

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance market data
- [Alpaca Markets](https://alpaca.markets/) - Commission-free trading API
- [Groq](https://groq.com/) - AI inference platform
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Chart.js](https://www.chartjs.org/) - Charting library

---

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on [GitHub](https://github.com/Ayush0135/Stock-market-analysis/issues)
- Check the troubleshooting section above
- Review the API documentation

---

## 📊 Future Enhancements

- [ ] Real-time WebSocket data streaming
- [ ] Mobile app (React Native)
- [ ] Advanced charting with TradingView integration
- [ ] More technical indicators (MACD, Bollinger Bands, Fibonacci)
- [ ] Social sentiment analysis from Twitter/Reddit
- [ ] Options trading support
- [ ] Multi-timeframe analysis
- [ ] Email/SMS alerts for price movements
- [ ] Advanced portfolio analytics (Sharpe ratio, beta, alpha)
- [ ] Strategy optimization with genetic algorithms

---

## ⚠️ Disclaimer

**This software is for educational purposes only. Do not use it for actual trading without understanding the risks involved. Stock trading involves risk and you can lose money. Past performance does not guarantee future results. Always do your own research and consult with a financial advisor before making investment decisions.**

---

Made with ❤️ by Ayush
