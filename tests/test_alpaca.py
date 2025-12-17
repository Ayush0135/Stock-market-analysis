import unittest
from unittest.mock import patch, MagicMock
import sys
import types

# Mock yfinance to avoid heavy imports during unit tests
mock_yf = types.ModuleType('yfinance')
mock_yf.Ticker = lambda *args, **kwargs: None
sys.modules['yfinance'] = mock_yf
# Mock websockets.sync.client used by yfinance
mock_ws = types.ModuleType('websockets')
mock_ws_sync = types.SimpleNamespace(client=types.SimpleNamespace(connect=lambda *a, **k: None))
mock_ws.sync = mock_ws_sync
sys.modules['websockets'] = mock_ws

import app  # assuming app.py is in the project root and importable

class TestAlpacaOrder(unittest.TestCase):
    @patch('app.get_alpaca_client')
    def test_place_order_success(self, mock_get_client):
        # Setup the mock client returned by get_alpaca_client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Configure the mock client's submit_order return
        mock_client.submit_order.return_value = MagicMock(_raw={'id': 'test-order'})
        
        resp = app.place_order('AAPL', 10, 'buy')
        
        self.assertEqual(resp['id'], 'test-order')
        # Verify get_alpaca_client was called
        mock_get_client.assert_called_once()
        # Verify submit_order was called on the mock client
        mock_client.submit_order.assert_called_once_with(
            symbol='AAPL', qty=10, side='buy', type='market', time_in_force='gtc'
        )

    @patch('app.get_alpaca_client')
    def test_place_order_failure(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_client.submit_order.side_effect = Exception('Alpaca error')
        
        with self.assertRaises(Exception):
            app.place_order('AAPL', 10, 'buy')

if __name__ == '__main__':
    unittest.main()
