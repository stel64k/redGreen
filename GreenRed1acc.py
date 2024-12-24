import ccxt
import numpy as np
import talib,ta
import pandas as pd
import time
import configparser
import logging, schedule
import telegram
from datetime import datetime,timedelta
from binance.client import Client
from configparser import ConfigParser
from requests.exceptions import ConnectionError, HTTPError
import os,json
from binance.exceptions import BinanceAPIException

# Загрузка конфигурации
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['Binance']['api_key']
api_secret = config['Binance']['api_secret']
telegram_token = config['telegram']['token']
telegram_chat_id = config['telegram']['chat_id']

margin_mode = config['Binance']['margin_mode']
position_size_percent = float(config['Binance']['position_size_percent'])
leverage = int(config['Binance']['leverage'])
take_profit_percent = float(config['Binance']['take_profit_percent'])
stop_loss_percent = float(config['Binance']['stop_loss_percent'])

# Настройка Telegram бота
telegram_bot = telegram.Bot(token=telegram_token)

# Настройка логирования
logging.basicConfig(filename='bot.log', level=logging.INFO, filemode='w',format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',)

# Создание экземпляра клиента Binance Futures (ccxt)
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
})
exchange.options['defaultType'] = 'future'

# Создание экземпляра клиента Binance (binance)
binance_client = Client(api_key=api_key, api_secret=api_secret)
open_orders = {}

blacklist = {'FTM/USDT','LINK/USDT','TUSD/USDT','NEO/USDT','BTC/USDT', 'ETH/USDT', 'XRP/USDT','BNB/USDT','LTC/USDT','EOS/USDT','ETC/USDT','BCHABC/USDT','USDC/USDT','NULS/USDT','BTT/USDT','PAX/USDT','WAVES/USDT','VEN/USDT','BCC/USDT','USDS/USDT','XMR/USDT'}

# Функция для чтения состояния пар из файла
def load_pair_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}  # Если файл пустой или поврежден, возвращаем пустой словарь
    return {}

# Функция для сохранения состояния пар в файл
def save_pair_state(file_path, pair_state):
    with open(file_path, 'w') as f:
        json.dump(pair_state, f, indent=4)

# Функция обновления состояния пары
def update_pair_state(file_path, trading_pair, new_state):
    pair_state = load_pair_state(file_path)  # Читаем текущее состояние
    pair_state[trading_pair] = new_state     # Обновляем состояние для конкретной пары
    save_pair_state(file_path, pair_state)   # Сохраняем обновленное состояние

def read_config(file_path):
    config = ConfigParser()
    try:
        config.read(file_path)
        settings = {
            'api_key': config.get('Binance', 'api_key'),
            'api_secret': config.get('Binance', 'api_secret'),
            'margin_mode': config.get('Binance', 'margin_mode'),
            'position_size_percent': float(config.get('Binance', 'position_size_percent')),
            'leverage': int(config.get('Binance', 'leverage')),
            'take_profit_percent': float(config.get('Binance', 'take_profit_percent')),
            'stop_loss_percent': float(config.get('Binance', 'stop_loss_percent')),
        }
        return settings
    except Exception as e:
        logging.error(f"Error reading config file: {e}")
        exit()

def initialize_client(api_key, api_secret):
    try:
        return Client(api_key=api_key, api_secret=api_secret)
    except Exception as e:
        logging.error(f"Error initializing Binance client: {e}")
        exit()
use_heikin_ashi = True
def send_telegram_message(message):
    try:
        telegram_bot.send_message(chat_id=telegram_chat_id, text=message)
        logging.info(f"Telegram message sent: {message}")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def fetch_ohlcv(symbol, timeframe='5m', limit=500):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Error fetching OHLCV data for {symbol} on {timeframe} timeframe: {e}")
        return None


# Функция для расчета канала Дончиана
def donchian_channel(df, period):
    if df is None or df.empty:
        return None
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    return (low_min + high_max) / 2



def calculate_heikin_ashi(df):
    """
    Рассчитывает свечи Хейкен Аши.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = ha_close.copy()
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.DataFrame({'ha_open': ha_open, 'ha_close': ha_close, 'high': df['high']}).max(axis=1)
    ha_low = pd.DataFrame({'ha_open': ha_open, 'ha_close': ha_close, 'low': df['low']}).min(axis=1)

    # Возврат результата в виде DataFrame
    return pd.DataFrame({
        'ha_open': ha_open,
        'ha_close': ha_close,
        'ha_high': ha_high,
        'ha_low': ha_low
    })



def calculate_ema(df, period):
    """
    Рассчитывает экспоненциальную скользящую среднюю (EMA).
    """
    ema = df['close'].ewm(span=period, adjust=False).mean()
    print(f"EMA ({period}):\n{ema.tail()}")
    return ema

# def check_signals(df, ema_period=12):
#     """
#     Проверяет сигналы на покупку и продажу на основе Хейкен Аши и трендового фильтра (EMA).
#     """
#     try:
#         if df.isnull().any().any():
#             print("Data contains NaN values.")
#             return None, None, None

#         # Рассчитать свечи Хейкен Аши
#         ha_df = calculate_heikin_ashi(df)

#         # Проверка, что данные Хейкен Аши не пустые
#         if ha_df.empty:
#             print("Heikin Ashi data is empty.")
#             return None, None, None

#         # Добавить EMA в основной DataFrame
#         df['ema'] = calculate_ema(df, ema_period)

#         # Определить сигналы
#         ha_df['is_bullish'] = ha_df['ha_close'] > ha_df['ha_open']
#         ha_df['is_bearish'] = ha_df['ha_close'] < ha_df['ha_open']
#         ha_df['above_ema'] = df['close'] > df['ema']
#         ha_df['below_ema'] = df['close'] < df['ema']

#         # Сигналы на покупку и продажу
#         ha_df['long_signal'] = ha_df['is_bullish'] & ha_df['above_ema']
#         ha_df['short_signal'] = ha_df['is_bearish'] & ha_df['below_ema']

#         # Вывод диагностической информации
#         print("Heikin Ashi DataFrame:")
#         print(ha_df[['ha_open', 'ha_close']].tail())
#         print(f"EMA ({ema_period}):")
#         print(df['ema'].tail())
#         print("DataFrame with EMA:")
#         print(df[['close', 'ema']].tail())
#         print("Signal flags:")
#         print(ha_df[['is_bullish', 'is_bearish', 'above_ema', 'below_ema']].tail())
#         print("Long/Short signals:")
#         print(ha_df[['long_signal', 'short_signal']].tail())

#         # Получить время и цену последней закрытой свечи
#         last_candle_time = df['timestamp'].iloc[-1] + pd.Timedelta(hours=3)
#         last_candle_price = df['close'].iloc[-2]

#         # Диагностика для последней свечи
#         print(f"Last candle time: {last_candle_time}, Last candle price: {last_candle_price}")
#         print(f"Close: {df['close'].iloc[-2]}, EMA: {df['ema'].iloc[-2]}")
#         print(f"Heikin Ashi Close: {ha_df['ha_close'].iloc[-2]}, Open: {ha_df['ha_open'].iloc[-2]}")
#         print(f"Below EMA: {ha_df['below_ema'].iloc[-2]}, Is Bearish: {ha_df['is_bearish'].iloc[-2]}")

#         # Определение сигнала и позиции
#         if ha_df['long_signal'].iloc[-2]:
#             position_side = 'LONG'
#             print("Signal: LONG")
#             return "LONG", last_candle_price, position_side
#         elif ha_df['short_signal'].iloc[-2]:
#             position_side = 'SHORT'
#             print("Signal: SHORT")
#             return "SHORT", last_candle_price, position_side
#         else:
#             print("No valid signal detected.")
#             return None, None, None

#     except Exception as e:
#         print(f"Error checking signals: {e}")
#         return None, None, None

def calculate_atr(df, period=14):
    """
    Рассчитывает ATR (средний истинный диапазон) для заданного DataFrame.

    :param df: DataFrame, содержащий столбцы ['high', 'low', 'close'].
    :param period: Период для расчета ATR.
    :return: pandas.Series с значениями ATR.
    """
    try:
        # Истинный диапазон
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Средний истинный диапазон (ATR)
        atr = df['true_range'].rolling(window=period, min_periods=1).mean()
        
        return atr
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return pd.Series([None] * len(df))


def check_signals(df, ema_period=9, atr_period=14, trend_ema_period=14):
    """
    Проверяет сигналы на покупку и продажу на основе Хейкен Аши, EMA и дополнительных фильтров.
    """
    try:
        if df.isnull().any().any():
            print("Data contains NaN values.")
            return None, None, None

        # Рассчитать свечи Хейкен Аши
        ha_df = calculate_heikin_ashi(df)

        # Проверка, что данные Хейкен Аши не пустые
        if ha_df.empty:
            print("Heikin Ashi data is empty.")
            return None, None, None

        # Рассчитать EMA на основе столбца закрытия
        df['ema'] = calculate_ema(df, ema_period)
        df['trend_ema'] = calculate_ema(df, trend_ema_period)

        # Рассчитать ATR для фильтра волатильности
        df['atr'] = calculate_atr(df, atr_period)

        # Определить сигналы с учетом текущей свечи
        ha_df['is_bullish'] = ha_df['ha_close'] > ha_df['ha_open']
        ha_df['is_bearish'] = ha_df['ha_close'] < ha_df['ha_open']
        ha_df['above_ema'] = df['close'] > df['ema']  # Сравнение текущей цены с текущей EMA
        ha_df['below_ema'] = df['close'] < df['ema']  # Сравнение текущей цены с текущей EMA
        ha_df['above_trend_ema'] = df['close'] > df['trend_ema']  # Сравнение с трендовой EMA
        ha_df['below_trend_ema'] = df['close'] < df['trend_ema']  # Сравнение с трендовой EMA

        # Фильтр тела свечи (должно быть больше 50% от диапазона свечи)
        ha_df['body_to_range'] = abs(ha_df['ha_close'] - ha_df['ha_open']) / \
                                 (ha_df['ha_high'] - ha_df['ha_low'] + 1e-10)
        ha_df['valid_body'] = ha_df['body_to_range'] > 0.5

        # Сигналы на покупку и продажу для текущей свечи с учетом фильтров
        ha_df['long_signal'] = (
            ha_df['is_bullish'] & ha_df['above_ema'] & ha_df['above_trend_ema'] &
            ha_df['valid_body'] & (ha_df['ha_high'] - ha_df['ha_low'] > df['atr'])
        )
        ha_df['short_signal'] = (
            ha_df['is_bearish'] & ha_df['below_ema'] & ha_df['below_trend_ema'] &
            ha_df['valid_body'] & (ha_df['ha_high'] - ha_df['ha_low'] > df['atr'])
        )

        # Получить время и цену текущей свечи
        current_candle_time = df['timestamp'].iloc[-1] + pd.Timedelta(hours=3)
        current_candle_price = df['close'].iloc[-1]

        # Определение сигнала и позиции
        if ha_df['long_signal'].iloc[-1]:
            position_side = 'LONG'
            print(f"Signal: LONG")
            return "LONG", current_candle_price, position_side
        elif ha_df['short_signal'].iloc[-1]:
            position_side = 'SHORT'
            print(f"Signal: SHORT")
            return "SHORT", current_candle_price, position_side
        else:
            print("No signal.")
            return None, None, None

    except Exception as e:
        print(f"Error checking signals: {e}")
        return None, None, None


def get_symbol_info(client, trading_pair):
    try:
        trading_pair = trading_pair.replace(':USDT', '').replace('/', '')  # Clean symbol
        symbol_info = client.futures_exchange_info()
        for symbol in symbol_info['symbols']:
            if symbol['symbol'] == trading_pair:
                step_size = float(symbol['filters'][1]['stepSize'])
                tick_size = float(symbol['filters'][0]['tickSize'])
                min_notional = float(symbol['filters'][5]['notional'])
                return step_size, tick_size, min_notional
        logging.error(f"Symbol info not found for {trading_pair}.")
        return None, None, None
    except Exception as e:
        logging.error(f"Error fetching symbol info: {e}")
        return None, None, None

def set_margin_mode(client, trading_pair, margin_mode):
    try:
        trading_pair = trading_pair.replace(':USDT', '').replace('/', '')  # Clean symbol
        if margin_mode.lower() == 'isolated':
            client.futures_change_margin_type(symbol=trading_pair, marginType='ISOLATED')
        elif margin_mode.lower() == 'cross':
            client.futures_change_margin_type(symbol=trading_pair, marginType='CROSSED')
        else:
            logging.error(f"Invalid margin mode: {margin_mode}")
            exit()
    except Exception as e:
        if "No need to change margin type" in str(e):
            logging.info("Margin mode already set.")
        else:
            logging.error(f"Error changing margin mode: {e}")
            exit()

def get_account_balance(client):
    try:
        account_info = client.futures_account()
        balance = float(account_info['totalWalletBalance'])
        return balance
    except Exception as e:
        logging.error(f"Error fetching account balance: {e}")
        exit()

def calculate_position_size(balance, position_size_percent, leverage, current_price, step_size, min_notional):
    try:
        if step_size is None or min_notional is None:
            logging.error("Failed to get symbol info for position size calculation.")
            return None

        notional_value = balance * position_size_percent / 100 * leverage
        position_size = notional_value / current_price
        position_size = round(position_size - (position_size % step_size), 3)

        if notional_value < min_notional:
            position_size = min_notional / current_price
            position_size = round(position_size - (position_size % step_size), 3)
            logging.info(f"Position size adjusted to minimum notional value: {position_size}")

        return position_size
    except Exception as e:
        logging.error(f"Error calculating position size: {e}")
        return None

def calculate_prices(current_price, take_profit_percent, stop_loss_percent, position_side, tick_size):
    try:
        if position_side == 'LONG':
            take_profit_price = current_price * (1 + take_profit_percent / 100)
            stop_loss_price = current_price * (1 - stop_loss_percent / 100)
        elif position_side == 'SHORT':
            take_profit_price = current_price * (1 - take_profit_percent / 100)
            stop_loss_price = current_price * (1 + stop_loss_percent / 100)
        else:
            logging.error(f"Invalid position_side: {position_side}")
            exit()

        take_profit_price = round(take_profit_price - (take_profit_price % tick_size), 5)
        stop_loss_price = round(stop_loss_price - (stop_loss_price % tick_size), 5)

        return take_profit_price, stop_loss_price
    except Exception as e:
        logging.error(f"Error calculating prices: {e}")
        return None, None

def count_open_positions(client, position_side):
    try:
        account_info = client.futures_account()
        positions = account_info['positions']
        count = 0
        for pos in positions:
            if pos['positionSide'] == position_side and float(pos['positionAmt']) != 0:
                count += 1
        return count
    except Exception as e:
        logging.error(f"Error counting open positions: {e}")
        return None

def cancel_all_orders(client, trading_pair):
    try:
        trading_pair = trading_pair.replace(':USDT', '').replace('/', '')  # Clean symbol
        open_orders = client.futures_get_open_orders(symbol=trading_pair)
        for order in open_orders:
            client.futures_cancel_order(symbol=trading_pair, orderId=order['orderId'])
        logging.info(f"Cancelled all open orders for {trading_pair}.")
    except Exception as e:
        logging.error(f"Error cancelling orders for {trading_pair}: {e}")

def cleanup_orders(client):
    try:
        # Получаем список всех открытых ордеров
        open_orders = client.futures_get_open_orders()
        # Создаем словарь, чтобы отслеживать ордера по символам
        orders_by_symbol = {}
        for order in open_orders:
            symbol = order['symbol']
            if symbol not in orders_by_symbol:
                orders_by_symbol[symbol] = []
            orders_by_symbol[symbol].append(order)
        
        # Проверяем позиции по каждому символу
        for symbol, orders in orders_by_symbol.items():
            open_positions = client.futures_position_information(symbol=symbol)
            has_open_position = any(float(pos['positionAmt']) != 0 for pos in open_positions)

            # Удаляем ордера, если позиции нет
            if not has_open_position:
                for order in orders:
                    if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET']:
                        client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                        logging.info(f"Removed {order['type']} order for {symbol} as no open position exists.")
    except Exception as e:
        logging.error(f"Error cleaning up orders: {e}")

def ensure_stop_loss_take_profit(client):
    try:
        open_positions = client.futures_position_information()
        for pos in open_positions:
            symbol = pos['symbol']
            position_amt = float(pos['positionAmt'])
            if position_amt == 0:
                continue
            
            entry_price = get_entry_price(client, symbol) or get_entry_price_via_trades(client, symbol)
            if entry_price is None:
                logging.error(f"Could not fetch entry price for {symbol}. Skipping...")
                continue
            
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])

            # Определяем сторону позиции и рассчитываем ROI
            if pos['positionSide'] == 'LONG':
                roi = ((current_price - entry_price) / entry_price) * 100 * leverage
            elif pos['positionSide'] == 'SHORT':
                roi = ((entry_price - current_price) / entry_price) * 100 * leverage
            else:
                logging.error(f"Invalid positionSide: {pos['positionSide']}")
                continue
            
            # logging.info(f"{symbol} {pos['positionSide']} ROI: {roi}%, entry_price: {entry_price}, current_price: {current_price}, leverage: {leverage}")
            
            step_size, tick_size, min_notional = get_symbol_info(client, symbol)
            if step_size is None or min_notional is None:
                continue

            # Не округляем position_amt
            position_amt = position_amt

            open_orders = client.futures_get_open_orders(symbol=symbol)
            has_take_profit = any(order['type'] == 'TAKE_PROFIT_MARKET' for order in open_orders)
            has_stop_loss = any(order['type'] == 'STOP_MARKET' for order in open_orders)

            take_profit_price, stop_loss_price = calculate_prices(current_price, take_profit_percent, stop_loss_percent, pos['positionSide'], tick_size)

            
            if not has_take_profit or not has_stop_loss:
                balance = get_account_balance(client)
                position_size = calculate_position_size(balance, position_size_percent, leverage, current_price, step_size, min_notional)

                take_profit_price = round(take_profit_price - (take_profit_price % tick_size), 5)
                stop_loss_price = round(stop_loss_price - (stop_loss_price % tick_size), 5)

                if not has_take_profit:
                    client.futures_create_order(
                        symbol=symbol,
                        side='SELL' if pos['positionSide'] == 'LONG' else 'BUY',
                        type='TAKE_PROFIT_MARKET',
                        quantity=abs(position_amt),
                        stopPrice=take_profit_price,
                        positionSide=pos['positionSide']
                    )
                    logging.info(f"Created TAKE_PROFIT order for {symbol} at {take_profit_price}")

                if not has_stop_loss:
                    client.futures_create_order(
                        symbol=symbol,
                        side='SELL' if pos['positionSide'] == 'LONG' else 'BUY',
                        type='STOP_MARKET',
                        quantity=abs(position_amt),
                        stopPrice=stop_loss_price,
                        positionSide=pos['positionSide']
                    )
                    logging.info(f"Created STOP_LOSS order for {symbol} at {stop_loss_price}")
            
            # Передаем рассчитанный ROI в функцию трейлинга
            trail_stop_and_take_profit(client, symbol, current_price, entry_price, position_amt, pos['positionSide'], pos['positionSide'], leverage, tick_size, roi)

    except Exception as e:
        logging.error(f"Error ensuring stop loss and take profit orders: {e}")

def trail_stop_and_take_profit(client, symbol, current_price, entry_price, position_amt, position_side_setting, position_side, leverage, tick_size, roi, roi_threshold=150, roi_extension=2, stop_distance_percent=2,pnl_threshold_percent=45):
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        logging.info(f"{symbol} {position_side} ROI: {roi}%, current_price: {current_price}, entry_price: {entry_price}, leverage: {leverage}")
        
         # Получаем информацию о всех открытых позициях
        positions = client.futures_position_information()

        # Рассчитываем совокупный PNL по всем позициям с учетом плеча
        total_pnl = sum(float(position['unRealizedProfit']) * float(position['leverage']) for position in positions)
        total_pnl= total_pnl / leverage
        print(total_pnl)
        # Получаем текущий баланс счета
        account_info = client.futures_account()
        balance = float(account_info['totalWalletBalance'])
        print(balance)
        time.sleep(0.5)
        # Проверяем, превышает ли PNL 5% от баланса
        if total_pnl > balance * pnl_threshold_percent / 100:
            logging.info(f"Total PNL {total_pnl} exceeds {pnl_threshold_percent}% of balance {balance}. Closing all positions.")

            # Закрываем все открытые позиции
            for position in positions:
                if float(position['positionAmt']) != 0:
                    side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
                    client.futures_create_order(
                        symbol=position['symbol'],
                        side=side,
                        type='MARKET',
                        quantity=abs(float(position['positionAmt'])),
                        positionSide=position['positionSide']
                    )
                    logging.info(f"Closed position for {position['symbol']}")

            # Отправляем уведомление в Telegram
            message = f"Closed all positions. Total PNL: {total_pnl}, Balance: {balance}"
            send_telegram_message(message)
            return

        # Проверяем, достигнут ли порог для трейлинга
        if roi >= roi_threshold:
            if position_side == 'LONG':
                new_take_profit_price = current_price * (1 + roi_extension / 100)
                new_stop_loss_price = current_price * (1 - stop_distance_percent / 100)
            elif position_side == 'SHORT':
                new_take_profit_price = current_price * (1 - roi_extension / 100)
                new_stop_loss_price = current_price * (1 + stop_distance_percent / 100)

            new_take_profit_price = round(new_take_profit_price - (new_take_profit_price % tick_size), 5)
            new_stop_loss_price = round(new_stop_loss_price - (new_stop_loss_price % tick_size), 5)

            logging.info(f"Calculated new prices for {symbol}: Take Profit - {new_take_profit_price}, Stop Loss - {new_stop_loss_price}")

            open_orders = client.futures_get_open_orders(symbol=symbol)
            current_stop_loss_order = next((order for order in open_orders if order['type'] == 'STOP_MARKET'), None)
            current_take_profit_order = next((order for order in open_orders if order['type'] == 'TAKE_PROFIT_MARKET'), None)

            # Обработка стоп-лосса
            if current_stop_loss_order:
                current_stop_loss_price = float(current_stop_loss_order['stopPrice'])
                current_stop_loss_price = round(current_stop_loss_price - (current_stop_loss_price % tick_size), 5)
                logging.info(f"Existing STOP_LOSS price for {symbol}: {current_stop_loss_price}")

                if position_side == 'LONG':
                    if new_stop_loss_price > current_stop_loss_price:
                        # Обновляем стоп-лосс только если новый стоп-лосс более выгодный
                        client.futures_cancel_order(symbol=symbol, orderId=current_stop_loss_order['orderId'])
                        client.futures_create_order(
                            symbol=symbol,
                            side='SELL' if position_side == 'LONG' else 'BUY',
                            type='STOP_MARKET',
                            quantity=abs(position_amt),
                            stopPrice=new_stop_loss_price,
                            positionSide=position_side_setting
                        )
                        logging.info(f"Updated STOP_LOSS order for {symbol} at {new_stop_loss_price}")
                    else:
                        logging.info(f"Current STOP_LOSS order for {symbol} is more favorable than the new stop loss price.")
                elif position_side == 'SHORT':
                    if new_stop_loss_price < current_stop_loss_price:
                        # Обновляем стоп-лосс только если новый стоп-лосс более выгодный
                        client.futures_cancel_order(symbol=symbol, orderId=current_stop_loss_order['orderId'])
                        client.futures_create_order(
                            symbol=symbol,
                            side='SELL' if position_side == 'LONG' else 'BUY',
                            type='STOP_MARKET',
                            quantity=abs(position_amt),
                            stopPrice=new_stop_loss_price,
                            positionSide=position_side_setting
                        )
                        logging.info(f"Updated STOP_LOSS order for {symbol} at {new_stop_loss_price}")
                    else:
                        logging.info(f"Current STOP_LOSS order for {symbol} is more favorable than the new stop loss price.")
            else:
                # Если стоп-лосс не установлен, создаем его
                client.futures_create_order(
                    symbol=symbol,
                    side='SELL' if position_side == 'LONG' else 'BUY',
                    type='STOP_MARKET',
                    quantity=abs(position_amt),
                    stopPrice=new_stop_loss_price,
                    positionSide=position_side_setting
                )
                logging.info(f"Created STOP_LOSS order for {symbol} at {new_stop_loss_price}")

            # Обработка тейк-профита
            if current_take_profit_order:
                current_take_profit_price = float(current_take_profit_order['stopPrice'])
                current_take_profit_price = round(current_take_profit_price - (current_take_profit_price % tick_size), 5)
                logging.info(f"Existing TAKE_PROFIT price for {symbol}: {current_take_profit_price}")

                if position_side == 'LONG':
                    if new_take_profit_price > current_take_profit_price:
                        # Обновляем тейк-профит только если новый тейк-профит более выгодный
                        client.futures_cancel_order(symbol=symbol, orderId=current_take_profit_order['orderId'])
                        client.futures_create_order(
                            symbol=symbol,
                            side='SELL' if position_side == 'LONG' else 'BUY',
                            type="TAKE_PROFIT_MARKET",
                            quantity=abs(position_amt),
                            stopPrice=new_take_profit_price,
                            positionSide=position_side_setting
                        )
                        logging.info(f"Updated TAKE_PROFIT order for {symbol} at {new_take_profit_price}")
                    else:
                        logging.info(f"Current TAKE_PROFIT order for {symbol} is more favorable than the new take profit price.")
                elif position_side == 'SHORT':
                    if new_take_profit_price < current_take_profit_price:
                        # Обновляем тейк-профит только если новый тейк-профит более выгодный
                        client.futures_cancel_order(symbol=symbol, orderId=current_take_profit_order['orderId'])
                        client.futures_create_order(
                            symbol=symbol,
                            side='SELL' if position_side == 'LONG' else 'BUY',
                            type="TAKE_PROFIT_MARKET",
                            quantity=abs(position_amt),
                            stopPrice=new_take_profit_price,
                            positionSide=position_side_setting
                        )
                        logging.info(f"Updated TAKE_PROFIT order for {symbol} at {new_take_profit_price}")
                    else:
                        logging.info(f"Current TAKE_PROFIT order for {symbol} is more favorable than the new take profit price.")
            else:
                # Если тейк-профит не установлен, создаем его
                client.futures_create_order(
                    symbol=symbol,
                    side='SELL' if position_side == 'LONG' else 'BUY',
                    type="TAKE_PROFIT_MARKET",
                    quantity=abs(position_amt),
                    stopPrice=new_take_profit_price,
                    positionSide=position_side_setting
                )
                logging.info(f"Created TAKE_PROFIT order for {symbol} at {new_take_profit_price}")

            message = (
                f"Updated orders for {symbol}:\n"
                f"New take profit price: {new_take_profit_price}\n"
                f"New stop loss price: {new_stop_loss_price}"
            )
            # send_telegram_message(message)
        else:
            logging.info(f"Trailing conditions not met for {symbol}. No update performed.")

    except Exception as e:
        logging.error(f"Error updating trailing stop and take profit: {e}")





def get_entry_price(client, symbol):
    try:
        # Получаем информацию о позиции для данного символа
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                # Возвращаем актуальную цену входа
                return float(pos['entryPrice'])
    except Exception as e:
        logging.error(f"Error fetching entry price for {symbol}: {e}")
        return None

def get_entry_price_via_trades(client, symbol):
    try:
        trades = client.futures_account_trades(symbol=symbol)
        if trades:
            # Возвращаем цену последней сделки
            return float(trades[-1]['price'])
    except Exception as e:
        logging.error(f"Error fetching trade data for {symbol}: {e}")
        return None






def cancel_take_profit_stop_loss_orders(client, trading_pair):
    try:
        trading_pair = trading_pair.replace(':USDT', '').replace('/', '')  # Clean symbol
        open_orders = client.futures_get_open_orders(symbol=trading_pair)
        for order in open_orders:
            if order['type'] in ['TAKE_PROFIT_MARKET', 'STOP_MARKET']:
                client.futures_cancel_order(symbol=trading_pair, orderId=order['orderId'])
                logging.info(f"Cancelled {order['type']} order for {trading_pair}.")
    except Exception as e:
        logging.error(f"Error cancelling take profit and stop loss orders for {trading_pair}: {e}")


def close_position(position_side, trading_pair, client):
    try:
        # Убедитесь, что trading_pair передается без "/"
        trading_pair = trading_pair.replace("/", "")  # Убираем "/"
        
        position_info = client.futures_position_information(symbol=trading_pair)
        for position in position_info:
            if position['positionSide'] == position_side:
                quantity = abs(float(position['positionAmt']))
                if quantity > 0:
                    if position_side == 'LONG':
                        logging.info(f"Closing LONG position for {trading_pair} with quantity {quantity}.")
                        client.futures_create_order(
                            symbol=trading_pair,
                            side='SELL',
                            type='MARKET',
                            quantity=quantity,
                            positionSide='LONG'  # Указание стороны позиции
                        )
                        send_telegram_message(f'❌⬆️   ✅⬇️Closed LONG, opened SHORT, {trading_pair}')
                    elif position_side == 'SHORT':
                        logging.info(f"Closing SHORT position for {trading_pair} with quantity {quantity}.")
                        client.futures_create_order(
                            symbol=trading_pair,
                            side='BUY',
                            type='MARKET',
                            quantity=quantity,
                            positionSide='SHORT'  # Указание стороны позиции
                        )
                        send_telegram_message(f'❌⬇️   ✅⬆️Closed SHORT, opened LONG {trading_pair}')
                else:
                    logging.info(f"No open position to close for {trading_pair} in {position_side}.")
    except Exception as e:
        logging.error(f"Error closing position for {trading_pair} ({position_side}): {e}")



def close_existing_positions(trading_pair, new_position_side, client):
    try:
        # Убираем символ "/" из trading_pair для корректного формата
        trading_pair = trading_pair.replace("/", "")  # Убираем "/"
        
        # Получаем информацию о текущих открытых позициях
        open_positions = client.futures_position_information(symbol=trading_pair)
        logging.info(f"Open positions response for {trading_pair}: {open_positions}")
        
        # Определяем количество открытых позиций
        active_positions = [p for p in open_positions if float(p['positionAmt']) != 0]
        logging.info(f"Number of open positions for {trading_pair}: {len(active_positions)}")
        
        # Если позиция уже открыта
        if len(active_positions) > 0:
            for position in active_positions:
                if new_position_side.lower() == 'long' and position['positionSide'] == 'SHORT':
                    logging.info(f"Closing SHORT position for {trading_pair} before opening LONG.")
                    close_position('SHORT', trading_pair, client)
                elif new_position_side.lower() == 'short' and position['positionSide'] == 'LONG':
                    logging.info(f"Closing LONG position for {trading_pair} before opening SHORT.")
                    close_position('LONG', trading_pair, client)
                else:
                    logging.info(f"Position already open for pair {trading_pair}. Skipping...")
        else:
            logging.info(f"No opposite position to close for {trading_pair}. Ready to open {new_position_side.upper()} position.")
    except Exception as e:
        logging.error(f"Error closing existing positions for {trading_pair}: {e}")
# def close_position(position_side, trading_pair, client):
#     try:
#         position_info = client.futures_position_information(symbol=trading_pair)
#         for position in position_info:
#             if position['positionSide'] == position_side:
#                 quantity = abs(float(position['positionAmt']))
#                 if quantity > 0:
#                     if position_side == 'LONG':
#                         logging.info(f"Closing LONG position for {trading_pair} with quantity {quantity}.")
#                         client.futures_create_order(
#                             symbol=trading_pair,
#                             side='SELL',
#                             type='MARKET',
#                             quantity=quantity,
#                             positionSide='LONG'  # Указание стороны позиции
#                         )
#                         send_telegram_message(f'❌⬆️   ✅⬇️Closed LONG, opened SHORT, {trading_pair}')
#                     elif position_side == 'SHORT':
#                         logging.info(f"Closing SHORT position for {trading_pair} with quantity {quantity}.")
#                         client.futures_create_order(
#                             symbol=trading_pair,
#                             side='BUY',
#                             type='MARKET',
#                             quantity=quantity,
#                             positionSide='SHORT'  # Указание стороны позиции
#                         )
#                         send_telegram_message(f'❌⬇️   ✅⬆️Closed SHORT, opened LONG {trading_pair}')
#                 else:
#                     logging.info(f"No open position to close for {trading_pair} in {position_side}.")
#     except Exception as e:
#         logging.error(f"Error closing position for {trading_pair} ({position_side}): {e}")


# def close_existing_positions(trading_pair, new_position_side, client):
#     try:
#         # Получаем информацию о текущих открытых позициях
#         open_positions = client.futures_position_information(symbol=trading_pair)
#         logging.info(f"Open positions response for {trading_pair}: {open_positions}")
        
#         # Определяем количество открытых позиций
#         active_positions = [p for p in open_positions if float(p['positionAmt']) != 0]
#         logging.info(f"Number of open positions for {trading_pair}: {len(active_positions)}")
        
#         # Если позиция уже открыта
#         if len(active_positions) > 0:
#             for position in active_positions:
#                 if new_position_side.lower() == 'long' and position['positionSide'] == 'SHORT':
#                     logging.info(f"Closing SHORT position for {trading_pair} before opening LONG.")
#                     close_position('SHORT', trading_pair, client)
#                 elif new_position_side.lower() == 'short' and position['positionSide'] == 'LONG':
#                     logging.info(f"Closing LONG position for {trading_pair} before opening SHORT.")
#                     close_position('LONG', trading_pair, client)
#                 else:
#                     logging.info(f"Position already open for pair {trading_pair}. Skipping...")
#         else:
#             logging.info(f"No opposite position to close for {trading_pair}. Ready to open {new_position_side.upper()} position.")
#     except Exception as e:
#         logging.error(f"Error closing existing positions for {trading_pair}: {e}")

# Функция для создания ордеров
def create_orders(client, trading_pair, position_size, take_profit_price, stop_loss_price, position_side_setting, position_side, file_path='pair_state.json'):
    max_retries = 30
    trading_pair = trading_pair.replace(':USDT', '').replace('/', '')

    # Загрузка текущего состояния пар
    pair_state = load_pair_state(file_path)
    last_state = pair_state.get(trading_pair, None)  # Получаем последнее состояние для текущей пары

    # Проверяем последнее состояние и не даем открыть противоположную позицию
    if last_state == 'LONG' and position_side == 'LONG':
        logging.info(f"Lost order was LONG for {trading_pair}, Whait SHORT.")
        return
    elif last_state == 'SHORT' and position_side == 'SHORT':
        logging.info(f"Lost order was SHORT for {trading_pair}, Whait LONG.")
        return

    for attempt in range(max_retries):
        try:
            # Check if there is an existing order for the trading pair
            if trading_pair in open_orders and (datetime.now() - open_orders[trading_pair]).total_seconds() < 30:
                logging.info(f"Order for pair {trading_pair} already exists. Skipping...")
                return

            # Check if there are any open positions for the trading pair
            open_positions = client.futures_position_information(symbol=trading_pair)
            logging.info(f"Open positions response for {trading_pair}: {open_positions}")

            # Count open positions where positionAmt is not '0.0'
            open_positions_count = sum(
                1 for pos in open_positions 
                if float(pos['positionAmt']) != 0 and pos['symbol'] == trading_pair
            )
            logging.info(f"Number of open positions for {trading_pair}: {open_positions_count}")

            if open_positions_count > 0:
                logging.info(f"Position already open for pair {trading_pair}. Skipping...")
                return

            # Limit number of open positions
            open_positions_count = count_open_positions(client, position_side)
            if open_positions_count is None or open_positions_count >= 30:
                logging.info(f"Exceeded number of open {position_side} positions. Skipping...")
                return

            # Create the market order
            market_order = client.futures_create_order(
                symbol=trading_pair,
                side=Client.SIDE_BUY if position_side == 'LONG' else Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=position_size,
                positionSide=position_side_setting
            )
            logging.info("Market order successfully created:")
            logging.info(market_order)

            open_orders[trading_pair] = datetime.now()
            current_price = float(client.get_symbol_ticker(symbol=trading_pair)['price'])
            balance = get_account_balance(client)
            message = (
                f"Opened {position_side} order for pair {trading_pair}\n"
                f"Position size: {position_size}\n"
                f"Current price: {current_price}\n"
                f"Take profit price: {take_profit_price}\n"
                f"Stop loss price: {stop_loss_price}\n"
                f"Current balance: {balance} USDT"
            )
            send_telegram_message(message)

            current_price = float(client.get_symbol_ticker(symbol=trading_pair)['price'])
            if (position_side == 'LONG' and (take_profit_price <= current_price or stop_loss_price >= current_price)) or \
               (position_side == 'SHORT' and (take_profit_price >= current_price or stop_loss_price <= current_price)):
                logging.error(f"Invalid take profit or stop loss price for {position_side} order: {trading_pair}")
                return

            cancel_take_profit_stop_loss_orders(client, trading_pair)

            # Create take profit order
            for attempt_tp in range(max_retries):
                try:
                    tp_order = client.futures_create_order(
                        symbol=trading_pair,
                        side=Client.SIDE_SELL if position_side == 'LONG' else Client.SIDE_BUY,
                        type="TAKE_PROFIT_MARKET",
                        quantity=position_size,
                        stopPrice=take_profit_price,
                        positionSide=position_side_setting
                    )
                    message = f"✅✅✅ Take profit order created for pair {trading_pair}"
                    send_telegram_message(message)
                    logging.info(tp_order)
                    break
                except (ConnectionError, HTTPError) as e:
                    logging.error(f"Error creating take profit order: {e}. Attempt {attempt_tp + 1} of {max_retries}")
                    time.sleep(5)
                    continue

            # Create stop loss order
            for attempt_sl in range(max_retries):
                try:
                    sl_order = client.futures_create_order(
                        symbol=trading_pair,
                        side=Client.SIDE_SELL if position_side == 'LONG' else Client.SIDE_BUY,
                        type="STOP_MARKET",
                        quantity=position_size,
                        stopPrice=stop_loss_price,
                        positionSide=position_side_setting
                    )
                    message = f"⛔️⛔️⛔️ Stop loss order created for pair {trading_pair}"
                    send_telegram_message(message)
                    logging.info(sl_order)
                    break
                except (ConnectionError, HTTPError) as e:
                    logging.error(f"Error creating stop loss order: {e}. Attempt {attempt_sl + 1} of {max_retries}")
                    time.sleep(5)
                    continue

            # Обновляем состояние пары после успешного открытия позиции
            new_state = 'LONG' if position_side == 'LONG' else 'SHORT'
            update_pair_state(file_path, trading_pair, new_state)

            return

        except (ConnectionError, HTTPError) as e:
            send_telegram_message(f"Error creating order: {e}. Attempt {attempt + 1} of {max_retries}")
            time.sleep(10)

    logging.error(f"Failed to create orders after {max_retries} attempts.")



def check_btc_volatility(binance_client):
    symbol = 'BTCUSDT'
    df = fetch_ohlcv(symbol, timeframe='5m')
    if df is not None and not df.empty:
        last_candle = df.iloc[-1]
        price_diff = last_candle['high'] - last_candle['low']
        if price_diff > 3000:
            message=('BTC >300 on 5 min')
            send_telegram_message(message)
            logging.info(f"Price difference for BTCUSDT is {price_diff}, greater than 300. Skipping processing.")
            return True
    return False

def handle_api_error(e):
    if isinstance(e, BinanceAPIException):
        if e.code == -1003:
            # Попробуем извлечь предполагаемое время разблокировки, если оно указано в сообщении
            ban_time = None
            try:
                ban_time = int(e.message.split("until ")[1])  # Парсинг времени из сообщения
                unlock_time = ban_time / 1000
                wait_time = max(0, unlock_time - time.time())
            except (IndexError, ValueError):
                # Если времени разблокировки нет в сообщении, устанавливаем ожидание вручную
                wait_time = 120  # Начальное время ожидания (в секундах)

            # Логгируем сообщение с временем ожидания
            message = f"API rate limit exceeded. Waiting for {wait_time / 60:.2f} minutes before retrying."
            logging.warning(message)
            send_telegram_message(message)  # Отправляем сообщение в Telegram

            # Ожидание с постепенным увеличением интервала проверок
            retry_delay = 60  # Начальная задержка проверки (в секундах)
            while True:
                remaining_time = unlock_time - time.time() if ban_time else wait_time
                if remaining_time <= 0:
                    break
                logging.info(f"Waiting for {remaining_time / 60:.2f} minutes until unlock.")
                time.sleep(min(remaining_time, retry_delay))
                retry_delay = min(retry_delay * 2, 300)  # Увеличиваем интервал до 5 минут

        elif e.code == -1021:
            # Обрабатываем ошибку рассинхронизации времени
            logging.warning("Timestamp for this request is outside of the recvWindow. Synchronizing time with Binance server.")
            try:
                server_time = binance_client.time()['serverTime'] / 1000  # Получаем серверное время в секундах
                local_time = time.time()
                offset = server_time - local_time
                # Устанавливаем корректировку для времени
                time.sleep(1)  # Ждем секунду перед повторной отправкой запроса
                message = f"Time synchronized with Binance. Offset: {offset:.2f} seconds."
                logging.info(message)
                send_telegram_message(message)
            except Exception as sync_error:
                logging.error(f"Failed to synchronize time with Binance: {sync_error}")
                send_telegram_message(f"Failed to synchronize time with Binance: {sync_error}")
        elif e.code == -2015:
            # Обработка ошибки -2015 (недействительный API-ключ, IP или права доступа)
            message = ("Invalid API-key, IP, or permissions for action. "
                       "Please check API settings, permissions, and IP whitelist on Binance.")
            
            send_telegram_message(message)  # Отправляем уведомление в Telegram
            logging.error(message)
        else:
            logging.error(f"Unexpected API error: {e}")
            send_telegram_message(f"Unexpected API error: {e}")


# Функция синхронизации времени
def synchronize_time(binance_client):
    try:
        # Получаем время сервера Binance Futures
        server_time = binance_client.futures_time()['serverTime']
        local_time = int(time.time() * 1000)
        time_offset = server_time - local_time
        logging.info(f"Time synchronized. Offset: {time_offset} ms.")
        return time_offset
    except Exception as e:
        logging.error(f"Initial time synchronization failed: {e}")
        raise SystemExit("Cannot synchronize time with Binance server.")

def process_signals():
    """
    Выполняет проверку сигналов и осуществляет торговые действия.
    """
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Running signal check at {current_time}")
        
        cleanup_orders(binance_client)
        
        if check_btc_volatility(binance_client):
            print("High BTC volatility detected. Skipping this round.")
            return

        markets = exchange.load_markets()
        usdt_pairs = [symbol for symbol in markets if symbol.endswith('USDT')]
        filtered_pairs = [symbol for symbol in usdt_pairs if symbol not in blacklist]
        limited_pairs = filtered_pairs[:10]

        for symbol in limited_pairs:
            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"Processing pair: {symbol} on 5m timeframe at {current_time}")

                df = fetch_ohlcv(symbol)
                if df is None:
                    continue

                signal, price, position_side = check_signals(df)

                if signal:
                    time.sleep(0.2)
                    symbol_no_slash = symbol.replace('/', '')
                    message = f"🟦🟦🟦{symbol} {signal} at price {price}. Position side: {position_side}🟦🟦🟦"
                    logging.info(message)
                    print(message)

                    step_size, tick_size, min_notional = get_symbol_info(binance_client, symbol_no_slash)
                    if step_size is None or min_notional is None:
                        continue

                    set_margin_mode(binance_client, symbol_no_slash, margin_mode)

                    balance = get_account_balance(binance_client)
                    
                    binance_client.futures_change_leverage(symbol=symbol_no_slash, leverage=leverage)

                    ticker = binance_client.get_symbol_ticker(symbol=symbol_no_slash)
                    current_price = float(ticker['price'])

                    position_size = calculate_position_size(balance, position_size_percent, leverage, current_price, step_size, min_notional)
                    if position_size is None:
                        continue

                    take_profit_price, stop_loss_price = calculate_prices(current_price, take_profit_percent, stop_loss_percent, position_side, tick_size)

                    position_mode = binance_client.futures_get_position_mode()
                    position_side_setting = 'BOTH' if not position_mode['dualSidePosition'] else position_side
                    close_existing_positions(symbol, position_side, binance_client)
                    create_orders(binance_client, symbol_no_slash, position_size, take_profit_price, stop_loss_price, position_side_setting, position_side)

            except Exception as e:
                logging.error(f"Error processing pair {symbol}: {e}")

    except BinanceAPIException as e:
        handle_api_error(e)
    except Exception as e:
        logging.error(f"Error during signal processing: {e}")

def schedule_check():
    now = datetime.now()
    # Найти следующую пятиминутку
    next_minute = (now.minute // 5 + 1) * 5
    next_check_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)
    
    # Сдвиг на 30 секунд назад для выполнения перед закрытием пятиминутки
    next_check_time -= timedelta(seconds=50)

    # Рассчитать оставшееся время до следующей проверки
    wait_seconds = (next_check_time - now).total_seconds()

    # Если время для следующей проверки уже прошло (например, 14:40:00, когда текущее время 14:42:00)
    if wait_seconds < 0:
        # Переносим на следующую пятиминутку
        next_check_time += timedelta(minutes=5)
        wait_seconds = (next_check_time - now).total_seconds()

    print(f"Next signal check at {next_check_time}. Waiting {int(wait_seconds)} seconds.")
    
    time.sleep(wait_seconds)
    process_signals()
    schedule_check()  # Повторно вызываем функцию для продолжения работы

if __name__ == "__main__":
    print("Bot started. Waiting for the first check.")
    schedule_check()  # Запуск первой проверки