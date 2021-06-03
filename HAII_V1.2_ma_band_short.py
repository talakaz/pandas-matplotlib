import pandas as pd
import numpy as np
from numba import jit
import sqlite3
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdate
import time
from itertools import product
import talib

params = {
    # 主動參數 | fast和signal要從2開始不然EMA會報錯
    'strategy': 'P1_MACD_SMA_Long',
    'year': '2019', 'res': '5m',
    'size': 1000, 'fee_rate': 0.0015,
    # var1 = ma_channel, var2 = ma_fast, var3 = bias, var4 = angle
    'var_code': '1.ma_slow 2.ma_fast 3.ma_bias 4.slow_angle',
    'var_1_range': [50, 51, 1], 'var_2_range': [18, 19, 1], 'var_3_range': [13, 14, 1], 'var_4_range': [13, 14, 1],

    # 被動參數
    'var_1': None, 'var_2': None, 'var_3': None, 'var_4': None}

db = {'path': r'C:\Users\Talak\BitMEX_BackTest\SQLite3\HistoricalData.db',
      'data_table': 'BitMEX_' + params['year'] + '_' + params['res'],
      'report_table': 'Report_' + params['year'] + '_' + params['res'] + '_dev'}

divisors = {'1d': 1, '1h': 24, '5m': 288, '1m': 1440}

reports = {'id': None, 'var_code': None, 'var_1': None, 'var_2': None, 'var_3': None, 'var_4': None,
           '總天數': None, '日交易': None, '勝率': None, '日獲利': None, '總獲利': None}

conn = sqlite3.connect(db['path'])
c = conn.cursor()
c.execute("PRAGMA page_size = 32768;")
df = pd.read_sql('SELECT * FROM %s' % (db['data_table']), conn)


def gen_params():
    # 由指定範圍產生組合參數
    a_values = pd.Series(np.arange(params['var_1_range'][0], params['var_1_range'][1], params['var_1_range'][2]))
    b_values = pd.Series(np.arange(params['var_2_range'][0], params['var_2_range'][1], params['var_2_range'][2]))
    c_values = pd.Series(np.arange(params['var_3_range'][0], params['var_3_range'][1], params['var_3_range'][2]))
    d_values = pd.Series(np.arange(params['var_4_range'][0], params['var_4_range'][1], params['var_4_range'][2]))

    values = pd.DataFrame(list(product(a_values, b_values, c_values, d_values)), columns=['var_1', 'var_2', 'var_3', 'var_4'])

    # 過濾掉fast大於slow的組合並重新建立索引
    dropindex = []
    for i in range(len(values['var_1'].index)):
        if values['var_2'][i] >= values['var_1'][i]:
            dropindex.append(i)
    values = values.drop(dropindex)
    values = values.reset_index(drop=True)
    values['id'] = list(range(0, len(values)))
    # 取出資料庫最後一筆資料索引
    cmd = 'SELECT * FROM %s ORDER BY id DESC LIMIT 1' % (db['report_table'])
    c.execute(cmd)
    lastid = (c.fetchall()[0][0])
    lastid = 0
    # 過濾掉已儲存的資料回傳
    droplist = list(range(0, lastid))
    values = values.drop(droplist)
    values = values.reset_index(drop=True)
    return values


# 產生訊號
@jit
def get_signal(close, slow, fast, bias, angle):
    close = close
    slow = slow
    fast = fast
    bias = bias
    angle = angle
    signal = []
    hold = False
    for i in range(0, len(close)):
        signal.append(np.nan)
        if angle[i] < params['var_4']:
            if bias[i] > params['var_3']:
                if hold is False:
                    signal[i] = True
                    hold = True
        if hold is True and fast[i] < slow[i]:
            signal[i] = False
            hold = False

    # print(i, close[i], l_last, hold, var_3[i], l_stop, signal[i])

    return signal


def backTestGo():
    df['slow'] = talib.SMA(df['close'].values, params['var_1'])
    df['fast'] = talib.SMA(df['close'].values, params['var_2'])
    df['bias'] = ((df['fast'] - df['slow']) / df['slow']) * 100
    df['shift'] = df['slow'].shift(20)
    df['angle'] = ((df['slow'] - df['shift']) / df['shift']) * 100

    # print(df[['slow', 'fast', 'bias', 'angle']].to_string())

    # print(df[['timestamp', 'close', 'macdline', 'signalline', 'var_3', 'var_3a', 'sma']].to_svar_4ng())  # 已驗證?
    df2 = df[['close', 'slow', 'fast', 'bias', 'angle']]
    df2 = df2.dropna()
    df3 = pd.DataFrame(df2)
    df3['signal'] = get_signal(df3['close'].values, df3['slow'].values, df3['fast'].values, df3['bias'].values, df3['angle'].values)
    df4 = pd.DataFrame(df3[['close', 'signal']])
    df4 = df4.dropna()
    # print(df4[['close', 'signal']].to_string())  # 已驗證?

    df4['shift'] = df4['close'].shift(1)

    df4['profit'] = np.nan
    df4['profit'] = np.where(df4['signal'] == 0, (((df4['shift'] - df4['close']) / df4['shift']) * params['size']) - (params['size'] * params['fee_rate']), df4['profit'])

    # print(df4[['shift', 'close', 'var_3', 'short', 'profit']].to_svar_4ng(), df4['profit'].sum())

    df4 = df4.dropna()
    # 統計數據
    total_days = np.around((len(df) / divisors[params['res']]))
    total_trades = len(df4)
    wins = len(df4[df4['profit'] > 0])
    daily_trades = np.around(total_trades / total_days, decimals=1)
    try:
        win_rate = np.around(wins / total_trades, decimals=2)
    except Exception as e:
        print(e.args)
        win_rate = 0
    totaprofits = df4['profit'].sum()
    daily_profit = np.around(totaprofits / total_days, decimals=1)
    print('總交易%d次 日交易%.1f次 勝率%.2f 日獲利%.1f 總獲利 %.1f' % (total_trades, daily_trades, win_rate, daily_profit, totaprofits))

    # 寫入暫存區
    reports['var_1'] = params['var_1']
    reports['var_2'] = params['var_2']
    reports['var_3'] = params['var_3']
    reports['var_4'] = params['var_4']
    reports['總天數'] = total_days
    reports['日交易'] = daily_trades
    reports['勝率'] = win_rate
    reports['日獲利'] = daily_profit
    reports['總獲利'] = totaprofits


# # 產出圖表及報表
# df['timestamp'] = pd.to_datetime(df['timestamp'])
# df['timestamp'] = df['timestamp'].map(mdate.date2num)
# df['sell'] = df3['signal'] > 0
# df['sell'] = np.where(df['sell'] == 1, df['close'], np.nan)
# df['deals'] = df3['signal'] < 1
# df['deals'] = np.where(df['deals'] == 1, df['close'], np.nan)
# # style.use('ggplot')
# ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
# ax1.xaxis_date()
# # ax1.plot(df['timestamp'], df['close'], color='#FFB548', label='Price')
# candlestick_ohlc(ax1, df.values, width=0.002, colorup='g')
# ax1.plot(df['timestamp'], df['sell'], marker=6, color='#12B347', markersize=10, label='Sell Signal')
# ax1.plot(df['timestamp'], df['deals'], marker=7, color='#D61E42', markersize=10, label='Deal Signal')
#
# ax1.legend(loc='upper right')
# ax2.plot(df['timestamp'], df['bias'], color='#0A2FFF', label='bias')
# ax2.plot(df['timestamp'], df['angle'], color='#CC3400', label='angle')
# plt.show()


def update_db():
    values = gen_params()
    for i in range(0, len(values)):
        print('----- %s策略 | %s %s 總共%d組排列組合 | 第%d組分析中... -----' % (params['strategy'], params['year'], params['res'], len(values), i))
        timestart = time.process_time()
        # print(i)
        params['var_1'] = values['var_1'][i]
        params['var_2'] = values['var_2'][i]
        params['var_3'] = values['var_3'][i] * 0.1
        params['var_4'] = values['var_4'][i] * 0.1
        reports['id'] = values['id'][i]
        reports['var_code'] = params['var_code']

        backTestGo()
        reports['var_1'] = str(reports['var_1'])
        reports['var_2'] = int(reports['var_2'])
        reports['var_3'] = float(reports['var_3'])
        reports['var_4'] = float(reports['var_4'])
        reports['id'] = int(reports['id'])
        print(reports)
        cmd = "REPLACE INTO %s (id, 變數代號, 變數1, 變數2, 變數3, 變數4, 總天數, 日交易, 勝率, 日獲利, 總獲利) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)" % (db['report_table'])
        arguments = (reports['id'], reports['var_code'], reports['var_1'], reports['var_2'], reports['var_3'], reports['var_4'], reports['總天數'], reports['日交易'], reports['勝率'], reports['日獲利'], reports['總獲利'])
        c.execute(cmd, arguments)
        conn.commit()
        reports['id'] = None
        reports['var_code'] = None
        reports['var_1'] = None
        reports['var_2'] = None
        reports['var_3'] = None
        reports['var_4'] = None
        reports['總天數'] = None
        reports['日交易'] = None
        reports['勝率'] = None
        reports['日獲利'] = None
        reports['總獲利'] = None
        timeend = time.process_time()
        print('----- 執行時間', timeend - timestart)


update_db()
