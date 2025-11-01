# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:27:21 2023

@author: Vipul Tanwar
"""

import talib
import datetime
import numpy as np
import pandas as pd
from talib import abstract
from fastquant import get_crypto_data


'''
- The following function works with default inputs but you can provide your own inputs as well as per requirement of symbol/time period
- If no input data is provided, the function automatically extracts data from fastquant get_crypto_data api which connects with binance data api
- if you want to provide your own OHLCV input, use the dataset format as extracted in line 387
- Use the provided main function code to integrate it in your system the way it works the best. 

'''

def LiveSignalGenerator(sym_string ="BTC/USDT",time_resolution_string = '5m',input_data=None):
    

    
    if input_data is None:
        
        strt_tm = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        end_tm = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(sym_string,time_resolution_string,strt_tm,end_tm)
        time_series_df = get_crypto_data(ticker=sym_string, start_date=strt_tm,end_date=end_tm,time_resolution=time_resolution_string)
        
    else:
        
        time_series_df = input_data
   
    fwd = [('Average Price', 2),
     ('Typical Price', 2),
     ('Weighted Close Price', 2),
     ('Median Price', 2),
     ('Average Price', 1),
     ('Weighted Close Price', 1),
     ('Median Price', 1),
     ('Typical Price', 1),
     ('STOCHRSI', 2),
     ('STOCHF', 2),
     ('STOCHRSI', 1),
     ('STOCHF', 1),
     ('STOCH', 1),
     ('WILLR', 2),
     ('MINMAX', 1)]

    rvs = [('KAMA', 1),
     ('BOP', 2),
     ('T3', 1),
     ('EMA', 1),
     ('MA', 1),
     ('T3', 2),
     ('CORREL', 1),
     ('OBV', 2),
     ('TEMA', 2),
     ('ADX', 1),
     ('OBV', 1),
     ('TEMA', 1),
     ('AD', 2),
     ('AD', 1),
     ('TRIMA', 1)]


    signals = getTACombinedSignals(time_series_df,True)

    x = pd.Series(index=signals.index)
    x = 0
    for strat in rvs:
        x += -1*signals[strat[0]].shift(strat[1]-1)
    for strat in fwd:
        x += signals[strat[0]].shift(strat[1]-1)    

    if x.iloc[-1] < 0:
        zignal = -1 

    elif x.iloc[-1] > 0:
        zignal = 1
        
    else:
        
        zignal = 0

    return zignal



def getTACombinedSignals(input_df, returnall=False):
    all_funcs = talib.get_functions()

    # Creating a pandas dataframe which will contain strategy signal in each column
    signals = pd.DataFrame(index=input_df.index)
    # Adding the most basic indicator which will‸ be used in many other signals
    signals["MA"] = np.where(abstract.Function("MA")(input_df) < input_df.close, 1, np.where(abstract.Function("MA")(input_df) > input_df.close, -1, 0))

    signals = _get_candle_pattern_signals(input_df, signals, all_funcs)
    signals = _get_price_transform_signals(input_df, signals, all_funcs)
    signals = _get_volume_based_signals(input_df, signals)
    signals = _get_volatility_based_signals(input_df, signals, signals["MA"])
    signals = _get_statistical_signals(input_df, signals, signals["MA"])
    signals = _get_momentum_signals(input_df, signals, signals["MA"])
    signals = _get_overlap_studies_signals(input_df, signals)
    signals = _get_math_operator_signals(input_df, signals)
    signals = _get_hilbert_transform_signals(input_df, signals, signals["MA"], signals["RSI"])

    if returnall:
        return signals
    else:
        return signals.iloc[-1]


def _get_candle_pattern_signals(input_df, signals, all_funcs):
    # Candle Pattern Related Charting Indicators (1-61)
    for func in all_funcs:
        if "CDL" in abstract.Function(func).info['name']:
            signals[abstract.Function(func).info['display_name']] = np.where(abstract.Function(func)(input_df) > 0, 1, np.where(abstract.Function(func)(input_df) < 0, -1, 0))
    return signals


def _get_price_transform_signals(input_df, signals, all_funcs):
    # Price Transform Technical Indicators (62-66)
    for func in all_funcs:
        if "PRICE" in abstract.Function(func).info['name']:
            signals[abstract.Function(func).info['display_name']] = np.where(abstract.Function(func)(input_df) > input_df['close'], 1, np.where(abstract.Function(func)(input_df) < input_df['close'], -1, 0))
    return signals


def _get_volume_based_signals(input_df, signals):
    # Volume Based Indicator (67-69)
    ### OBV: When OBV is above its 5 day mean, buy else sell
    signals["OBV"] = np.where(abstract.Function("OBV")(input_df) > abstract.Function("OBV")(input_df).rolling(5).mean().fillna(method="bfill"), 1, 0)
    ### Chaikin Oscillator: When signal crosses 0 from below, buy. Vice versa for sell
    signals["ADOSC"] = np.where(((abstract.Function("ADOSC")(input_df) >= 0) & (abstract.Function("ADOSC")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("ADOSC")(input_df) < 0) & (abstract.Function("ADOSC")(input_df).shift(1) >= 0)), -1, 0))
    ### Chaikin A/D Line: When signal is above its 5 day mean, buy else sell
    signals["AD"] = np.where(abstract.Function("AD")(input_df) > abstract.Function("AD")(input_df).rolling(5).mean().fillna(method="bfill"), 1, 0)
    return signals


def _get_volatility_based_signals(input_df, signals, ma_signal):
    # Volatility Based Indicators (70)
    ### Normalized Average True Range: Strong trend when value above 0.6. We do not consider it to be directional but supporting the directional indicators such as MACD
    ####Out of True Range, Average True Range and Normalized Average True Range: We only use the last one, rest two are inefficient versions of the same
    signals["NATR"] = np.where(abstract.Function("NATR")(input_df) > 0.6, 1, 0)
    signals["NATR"] = ma_signal * signals["NATR"]
    return signals


def _get_statistical_signals(input_df, signals, ma_signal):
    # Statistical Indicators (71-74)
    ### Beta: Buy when beta > 1.5 , sell when beta < - 1.5
    signals["BETA"] = np.where(abstract.Function("BETA")(input_df) > 1.5, 1, np.where(abstract.Function("BETA")(input_df) < -1.5, -1, 0))
    ### Correlation: Trending when Correlation > 0.85, else not trending
    signals["CORREL"] = np.where(abstract.Function("CORREL")(input_df) > 0.85, 1, 0)
    signals["CORREL"] = ma_signal * signals["CORREL"]
    ### Linear Regression: Buy/Sell when crossovers with Close price
    signals["LINEARREG"] = np.where(((input_df.close >= abstract.Function("LINEARREG")(input_df)) & (input_df.close.shift(1) < abstract.Function("LINEARREG")(input_df).shift(1))), 1, np.where(((input_df.close <= abstract.Function("LINEARREG")(input_df)) & (input_df.close.shift(1) > abstract.Function("LINEARREG")(input_df).shift(1))), -1, 0))
    ### TimeSeriesForecast: Buy/Sell when crossovers with Close price
    signals["TSF"] = np.where(((input_df.close >= abstract.Function("TSF")(input_df)) & (input_df.close.shift(1) < abstract.Function("TSF")(input_df).shift(1))), 1, np.where(((input_df.close <= abstract.Function("TSF")(input_df)) & (input_df.close.shift(1) > abstract.Function("TSF")(input_df).shift(1))), -1, 0))
    return signals


def _get_momentum_signals(input_df, signals, ma_signal):
    # Momentum Indicators (75-95)
    ### ADX: Trending when ADX > 25, else not trending
    signals["ADX"] = np.where(abstract.Function("ADX")(input_df) > 25, 1, 0)
    signals["ADX"] = ma_signal * signals["ADX"]
    ### ADXR: Buy when crossovers ADX < ADXR, vice versa
    signals["ADXR"] = np.where(((abstract.Function("ADXR")(input_df) >= abstract.Function("ADX")(input_df)) & (abstract.Function("ADXR")(input_df).shift(1) < abstract.Function("ADX")(input_df).shift(1))), 1, np.where(((abstract.Function("ADXR")(input_df) <= abstract.Function("ADX")(input_df)) & (abstract.Function("ADXR")(input_df).shift(1) > abstract.Function("ADX")(input_df).shift(1))), -1, 0))
    ### APO: Buy when crossovers APO < 0, vice versa
    signals["APO"] = np.where(((abstract.Function("APO")(input_df) >= 0) & (abstract.Function("APO")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("APO")(input_df) <= 0) & (abstract.Function("APO")(input_df).shift(1) > 0)), -1, 0))
    ### AROON OSC: Buy when ADX > 40 Sell when < -40
    signals["AROONOSC"] = np.where(abstract.Function("AROONOSC")(input_df) > 40, 1, np.where(abstract.Function("AROONOSC")(input_df) < -40, -1, 0))
    ### Balance of Power: Buy when crossovers BOP > 0 Sell when vice versa
    signals["BOP"] = np.where(((abstract.Function("BOP")(input_df) >= 0) & (abstract.Function("BOP")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("BOP")(input_df) <= 0) & (abstract.Function("BOP")(input_df).shift(1) > 0)), -1, 0))
    ### Commodity channel index: Buy when CCI > 100 Sell when CCI < 100
    signals["CCI"] = np.where(abstract.Function("CCI")(input_df) > 100, 1, np.where(abstract.Function("CCI")(input_df) < -100, -1, 0))
    ### Chande Momentum Oscillator: Buy when CMO < -50 Sell when CMO > 50
    signals["CMO"] = np.where(abstract.Function("CMO")(input_df) < -50, 1, np.where(abstract.Function("CMO")(input_df) > 50, -1, 0))
    ### Directional Movement Index: When DX is above its 5 period mean, buy else sell
    signals["DX"] = np.where(abstract.Function("DX")(input_df) > abstract.Function("DX")(input_df).rolling(5).mean().fillna(method="bfill"), 1, np.where(abstract.Function("DX")(input_df) < abstract.Function("DX")(input_df).rolling(5).mean().fillna(method="bfill"), -1, 0))
    ### MACD: Buy when crossovers MACDhist < 0 Sell when vice versa
    signals["MACD"] = np.where(((abstract.Function("MACD")(input_df).macdhist >= 0) & (abstract.Function("MACD")(input_df).macdhist.shift(1) < 0)), -1, np.where(((abstract.Function("MACD")(input_df).macdhist <= 0) & (abstract.Function("MACD")(input_df).macdhist.shift(1) > 0)), 1, 0))
    ### MACD EXT: Buy when crossovers MACDhist < 0 Sell when vice versa
    signals["MACDEXT"] = np.where(((abstract.Function("MACDEXT")(input_df).macdhist >= 0) & (abstract.Function("MACDEXT")(input_df).macdhist.shift(1) < 0)), -1, np.where(((abstract.Function("MACDEXT")(input_df).macdhist <= 0) & (abstract.Function("MACDEXT")(input_df).macdhist.shift(1) > 0)), 1, 0))
    ### MACD FIX: Buy when crossovers MACDhist < 0 Sell when vice versa
    signals["MACDFIX"] = np.where(((abstract.Function("MACDFIX")(input_df).macdhist >= 0) & (abstract.Function("MACDFIX")(input_df).macdhist.shift(1) < 0)), -1, np.where(((abstract.Function("MACDFIX")(input_df).macdhist <= 0) & (abstract.Function("MACDFIX")(input_df).macdhist.shift(1) > 0)), 1, 0))
    ### Money Flow index: Buy when MFI < 30 Sell when MFI > 80 (Overbought and oversold)
    signals["MFI"] = np.where(abstract.Function("MFI")(input_df) < 30, 1, np.where(abstract.Function("MFI")(input_df) > 80, -1, 0))
    ### Momentum Indicator: Buy when MOM > 100 Sell when MOM < -100
    signals["MOM"] = np.where(abstract.Function("MOM")(input_df) > 100, 1, np.where(abstract.Function("MOM")(input_df) < -100, -1, 0))
    ### Rate of Change Indicator: Buy when ROC > 0.25 Sell when ROC < -0.25
    signals["ROC"] = np.where(abstract.Function("ROC")(input_df) > 0.25, 1, np.where(abstract.Function("ROC")(input_df) < -0.25, -1, 0))
    ### Relative Strength Index: Buy when RSI < 30 Sell when RSI > 70 (Overbought and oversold)
    signals["RSI"] = np.where(abstract.Function("RSI")(input_df) < 30, 1, np.where(abstract.Function("RSI")(input_df) > 70, -1, 0))
    ### Stochastic: Buy when STOCH.slowk < 20 Sell when STOCH.slowk > 80 (Overbought and oversold)
    signals["STOCH"] = np.where(abstract.Function("STOCH")(input_df).slowk < 20, 1, np.where(abstract.Function("STOCH")(input_df).slowk > 80, -1, 0))
    ### Stochastic Fast: Buy when STOCH.slowk < 20 Sell when STOCH.slowk > 80 (Overbought and oversold)
    signals["STOCHF"] = np.where(abstract.Function("STOCHF")(input_df).fastk < 20, 1, np.where(abstract.Function("STOCHF")(input_df).fastk > 80, -1, 0))
    ### Stochastic RSI: Buy when STOCH.fastk < 20 Sell when STOCH.fastk > 80 (Overbought and oversold)
    signals["STOCHRSI"] = np.where(abstract.Function("STOCHRSI")(input_df).fastk < 20, 1, np.where(abstract.Function("STOCHRSI")(input_df).fastk > 80, -1, 0))
    ### Triple Exp Avg: Buy when TRIX crossovers < 0 Sell when vice versa
    signals["TRIX"] = np.where(((abstract.Function("TRIX")(input_df) >= 0) & (abstract.Function("TRIX")(input_df).shift(1) < 0)), -1, np.where(((abstract.Function("TRIX")(input_df) <= 0) & (abstract.Function("TRIX")(input_df).shift(1) > 0)), 1, 0))
    ### Ultimate Oscillator: Buy when ULTOSC < 30 Sell when ULTOSC > 70 (Overbought and oversold)
    signals["ULTOSC"] = np.where(abstract.Function("ULTOSC")(input_df) < 30, 1, np.where(abstract.Function("ULTOSC")(input_df) > 70, -1, 0))
    ### Williams %R: Buy when WILLR < 30 Sell when WILLR > 70 (Overbought and oversold)
    signals["WILLR"] = np.where(abstract.Function("WILLR")(input_df) < -80, 1, np.where(abstract.Function("WILLR")(input_df) > -20, -1, 0))
    return signals


def _get_overlap_studies_signals(input_df, signals):
    # Overlap Studies (96-108)
    ### Moving Averages: Buy when price > MA else sell
    signals["MA"] = np.where(abstract.Function("MA")(input_df) < input_df.close, 1, np.where(abstract.Function("MA")(input_df) > input_df.close, -1, 0))
    ### Exponential Moving Averages: Buy when price > EMA else sell
    signals["EMA"] = np.where(abstract.Function("EMA")(input_df) < input_df.close, 1, np.where(abstract.Function("EMA")(input_df) > input_df.close, -1, 0))
    ### Double Exponential Moving Averages: Buy when price > DEMA else sell
    signals["DEMA"] = np.where(abstract.Function("DEMA")(input_df) < input_df.close, 1, np.where(abstract.Function("DEMA")(input_df) > input_df.close, -1, 0))
    ### Kaufman Adaptive Moving Average: Buy when price > KAMA else sell
    signals["KAMA"] = np.where(abstract.Function("KAMA")(input_df) < input_df.close, 1, np.where(abstract.Function("KAMA")(input_df) > input_df.close, -1, 0))
    ### MESA Adaptive Moving Average: Buy when fama > mama else sell
    signals["MAMA"] = np.where(abstract.Function("MAMA")(input_df).mama > abstract.Function("MAMA")(input_df).fama, 1, np.where(abstract.Function("MAMA")(input_df).mama < abstract.Function("MAMA")(input_df).fama, -1, 0))
    ### MIDPOINT: When MP is above its 5 period mean, buy else sell
    signals["MIDPOINT"] = np.where(abstract.Function("MIDPOINT")(input_df) > abstract.Function("MIDPOINT")(input_df).rolling(5).mean().fillna(method="bfill"), 1, np.where(abstract.Function("MIDPOINT")(input_df) < abstract.Function("MIDPOINT")(input_df).rolling(5).mean().fillna(method="bfill"), -1, 0))
    ### MIDPRICE: When MPX is above its 5 period mean, buy else sell
    signals["MIDPRICE"] = np.where(abstract.Function("MIDPRICE")(input_df) > abstract.Function("MIDPRICE")(input_df).rolling(5).mean().fillna(method="bfill"), 1, np.where(abstract.Function("MIDPRICE")(input_df) < abstract.Function("MIDPRICE")(input_df).rolling(5).mean().fillna(method="bfill"), -1, 0))
    ### Parabolic SAR: Buy/Sell when crossovers with Close price
    signals["SAR"] = -1 * np.where(((input_df.close >= abstract.Function("SAR")(input_df)) & (input_df.close.shift(1) < abstract.Function("SAR")(input_df).shift(1))), 1, np.where(((input_df.close <= abstract.Function("SAR")(input_df)) & (input_df.close.shift(1) > abstract.Function("SAR")(input_df).shift(1))), -1, 0))
    ### Parabolic SAR Extended: Buy when >0 else sell
    signals["SAREXT"] = -1 * np.where(((abstract.Function("SAREXT")(input_df) >= 0) & (abstract.Function("SAREXT")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("SAREXT")(input_df) <= 0) & (abstract.Function("SAREXT")(input_df).shift(1) > 0)), -1, 0))
    ### Triple Exponential Moving Average: Buy when price > T3 else sell
    signals["T3"] = np.where(abstract.Function("T3")(input_df) < input_df.close, 1, np.where(abstract.Function("T3")(input_df) > input_df.close, -1, 0))
    ### Triple Exponential Moving Average: Buy when price > TEMA else sell
    signals["TEMA"] = np.where(abstract.Function("TEMA")(input_df) < input_df.close, 1, np.where(abstract.Function("TEMA")(input_df) > input_df.close, -1, 0))
    ### Triangular Moving Average: Buy when price > TRIMA else sell
    signals["TRIMA"] = np.where(abstract.Function("TRIMA")(input_df) < input_df.close, 1, np.where(abstract.Function("TRIMA")(input_df) > input_df.close, -1, 0))
    ### Weighted Moving Average: Buy when price > WMA else sell
    signals["WMA"] = np.where(abstract.Function("WMA")(input_df) < input_df.close, 1, np.where(abstract.Function("WMA")(input_df) > input_df.close, -1, 0))
    return signals


def _get_math_operator_signals(input_df, signals):
    # Math Operators (109)
    ### Max and Min: Buy when price = Min, sell when price = Max
    signals["MINMAX"] = np.where(abstract.Function("MIN")(input_df) == input_df.close, 1, np.where(abstract.Function("MAX")(input_df) == input_df.close, -1, 0))
    return signals


def _get_hilbert_transform_signals(input_df, signals, ma_signal, rsi_signal):
    # Hilbert Transform Cycle Indicators (110-111)
    ## Dominant Cycle Period: Trend confirmed when HT_DCPERIOD > 25
    signals["HT_DCPERIOD"] = np.where(abstract.Function("HT_DCPERIOD")(input_df) > 25, 1, 0)
    signals["HT_DCPERIOD"] = ma_signal * signals["HT_DCPERIOD"]
    ## Trend vs Cycle Mode: Use trending strategy (MA) when HT_TRENDMODE = 1 else Use cyclic strategy (RSI)
    signals["HT_TRENDMODE"] = np.where(abstract.Function("HT_TRENDMODE")(input_df) == 1, ma_signal, rsi_signal)
    return signals




if __name__ == "__main__": 
    
    print("Testing LiveSignalGenerator without any inputs\n")
    
    print(LiveSignalGenerator())
    
    print("Testing LiveSignalGenerator with input variables\n")
    
    print(LiveSignalGenerator(sym_string="LTC/USDT",time_resolution_string="1h"))
    
    print("Testing LiveSignalGenerator with input data\n")
    
    crypto = get_crypto_data("BTC/USDT", "2023-02-01", "2023-05-22",time_resolution='15m')

    print(LiveSignalGenerator(input_data=crypto))