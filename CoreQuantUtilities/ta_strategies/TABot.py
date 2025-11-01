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



'''
- The following function works with default inputs but you can provide your own inputs as well as per requirement of symbol/time period
- if you want to provide your own OHLCV input, you can use this module to get signals
- Use the provided main function code to integrate it in your system the way it works the best. 

'''





def getTACombinedSignals(input_df, returnall=False):
    all_funcs = talib.get_functions()

    # Creating a pandas dataframe which will contain strategy signal in each column
    signals = pd.DataFrame(index=input_df.index)
    # Adding the most basic indicator which will be used in many other signals
    ma_signal = np.where(abstract.Function("MA")(input_df) < input_df.close, 1, np.where(abstract.Function("MA")(input_df) > input_df.close, -1, 0))
    signals["MA"] = ma_signal

    # RSI is needed for HT_TRENDMODE
    rsi_signal = np.where(abstract.Function("RSI")(input_df) < 30, 1, np.where(abstract.Function("RSI")(input_df) > 70, -1, 0))
    signals["RSI"] = rsi_signal

    all_signals = {}
    all_signals.update(_get_candle_pattern_signals(input_df, all_funcs))
    all_signals.update(_get_price_transform_signals(input_df, all_funcs))
    all_signals.update(_get_volume_based_signals(input_df))
    all_signals.update(_get_volatility_based_signals(input_df, ma_signal))
    all_signals.update(_get_statistical_signals(input_df, ma_signal))
    all_signals.update(_get_momentum_signals(input_df, ma_signal))
    all_signals.update(_get_overlap_studies_signals(input_df))
    all_signals.update(_get_math_operator_signals(input_df))
    all_signals.update(_get_hilbert_transform_signals(input_df, ma_signal, rsi_signal))

    signals = pd.concat([signals, pd.DataFrame(all_signals, index=input_df.index)], axis=1)

    if returnall:
        return signals
    else:
        return signals.iloc[-1]


def _get_candle_pattern_signals(input_df, all_funcs):
    # Candle Pattern Related Charting Indicators (1-61)
    new_signals = {}
    for func in all_funcs:
        if "CDL" in abstract.Function(func).info['name']:
            new_signals[abstract.Function(func).info['display_name']] = np.where(abstract.Function(func)(input_df) > 0, 1, np.where(abstract.Function(func)(input_df) < 0, -1, 0))
    return new_signals


def _get_price_transform_signals(input_df, all_funcs):
    # Price Transform Technical Indicators (62-66)
    new_signals = {}
    for func in all_funcs:
        if "PRICE" in abstract.Function(func).info['name']:
            new_signals[abstract.Function(func).info['display_name']] = np.where(abstract.Function(func)(input_df) > input_df['close'], 1, np.where(abstract.Function(func)(input_df) < input_df['close'], -1, 0))
    return new_signals


def _get_volume_based_signals(input_df):
    # Volume Based Indicator (67-69)
    new_signals = {}
    ### OBV: When OBV is above its 5 day mean, buy else sell
    new_signals["OBV"] = np.where(abstract.Function("OBV")(input_df) > abstract.Function("OBV")(input_df).rolling(5).mean().bfill(), 1, 0)
    ### Chaikin Oscillator: When signal crosses 0 from below, buy. Vice versa for sell
    new_signals["ADOSC"] = np.where(((abstract.Function("ADOSC")(input_df) >= 0) & (abstract.Function("ADOSC")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("ADOSC")(input_df) < 0) & (abstract.Function("ADOSC")(input_df).shift(1) >= 0)), -1, 0))
    ### Chaikin A/D Line: When signal is above its 5 day mean, buy else sell
    new_signals["AD"] = np.where(abstract.Function("AD")(input_df) > abstract.Function("AD")(input_df).rolling(5).mean().bfill(), 1, 0)
    return new_signals


def _get_volatility_based_signals(input_df, ma_signal):
    # Volatility Based Indicators (70)
    new_signals = {}
    ### Normalized Average True Range: Strong trend when value above 0.6. We do not consider it to be directional but supporting the directional indicators such as MACD
    ####Out of True Range, Average True Range and Normalized Average True Range: We only use the last one, rest two are inefficient versions of the same
    new_signals["NATR"] = np.where(abstract.Function("NATR")(input_df) > 0.6, 1, 0)
    new_signals["NATR"] = ma_signal * new_signals["NATR"]
    return new_signals


def _get_statistical_signals(input_df, ma_signal):
    # Statistical Indicators (71-74)
    new_signals = {}
    ### Beta: Buy when beta > 1.5 , sell when beta < - 1.5
    new_signals["BETA"] = np.where(abstract.Function("BETA")(input_df) > 1.5, 1, np.where(abstract.Function("BETA")(input_df) < -1.5, -1, 0))
    ### Correlation: Trending when Correlation > 0.85, else not trending
    new_signals["CORREL"] = np.where(abstract.Function("CORREL")(input_df) > 0.85, 1, 0)
    new_signals["CORREL"] = ma_signal * new_signals["CORREL"]
    ### Linear Regression: Buy/Sell when crossovers with Close price
    new_signals["LINEARREG"] = np.where(((input_df.close >= abstract.Function("LINEARREG")(input_df)) & (input_df.close.shift(1) < abstract.Function("LINEARREG")(input_df).shift(1))), 1, np.where(((input_df.close <= abstract.Function("LINEARREG")(input_df)) & (input_df.close.shift(1) > abstract.Function("LINEARREG")(input_df).shift(1))), -1, 0))
    ### TimeSeriesForecast: Buy/Sell when crossovers with Close price
    new_signals["TSF"] = np.where(((input_df.close >= abstract.Function("TSF")(input_df)) & (input_df.close.shift(1) < abstract.Function("TSF")(input_df).shift(1))), 1, np.where(((input_df.close <= abstract.Function("TSF")(input_df)) & (input_df.close.shift(1) > abstract.Function("TSF")(input_df).shift(1))), -1, 0))
    return new_signals


def _get_momentum_signals(input_df, ma_signal):
    # Momentum Indicators (75-95)
    new_signals = {}
    ### ADX: Trending when ADX > 25, else not trending
    new_signals["ADX"] = np.where(abstract.Function("ADX")(input_df) > 25, 1, 0)
    new_signals["ADX"] = ma_signal * new_signals["ADX"]
    ### ADXR: Buy when crossovers ADX < ADXR, vice versa
    new_signals["ADXR"] = np.where(((abstract.Function("ADXR")(input_df) >= abstract.Function("ADX")(input_df)) & (abstract.Function("ADXR")(input_df).shift(1) < abstract.Function("ADX")(input_df).shift(1))), 1, np.where(((abstract.Function("ADXR")(input_df) <= abstract.Function("ADX")(input_df)) & (abstract.Function("ADXR")(input_df).shift(1) > abstract.Function("ADX")(input_df).shift(1))), -1, 0))
    ### APO: Buy when crossovers APO < 0, vice versa
    new_signals["APO"] = np.where(((abstract.Function("APO")(input_df) >= 0) & (abstract.Function("APO")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("APO")(input_df) <= 0) & (abstract.Function("APO")(input_df).shift(1) > 0)), -1, 0))
    ### AROON OSC: Buy when ADX > 40 Sell when < -40
    new_signals["AROONOSC"] = np.where(abstract.Function("AROONOSC")(input_df) > 40, 1, np.where(abstract.Function("AROONOSC")(input_df) < -40, -1, 0))
    ### Balance of Power: Buy when crossovers BOP > 0 Sell when vice versa
    new_signals["BOP"] = np.where(((abstract.Function("BOP")(input_df) >= 0) & (abstract.Function("BOP")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("BOP")(input_df) <= 0) & (abstract.Function("BOP")(input_df).shift(1) > 0)), -1, 0))
    ### Commodity channel index: Buy when CCI > 100 Sell when CCI < 100
    new_signals["CCI"] = np.where(abstract.Function("CCI")(input_df) > 100, 1, np.where(abstract.Function("CCI")(input_df) < -100, -1, 0))
    ### Chande Momentum Oscillator: Buy when CMO < -50 Sell when CMO > 50
    new_signals["CMO"] = np.where(abstract.Function("CMO")(input_df) < -50, 1, np.where(abstract.Function("CMO")(input_df) > 50, -1, 0))
    ### Directional Movement Index: When DX is above its 5 period mean, buy else sell
    new_signals["DX"] = np.where(abstract.Function("DX")(input_df) > abstract.Function("DX")(input_df).rolling(5).mean().bfill(), 1, np.where(abstract.Function("DX")(input_df) < abstract.Function("DX")(input_df).rolling(5).mean().bfill(), -1, 0))
    ### MACD: Buy when crossovers MACDhist < 0 Sell when vice versa
    new_signals["MACD"] = np.where(((abstract.Function("MACD")(input_df).macdhist >= 0) & (abstract.Function("MACD")(input_df).macdhist.shift(1) < 0)), -1, np.where(((abstract.Function("MACD")(input_df).macdhist <= 0) & (abstract.Function("MACD")(input_df).macdhist.shift(1) > 0)), 1, 0))
    ### MACD EXT: Buy when crossovers MACDhist < 0 Sell when vice versa
    new_signals["MACDEXT"] = np.where(((abstract.Function("MACDEXT")(input_df).macdhist >= 0) & (abstract.Function("MACDEXT")(input_df).macdhist.shift(1) < 0)), -1, np.where(((abstract.Function("MACDEXT")(input_df).macdhist <= 0) & (abstract.Function("MACDEXT")(input_df).macdhist.shift(1) > 0)), 1, 0))
    ### MACD FIX: Buy when crossovers MACDhist < 0 Sell when vice versa
    new_signals["MACDFIX"] = np.where(((abstract.Function("MACDFIX")(input_df).macdhist >= 0) & (abstract.Function("MACDFIX")(input_df).macdhist.shift(1) < 0)), -1, np.where(((abstract.Function("MACDFIX")(input_df).macdhist <= 0) & (abstract.Function("MACDFIX")(input_df).macdhist.shift(1) > 0)), 1, 0))
    ### Money Flow index: Buy when MFI < 30 Sell when MFI > 80 (Overbought and oversold)
    new_signals["MFI"] = np.where(abstract.Function("MFI")(input_df) < 30, 1, np.where(abstract.Function("MFI")(input_df) > 80, -1, 0))
    ### Momentum Indicator: Buy when MOM > 100 Sell when MOM < -100
    new_signals["MOM"] = np.where(abstract.Function("MOM")(input_df) > 100, 1, np.where(abstract.Function("MOM")(input_df) < -100, -1, 0))
    ### Rate of Change Indicator: Buy when ROC > 0.25 Sell when ROC < -0.25
    new_signals["ROC"] = np.where(abstract.Function("ROC")(input_df) > 0.25, 1, np.where(abstract.Function("ROC")(input_df) < -0.25, -1, 0))
    ### Relative Strength Index: Buy when RSI < 30 Sell when RSI > 70 (Overbought and oversold)
    new_signals["RSI"] = np.where(abstract.Function("RSI")(input_df) < 30, 1, np.where(abstract.Function("RSI")(input_df) > 70, -1, 0))
    ### Stochastic: Buy when STOCH.slowk < 20 Sell when STOCH.slowk > 80 (Overbought and oversold)
    new_signals["STOCH"] = np.where(abstract.Function("STOCH")(input_df).slowk < 20, 1, np.where(abstract.Function("STOCH")(input_df).slowk > 80, -1, 0))
    ### Stochastic Fast: Buy when STOCH.slowk < 20 Sell when STOCH.slowk > 80 (Overbought and oversold)
    new_signals["STOCHF"] = np.where(abstract.Function("STOCHF")(input_df).fastk < 20, 1, np.where(abstract.Function("STOCHF")(input_df).fastk > 80, -1, 0))
    ### Stochastic RSI: Buy when STOCH.fastk < 20 Sell when STOCH.fastk > 80 (Overbought and oversold)
    new_signals["STOCHRSI"] = np.where(abstract.Function("STOCHRSI")(input_df).fastk < 20, 1, np.where(abstract.Function("STOCHRSI")(input_df).fastk > 80, -1, 0))
    ### Triple Exp Avg: Buy when TRIX crossovers < 0 Sell when vice versa
    new_signals["TRIX"] = np.where(((abstract.Function("TRIX")(input_df) >= 0) & (abstract.Function("TRIX")(input_df).shift(1) < 0)), -1, np.where(((abstract.Function("TRIX")(input_df) <= 0) & (abstract.Function("TRIX")(input_df).shift(1) > 0)), 1, 0))
    ### Ultimate Oscillator: Buy when ULTOSC < 30 Sell when ULTOSC > 70 (Overbought and oversold)
    new_signals["ULTOSC"] = np.where(abstract.Function("ULTOSC")(input_df) < 30, 1, np.where(abstract.Function("ULTOSC")(input_df) > 70, -1, 0))
    ### Williams %R: Buy when WILLR < 30 Sell when WILLR > 70 (Overbought and oversold)
    new_signals["WILLR"] = np.where(abstract.Function("WILLR")(input_df) < -80, 1, np.where(abstract.Function("WILLR")(input_df) > -20, -1, 0))
    return new_signals


def _get_overlap_studies_signals(input_df):
    # Overlap Studies (96-108)
    new_signals = {}
    ### Exponential Moving Averages: Buy when price > EMA else sell
    new_signals["EMA"] = np.where(abstract.Function("EMA")(input_df) < input_df.close, 1, np.where(abstract.Function("EMA")(input_df) > input_df.close, -1, 0))
    ### Double Exponential Moving Averages: Buy when price > DEMA else sell
    new_signals["DEMA"] = np.where(abstract.Function("DEMA")(input_df) < input_df.close, 1, np.where(abstract.Function("DEMA")(input_df) > input_df.close, -1, 0))
    ### Kaufman Adaptive Moving Average: Buy when price > KAMA else sell
    new_signals["KAMA"] = np.where(abstract.Function("KAMA")(input_df) < input_df.close, 1, np.where(abstract.Function("KAMA")(input_df) > input_df.close, -1, 0))
    ### MESA Adaptive Moving Average: Buy when fama > mama else sell
    new_signals["MAMA"] = np.where(abstract.Function("MAMA")(input_df).mama > abstract.Function("MAMA")(input_df).fama, 1, np.where(abstract.Function("MAMA")(input_df).mama < abstract.Function("MAMA")(input_df).fama, -1, 0))
    ### MIDPOINT: When MP is above its 5 period mean, buy else sell
    new_signals["MIDPOINT"] = np.where(abstract.Function("MIDPOINT")(input_df) > abstract.Function("MIDPOINT")(input_df).rolling(5).mean().bfill(), 1, np.where(abstract.Function("MIDPOINT")(input_df) < abstract.Function("MIDPOINT")(input_df).rolling(5).mean().bfill(), -1, 0))
    ### MIDPRICE: When MPX is above its 5 period mean, buy else sell
    new_signals["MIDPRICE"] = np.where(abstract.Function("MIDPRICE")(input_df) > abstract.Function("MIDPRICE")(input_df).rolling(5).mean().bfill(), 1, np.where(abstract.Function("MIDPRICE")(input_df) < abstract.Function("MIDPRICE")(input_df).rolling(5).mean().bfill(), -1, 0))
    ### Parabolic SAR: Buy/Sell when crossovers with Close price
    new_signals["SAR"] = -1 * np.where(((input_df.close >= abstract.Function("SAR")(input_df)) & (input_df.close.shift(1) < abstract.Function("SAR")(input_df).shift(1))), 1, np.where(((input_df.close <= abstract.Function("SAR")(input_df)) & (input_df.close.shift(1) > abstract.Function("SAR")(input_df).shift(1))), -1, 0))
    ### Parabolic SAR Extended: Buy when >0 else sell
    new_signals["SAREXT"] = -1 * np.where(((abstract.Function("SAREXT")(input_df) >= 0) & (abstract.Function("SAREXT")(input_df).shift(1) < 0)), 1, np.where(((abstract.Function("SAREXT")(input_df) <= 0) & (abstract.Function("SAREXT")(input_df).shift(1) > 0)), -1, 0))
    ### Triple Exponential Moving Average: Buy when price > T3 else sell
    new_signals["T3"] = np.where(abstract.Function("T3")(input_df) < input_df.close, 1, np.where(abstract.Function("T3")(input_df) > input_df.close, -1, 0))
    ### Triple Exponential Moving Average: Buy when price > TEMA else sell
    new_signals["TEMA"] = np.where(abstract.Function("TEMA")(input_df) < input_df.close, 1, np.where(abstract.Function("TEMA")(input_df) > input_df.close, -1, 0))
    ### Triangular Moving Average: Buy when price > TRIMA else sell
    new_signals["TRIMA"] = np.where(abstract.Function("TRIMA")(input_df) < input_df.close, 1, np.where(abstract.Function("TRIMA")(input_df) > input_df.close, -1, 0))
    ### Weighted Moving Average: Buy when price > WMA else sell
    new_signals["WMA"] = np.where(abstract.Function("WMA")(input_df) < input_df.close, 1, np.where(abstract.Function("WMA")(input_df) > input_df.close, -1, 0))
    return new_signals


def _get_math_operator_signals(input_df):
    # Math Operators (109)
    new_signals = {}
    ### Max and Min: Buy when price = Min, sell when price = Max
    new_signals["MINMAX"] = np.where(abstract.Function("MIN")(input_df) == input_df.close, 1, np.where(abstract.Function("MAX")(input_df) == input_df.close, -1, 0))
    return new_signals


def _get_hilbert_transform_signals(input_df, ma_signal, rsi_signal):
    # Hilbert Transform Cycle Indicators (110-111)
    new_signals = {}
    ## Dominant Cycle Period: Trend confirmed when HT_DCPERIOD > 25
    new_signals["HT_DCPERIOD"] = np.where(abstract.Function("HT_DCPERIOD")(input_df) > 25, 1, 0)
    new_signals["HT_DCPERIOD"] = ma_signal * new_signals["HT_DCPERIOD"]
    ## Trend vs Cycle Mode: Use trending strategy (MA) when HT_TRENDMODE = 1 else Use cyclic strategy (RSI)
    new_signals["HT_TRENDMODE"] = np.where(abstract.Function("HT_TRENDMODE")(input_df) == 1, ma_signal, rsi_signal)
    return new_signals




if __name__ == '__main__':
    # This is a placeholder for testing the LiveSignalGenerator function.
    # You can create a sample dataframe with columns: open, high, low, close, volume
    # and pass it to the LiveSignalGenerator function.
    pass