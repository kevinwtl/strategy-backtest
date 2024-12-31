from abc import ABC, abstractmethod
from datetime import datetime
from functools import cache, cached_property
from typing import Literal

import numpy as np
import pandas as pd
import pandas_ta as ta
import statsmodels.api as sm
import yfinance as yf
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import Range1d
from bokeh.resources import INLINE
from bokeh.plotting import figure, show
from pytimeparse.timeparse import timeparse
from sorcery import dict_of
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

output_notebook(INLINE)


def zscore(x: pd.DataFrame, lookback=None):
    if lookback is None:
        return ((x - x.mean()) / x.std()).fillna(0)
    else:
        return ((x - x.rolling(lookback).mean()) / x.rolling(lookback).std()).fillna(np.nan)

def slope(x):
    return np.polyfit(np.arange(len(x)), x, 1)[0]

class BacktestTemplate(ABC):
    def __init__(
        self,
        symbol,
        backtest_model: Literal["zscore", "percentile", "atr", "time", "signal"],
        transaction_fee=0.0005,
        execution_delay=0,
        percentage_allocation=1,
        sharpe_freq="1d",
    ) -> None:
        self.transaction_fee = transaction_fee
        self.symbol = symbol
        self.execution_delay = execution_delay
        self.backtest_model = backtest_model
        self.percentage_allocation = percentage_allocation
        self.sharpe_freq = sharpe_freq

        self.param = None

    def set_param(
        self,
        time_stop: int = None,
        trailing_percentage_stop: float = None,
        percentage_stop: float = None,
        percentage_target: float = None,
        price_stop: str = None,
        price_target: str = None,
        atr_stop: float = None,
        atr_target: float = None,
        atr_period: int = None,
        **kwargs,
    ):

        self.param = dict(
            time_stop=time_stop,
            trailing_percentage_stop=trailing_percentage_stop,
            percentage_stop=percentage_stop,
            percentage_target=percentage_target,
            price_stop=price_stop,
            price_target=price_target,
            atr_stop=atr_stop,
            atr_target=atr_target,
            atr_period=atr_period,
        )
        self.param.update(kwargs)

    def run_backtest(self) -> pd.DataFrame:
        if self.backtest_model == "zscore":
            df = self._zscore_model_backtest()
        elif self.backtest_model == "signal":
            df = self._signal_model_backtest()

        df["holding period"] = df["exit_time"] - df["entry_time"]
        return df

    # TODO: rename to indicator model
    def _zscore_model_backtest(self) -> pd.DataFrame:
        if self.param["enter_Z"] is None:
            raise ValueError("'enter_Z' not set")

        if self.param["exit_Z"] is None:
            self.param["exit_Z"] = self.param["enter_Z"]

        df = self._process_signal().copy()

        if self.param["atr_stop"] or self.param["atr_target"]:
            df["atr"] = df.ta.atr(self.param["atr_period"])

        qty = 0
        px_entry = None
        px_exit = None
        entry_time = None
        exit_time = None
        balance = 100
        counter = 0
        trailing_peak = None
        stop_price = None
        target_price = None
        trading_log = []

        for index, row in df.iterrows():
            # Create new position
            if qty == 0 and abs(row["z_scores"]) >= self.param["enter_Z"] and balance > 0:
                entry_time = index
                px_entry = row["close"]

                # Check for stop loss price
                if self.param["price_stop"]:
                    stop_price = row[self.param["price_stop"]]
                elif self.param["atr_stop"]:
                    stop_price = px_entry - row["atr"] * self.param["atr_stop"] * np.sign(row["z_scores"])
                elif self.param["percentage_stop"]:
                    stop_price = px_entry * (1 - self.param["percentage_stop"] * np.sign(row["z_scores"]))
                elif self.param["trailing_percentage_stop"]:
                    stop_price = px_entry * (1 - self.param["trailing_percentage_stop"] * np.sign(row["z_scores"]))

                # Check for target price
                if self.param["price_target"]:
                    target_price = row[self.param["price_target"]]
                elif self.param["atr_target"]:
                    target_price = px_entry + row["atr"] * self.param["atr_target"] * np.sign(row["z_scores"])
                elif self.param["percentage_target"]:
                    target_price = px_entry * (1 + self.param["percentage_target"] * np.sign(row["z_scores"]))

                # Calculate position size
                qty = np.sign(row["z_scores"]) * balance * self.percentage_allocation / px_entry
                fees = abs(qty) * self.transaction_fee * px_entry  # entry fees paid in BTC
                qty = qty * (1 - self.transaction_fee)  # less BTC exposure after fees

                should_exit = False

            # Check for exit
            elif qty != 0:
                should_exit = False

                # Check z-score exit condition
                if (qty > 0 and row["z_scores"] < self.param["exit_Z"]) or (qty < 0 and row["z_scores"] > -self.param["exit_Z"]):
                    should_exit = True

                # Check stop loss conditions
                if stop_price:
                    if (qty > 0 and row["low"] < stop_price) or (qty < 0 and row["high"] > stop_price):
                        should_exit = True
                        px_exit = stop_price

                if self.param["trailing_percentage_stop"]:
                    if trailing_peak is None:
                        trailing_peak = row["high"] if qty > 0 else row["low"]
                    else:
                        trailing_peak = max(trailing_peak, row["high"]) if qty > 0 else min(trailing_peak, row["low"])
                        stop = trailing_peak * (1 - self.param["trailing_percentage_stop"]) if qty > 0 else trailing_peak * (1 + self.param["trailing_percentage_stop"])
                        if (row["low"] < stop and qty > 0) or (row["high"] > stop and qty < 0):
                            should_exit = True
                            px_exit = stop

                # Check take profit conditions
                if target_price:
                    if (qty > 0 and row["high"] > target_price) or (qty < 0 and row["low"] < target_price):
                        should_exit = True
                        px_exit = target_price

                # Check time stop conditions
                if self.param["time_stop"]:
                    if counter >= self.param["time_stop"]:
                        should_exit = True

                if should_exit:
                    px_exit = row["close"] if px_exit is None else px_exit
                    exit_time = index
                    fees += abs(qty) * px_exit * self.transaction_fee  # exit fees paid in USDT
                    net_profit = qty * (px_exit - px_entry) - fees
                    pct_ret = net_profit / balance
                    post_trade_balance = balance + net_profit

                    trading_log.append(dict_of(entry_time, exit_time, qty, px_entry, px_exit, fees, net_profit, pct_ret, balance, post_trade_balance))

                    # Reset Holdings
                    qty = 0
                    fees = 0
                    counter = 0
                    balance = post_trade_balance
                    px_exit = None
                    stop_price = None
                    target_price = None
                    trailing_peak = None

            if qty != 0:
                counter += 1

        results = pd.DataFrame(trading_log)
        return results

    def _signal_model_backtest(self) -> pd.DataFrame:
        df = self._process_signal().copy()
        if self.param["atr_stop"] or self.param["atr_target"]:
            df["atr"] = df.ta.atr(self.param["atr_period"])

        qty = 0
        px_entry = None
        px_exit = None
        entry_time = None
        exit_time = None
        balance = 100
        counter = 0
        trailing_peak = None
        stop_price = None
        target_price = None
        trading_log = []

        for index, row in df.iterrows():
            # Create new position
            if qty == 0 and row["signal"] != 0 and balance > 0:
                entry_time = index
                px_entry = row["close"]

                # Check for stop loss price
                if self.param["price_stop"]:
                    stop_price = row[self.param["price_stop"]]
                elif self.param["atr_stop"]:
                    stop_price = px_entry - row["atr"] * self.param["atr_stop"] * np.sign(row["signal"])
                elif self.param["percentage_stop"]:
                    stop_price = px_entry * (1 - self.param["percentage_stop"] * np.sign(row["signal"]))
                elif self.param["trailing_percentage_stop"]:
                    stop_price = px_entry * (1 - self.param["trailing_percentage_stop"] * np.sign(row["signal"]))

                # Check for target price
                if self.param["price_target"]:
                    target_price = row[self.param["price_target"]]
                elif self.param["atr_target"]:
                    target_price = px_entry + row["atr"] * self.param["atr_target"] * np.sign(row["signal"])
                elif self.param["percentage_target"]:
                    target_price = px_entry * (1 + self.param["percentage_target"] * np.sign(row["signal"]))

                # Calculate position size
                qty = np.sign(row["signal"]) * balance * self.percentage_allocation / px_entry  # / sl_distance
                fees = abs(qty) * self.transaction_fee * px_entry  # entry fees paid in BTC
                qty = qty * (1 - self.transaction_fee)  # less BTC exposure after fees

                should_exit = False

            # Check for exit
            elif qty != 0:
                # Check stop loss conditions
                if stop_price:
                    if (qty > 0 and row["low"] < stop_price) or (qty < 0 and row["high"] > stop_price):
                        should_exit = True
                        px_exit = stop_price

                if self.param["trailing_percentage_stop"]:
                    if trailing_peak is None:
                        trailing_peak = row["high"] if qty > 0 else row["low"]
                    else:
                        trailing_peak = max(trailing_peak, row["high"]) if qty > 0 else min(trailing_peak, row["low"])
                        stop = trailing_peak * (1 - self.param["trailing_percentage_stop"]) if qty > 0 else trailing_peak * (1 + self.param["trailing_percentage_stop"])
                        if (row["low"] < stop and qty > 0) or (row["high"] > stop and qty < 0):
                            should_exit = True
                            px_exit = stop

                # Check take profit conditions
                if target_price:
                    if (qty > 0 and row["high"] > target_price) or (qty < 0 and row["low"] < target_price):
                        should_exit = True
                        px_exit = target_price

                # Check time stop conditions
                if self.param["time_stop"]:
                    if counter >= self.param["time_stop"]:
                        should_exit = True

                if should_exit:
                    px_exit = row["close"] if px_exit is None else px_exit
                    exit_time = index
                    fees += abs(qty) * px_exit * self.transaction_fee  # exit fees paid in USDT
                    net_profit = qty * (px_exit - px_entry) - fees
                    pct_ret = net_profit / balance
                    post_trade_balance = balance + net_profit

                    trading_log.append(dict_of(entry_time, exit_time, qty, px_entry, px_exit, fees, net_profit, pct_ret, balance, post_trade_balance))

                    # Reset Holdings
                    qty = 0
                    fees = 0
                    counter = 0
                    balance = post_trade_balance
                    px_exit = None
                    stop_price = None
                    target_price = None
                    trailing_peak = None

            if qty != 0:
                counter += 1

        results = pd.DataFrame(trading_log)
        return results

    @cached_property
    def _1m_candle(self):
        try:
            klines = pd.read_pickle(f"trading/crypto/data/klines/spot/{self.symbol}_klines.pkl")
            klines = (
                klines.resample("1min")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "turnover": "sum", "trades": "sum", "taker_buy_volume": "sum", "taker_buy_turnover": "sum"})
                .ffill()
            )  # Missing spot data in 2023 March 24
            klines = klines.shift(-self.execution_delay)  # Execution Delay
            return klines
        except FileNotFoundError:
            return None

    @cached_property
    def _1m_futures_candle(self):
        try:
            klines = pd.read_pickle(f"trading/crypto/data/klines/futures/{self.symbol}_klines.pkl")
            klines = (
                klines.resample("1min")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "turnover": "sum", "trades": "sum", "taker_buy_volume": "sum", "taker_buy_turnover": "sum"})
                .ffill()
            )  # Missing spot data in 2023 March 24
            klines = klines.shift(-self.execution_delay)  # Execution Delay
            return klines
        except FileNotFoundError:
            return None

    @abstractmethod
    def _process_signal(self) -> pd.DataFrame:
        """Return a dataframe with the indicator column."""

    def _unrealized_equity_curve(
        self,
        price_candle: Literal["_1m_candle", "_1m_futures_candle"] = "_1m_futures_candle",
    ):
        result_df = self.run_backtest().copy()
        shift_minutes = int(timeparse("15m") / 60)
        shifted_candle = getattr(self, price_candle).copy()["close"]  # .shift(-shift_minutes)

        equity_curve = pd.Series(index=shifted_candle.index, dtype=float)

        balance = 100
        for _, row in result_df.iterrows():
            entry_time, exit_time = row["entry_time"], row["exit_time"]
            trade_period = shifted_candle.loc[entry_time:exit_time].index

            entry_fee = row["px_entry"] * abs(row["qty"]) * self.transaction_fee
            balance -= entry_fee

            trade_prices = shifted_candle.loc[trade_period]
            pnl = (trade_prices - row["px_entry"]) * row["qty"]
            equity_curve.loc[trade_period] = pnl + balance

            exit_fee = row["px_exit"] * abs(row["qty"]) * self.transaction_fee
            balance = row["post_trade_balance"]  # pnl.iloc[-1] + balance - exit_fee
            equity_curve.loc[exit_time] = balance

        equity_curve = equity_curve.ffill().fillna(100)
        equity_curve = equity_curve.loc[equity_curve.ne(100).idxmax() :]  # remove all trailing 100

        return equity_curve.sort_index()

    def plot_returns_distribution(self):
        result_df = self.run_backtest()
        result_df["pct_ret"].hist(bins=100)

    def sharpe_ratio(self):
        rets = self._unrealized_equity_curve().resample(self.sharpe_freq).last().ffill().pct_change()
        n_per_peroid = {"1d": 365, "1h": 8760}[self.sharpe_freq]
        avg_ret = rets.mean() * n_per_peroid
        std = rets.std() * n_per_peroid**0.5
        return avg_ret / std

    def mdd(self):
        equity_curve = self._unrealized_equity_curve()
        cum_max = equity_curve.cummax()
        roll_drawdown = equity_curve / cum_max - 1
        mdd = -roll_drawdown.min()
        return mdd

    def calmar_ratio(self):
        rets = self._unrealized_equity_curve().resample("1d").last().ffill().pct_change()
        avg_ret = rets.mean() * 365
        return avg_ret / self.mdd()

    def annualized_return(self):
        equity_curve = self._unrealized_equity_curve().resample("1d").last().ffill()
        n_years = (equity_curve.index.max() - equity_curve.index.min()).days / 365
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1

    def avg_trades_per_week(self):
        result_df = self.run_backtest().copy()
        if len(result_df) == 0:
            raise ValueError("No trades executed")
        days = (result_df["entry_time"].max() - result_df["entry_time"].min()).days
        if days == 0: # Only 1 trade
            return 0
        return len(result_df) / (days / 7)

    def backtest_summary(self):
        result_df = self.run_backtest().copy()

        metrics = {
            "Final Balance": f"{result_df['balance'].iloc[-1]:.2f}",
            "Win Rate": f"{len(result_df[result_df['net_profit'] > 0]) / len(result_df):.2%}",
            "Number of trades": str(len(result_df)),
            "Sharpe": f"{self.sharpe_ratio():.2f}",
            "Calmar": f"{self.calmar_ratio():.2f}",
            "Max Drawdown": f"{self.mdd():.2%}",
            "Annualized Return": f"{self.annualized_return():.2%}",
            "Average Trades per Week": f"{self.avg_trades_per_week():.1f}",
            "Median Holding Period": str(result_df["holding period"].median()),
        }

        return "\n".join(f"{k}: {v}" for k, v in metrics.items())

    def plot_results(self):
        df = self._process_signal().copy()
        result_df = self.run_backtest().copy()

        print(self.backtest_summary())

        x_range = Range1d(start=df.index[0], end=df.index[-1])

        # Create 4 figures sharing x axis
        p1 = figure(width=800, height=300, x_axis_type="datetime", title=f"{self.symbol} Close Price", x_range=x_range)
        p1.line(df.index, df["close"], line_width=2)
        for i in range(len(result_df)):
            p1.varea(x=[result_df["entry_time"][i], result_df["exit_time"][i]], y1=df["close"].min(), y2=df["close"].max(), fill_color="green" if result_df["qty"][i] > 0 else "red", fill_alpha=0.1)

        p2 = figure(width=800, height=300, x_axis_type="datetime", title="Indicator", x_range=p1.x_range)

        if self.backtest_model == "zscore":
            p2.line(df.index, df["z_scores"], line_width=2)
            for i in range(len(result_df)):
                p2.varea(
                    x=[result_df["entry_time"][i], result_df["exit_time"][i]],
                    y1=df["z_scores"].min(),
                    y2=df["z_scores"].max(),
                    fill_color="green" if result_df["qty"][i] > 0 else "red",
                    fill_alpha=0.1,
                )

            # Show Z-score thresholds
            p2.line(df.index, [self.param["enter_Z"]] * len(df), line_dash="dashed", color="red")
            p2.line(df.index, [-self.param["enter_Z"]] * len(df), line_dash="dashed", color="red")
            p2.line(df.index, [self.param["exit_Z"]] * len(df), line_dash="dashed", color="grey")
            p2.line(df.index, [-self.param["exit_Z"]] * len(df), line_dash="dashed", color="grey")

        elif self.backtest_model == "signal":
            p2.line(df.index, df["signal"], line_width=2)

        p3 = figure(width=800, height=300, x_axis_type="datetime", title="Equity Curve (Unrealized)", x_range=p1.x_range)
        unrealized = self._unrealized_equity_curve()
        p3.line(unrealized.index, unrealized.values, line_width=2)

        grid = gridplot([[p1], [p2], [p3]])
        show(grid)

    def optimize(self, param_grid: dict, workers: int = 8):
        """
        Optimize strategy parameters by testing different combinations.

        Args:
            param_grid: Dictionary of parameter combinations to test

        Returns:
            List of dictionaries containing results for each parameter combination
        """


        # Convert param_grid to list of parameter combinations if needed
        if isinstance(param_grid, dict):
            import itertools

            keys = param_grid.keys()
            values = param_grid.values()
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        def get_metrics(params):
            try:
                # Run backtest with current parameters
                self.set_param(**params)
                result = self.run_backtest()

                # Calculate performance metrics
                metrics = {
                    "sharpe": self.sharpe_ratio(),
                    "mdd": self.mdd(),
                    "calmar": self.calmar_ratio(),
                    "final_bal": result["balance"].iloc[-1],
                    "n_trades": len(result),
                    "hit_rate": len(result[result["net_profit"] > 0]) / len(result),
                    "long_hit_rate": len(result.query("qty > 0 and net_profit > 0")) / max(1, len(result.query("qty > 0"))),
                    "short_hit_rate": len(result.query("qty < 0 and net_profit > 0")) / max(1, len(result.query("qty < 0"))),
                    "avg_hold_period": result["holding period"].mean(),
                }

                return {**params, **metrics}

            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                return None

        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(get_metrics, params) for params in param_combinations]
            for future in tqdm(as_completed(futures), total=len(futures)):
                if (result := future.result()) is not None:
                    results.append(result)

        return pd.DataFrame(results)
