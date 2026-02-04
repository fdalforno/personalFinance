import numpy as np
import pandas as pd

def calculate_moving_average_signals(prices, short_window=20, long_window=250):
    """
    Calcola segnali di trading basati su medie mobili.
    Segnale di acquisto quando MA corta > MA lunga
    Segnale di vendita quando MA corta < MA lunga
    
    :param prices: array di prezzi (1D)
    :param short_window: finestra per MA corta
    :param long_window: finestra per MA lunga
    :return: array di segnali (1=long, 0=neutral, -1=short)
    """
    signals = np.zeros(len(prices))
    
    # Calcola medie mobili
    ma_short = pd.Series(prices).rolling(window=short_window).mean().values
    ma_long = pd.Series(prices).rolling(window=long_window).mean().values
    
    # Genera segnali
    signals[ma_short > ma_long] = 1  # Long
    signals[ma_short < ma_long] = -1  # Short (o fuori dal mercato)
    
    return signals


def calculate_rsi_signals(prices, period=14, oversold=30, overbought=70):
    """
    Calcola segnali di trading basati su RSI (Relative Strength Index).
    Segnale di acquisto quando RSI < oversold
    Segnale di vendita quando RSI > overbought
    
    :param prices: array di prezzi (1D)
    :param period: periodo per il calcolo dell'RSI
    :param oversold: soglia di ipervenduto
    :param overbought: soglia di ipercomprato
    :return: array di segnali (1=long, 0=neutral, -1=short)
    """
    prices_series = pd.Series(prices)
    
    # Calcola variazioni di prezzo
    delta = prices_series.diff()
    
    # Separa guadagni e perdite
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calcola medie
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    # Calcola RS e RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Genera segnali
    signals = np.zeros(len(prices))
    signals[rsi < oversold] = 1  # Long (ipervenduto)
    signals[rsi > overbought] = -1  # Short o esci (ipercomprato)
    
    return signals


def calculate_macd_signals(prices, fast=12, slow=26, signal=9):
    """
    Calcola segnali di trading basati su MACD (Moving Average Convergence Divergence).
    Segnale di acquisto quando MACD > Signal line
    Segnale di vendita quando MACD < Signal line
    
    :param prices: array di prezzi (1D)
    :param fast: periodo EMA veloce
    :param slow: periodo EMA lenta
    :param signal: periodo signal line
    :return: array di segnali (1=long, 0=neutral, -1=short)
    """
    prices_series = pd.Series(prices)
    
    # Calcola EMA
    ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
    ema_slow = prices_series.ewm(span=slow, adjust=False).mean()
    
    # Calcola MACD e signal line
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # Genera segnali
    signals = np.zeros(len(prices))
    signals[macd > signal_line] = 1  # Long
    signals[macd < signal_line] = -1  # Short o fuori
    
    return signals

def calculate_buy_and_hold_signals(prices):
    """
    Genera segnali di trading per la strategia Buy and Hold.
    Sempre long.
    
    :param prices: array di prezzi (1D)
    :return: array di segnali (1=long)
    """
    signals = np.ones(len(prices))  # Sempre long
    return signals


def backtest_vectorized(prices, signals, initial_capital=10000,
                       commission_rate=0.001, slippage_rate=0.0005,
                       capital_gains_tax=0.26, periods_per_year=252):
    """
    Backtesting completamente vettorializzato usando pandas (molto più veloce).
    
    :param prices: array di prezzi (1D) o pandas Series
    :param signals: array di segnali (1=long, 0/other=out) o pandas Series
    :param initial_capital: capitale iniziale
    :param commission_rate: commissione come frazione (es. 0.001 = 0.1%)
    :param slippage_rate: slippage come frazione (es. 0.0005 = 0.05%)
    :param capital_gains_tax: tassa sui capital gain (es. 0.26 = 26%)
    :param periods_per_year: periodi per anno (default 252 giorni trading)
    :return: dictionary con metriche di performance
    """
    # Converti in pandas Series se necessario
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    if not isinstance(signals, pd.Series):
        signals = pd.Series(signals, index=prices.index)
    
    prices = prices.astype(float)
    signals = signals.reindex(prices.index).fillna(0)
    
    # Normalizza segnali: 1 = long, 0 = out
    pos = (signals == 1).astype(float)
    
    # No lookahead: usa posizione del giorno precedente
    pos = pos.shift(1).fillna(0)
    
    # Calcola rendimenti del prezzo
    ret = prices.pct_change().fillna(0.0)
    
    # --- Calcola costi di trading ---
    # Turnover: quanto cambia la posizione
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    
    # Costi di transazione (commissioni + slippage)
    trading_cost = turnover * (commission_rate + slippage_rate)
    
    # Rendimento lordo della strategia
    strat_ret_gross = pos * ret - trading_cost
    
    # Equity curve lorda (senza tasse)
    equity_gross = (1 + strat_ret_gross).cumprod() * initial_capital
    
    # --- Calcola tasse sui capital gain ---
    # Identifica entrate e uscite
    pos_change = pos.diff().fillna(0)
    entry = (pos_change > 0)  # Entrata in posizione
    exit_ = (pos_change < 0)  # Uscita da posizione
    
    # Prezzo di entrata e uscita
    entry_prices = prices.where(entry)
    exit_prices = prices.where(exit_)
    
    # Forward fill del prezzo di entrata per calcolare il profitto
    entry_forward = entry_prices.ffill()
    
    # Profitto realizzato (solo alle uscite)
    profit = (exit_prices - entry_forward).where(exit_).fillna(0)
    
    # Tasse solo sui profitti positivi
    tax_amount = profit.where(profit > 0, 0) * capital_gains_tax
    
    # Impatto delle tasse sul rendimento
    # Le tasse riducono l'equity quando vendiamo in profitto
    tax_impact = (tax_amount / equity_gross.shift(1)).fillna(0)
    
    # Rendimento netto (dopo tasse)
    strat_ret_net = strat_ret_gross - tax_impact
    
    # Equity curve finale
    equity = (1 + strat_ret_net).cumprod() * initial_capital
    
    # --- Calcola metriche ---
    final_value = equity.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    
    # Sharpe ratio
    returns = strat_ret_net
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max - 1.0) * 100
    max_drawdown = drawdown.min()
    
    # Conta trade e calcola statistiche
    num_trades = int(turnover.sum() / 2)  # Diviso 2 perché entry+exit = 2 turnovers
    
    # Calcola win rate sui trade chiusi
    exits_mask = exit_.values
    profits = profit.values[exits_mask]
    winning_trades = np.sum(profits > 0)
    losing_trades = np.sum(profits < 0)
    trade_win_rate = (winning_trades / (winning_trades + losing_trades) * 100) if (winning_trades + losing_trades) > 0 else 0
    
    # Costi totali
    total_commissions = (turnover * commission_rate * equity_gross.shift(1).fillna(initial_capital)).sum()
    total_slippage = (turnover * slippage_rate * equity_gross.shift(1).fillna(initial_capital)).sum()
    total_taxes = tax_amount.sum()
    total_costs = total_commissions + total_slippage + total_taxes
    costs_pct = (total_costs / initial_capital) * 100
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'portfolio_values': equity.values,
        'num_trades': num_trades,
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': trade_win_rate,
        'total_commissions': total_commissions,
        'total_slippage_cost': total_slippage,
        'total_taxes': total_taxes,
        'total_costs': total_costs,
        'costs_pct': costs_pct
    }





def compare_strategies_with_costs(price_paths, initial_capital=10000,
                                  commission_rate=0.001, slippage_rate=0.0005,
                                  capital_gains_tax=0.26):
    """
    Confronta multiple strategie con costi realistici su tutte le simulazioni.
    Usa la versione vettorializzata del backtest per performance ottimali.
    
    :param price_paths: matrice dei prezzi (num_simulations, trading_days + 1)
    :param initial_capital: capitale iniziale
    :param commission_rate: commissione come frazione (es. 0.001 = 0.1%)
    :param slippage_rate: slippage come frazione (es. 0.0005 = 0.05%)
    :param capital_gains_tax: tassa sui capital gain (es. 0.26 = 26%)
    :return: DataFrame con statistiche comparative
    """
    strategies = {
        'Buy & Hold': [],
        'Moving Average (20/250)': [],
        'RSI (14)': [],
        'MACD (12/26/9)': []
    }
    
    num_simulations = price_paths.shape[0]
    
    print(f"Processing {num_simulations} simulations...")
    
    for i in range(num_simulations):
        prices = price_paths[i]
        
        try:
            # Buy and Hold
            bh_signals = calculate_buy_and_hold_signals(prices)
            bh_result = backtest_vectorized(prices, bh_signals, initial_capital,
                                           commission_rate, slippage_rate, capital_gains_tax)
            strategies['Buy & Hold'].append(bh_result)
            
            # Moving Average
            ma_signals = calculate_moving_average_signals(prices)
            ma_result = backtest_vectorized(prices, ma_signals, initial_capital,
                                           commission_rate, slippage_rate, capital_gains_tax)
            strategies['Moving Average (20/250)'].append(ma_result)
            
            # RSI
            rsi_signals = calculate_rsi_signals(prices)
            rsi_result = backtest_vectorized(prices, rsi_signals, initial_capital,
                                            commission_rate, slippage_rate, capital_gains_tax)
            strategies['RSI (14)'].append(rsi_result)
            
            # MACD
            macd_signals = calculate_macd_signals(prices)
            macd_result = backtest_vectorized(prices, macd_signals, initial_capital,
                                             commission_rate, slippage_rate, capital_gains_tax)
            strategies['MACD (12/26/9)'].append(macd_result)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_simulations} simulations")
                
        except Exception as e:
            print(f"Error in simulation {i}: {str(e)}")
            continue
    
    # Verifica che ci siano risultati
    if not strategies['Buy & Hold']:
        print("WARNING: No results collected!")
        return pd.DataFrame()
    
    print(f"Collected {len(strategies['Buy & Hold'])} results per strategy")
    
    # Calcola statistiche aggregate
    summary = []
    for strategy_name, results in strategies.items():
        if not results:
            continue
            
        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        costs = [r['costs_pct'] for r in results]
        trades = [r['num_trades'] for r in results]
        win_rates = [r['win_rate'] for r in results]
        
        summary.append({
            'Strategy': strategy_name,
            'Avg Return (%)': np.mean(returns),
            'Std Return (%)': np.std(returns),
            'Avg Sharpe Ratio': np.mean(sharpes),
            'Avg Max Drawdown (%)': np.mean(drawdowns),
            'Win Rate (%)': sum(1 for r in returns if r > 0) / len(returns) * 100,
            'Avg Trades': np.mean(trades),
            'Avg Trade Win Rate (%)': np.mean(win_rates),
            'Avg Total Costs (%)': np.mean(costs)
        })
    
    return pd.DataFrame(summary)

