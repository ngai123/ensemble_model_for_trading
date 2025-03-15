import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import copy
import backtrader as bt
import cvxpy as cp

# Machine learning imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Set display options and random seeds for reproducibility
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('ggplot')  # Using a valid matplotlib style
np.random.seed(42)
random.seed(42)


def load_stock_data(ticker, start_date, end_date):
    """Load stock data using yfinance and standardize column names"""
    print(f"Loading data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Print original column names for debugging
    print("Original columns:", data.columns)
    
    # Standardize column names (flatten multi-index columns)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() for col in data.columns]
    else:
        data.columns = [col.lower() for col in data.columns]
    
    print("Standardized columns:", data.columns)
    return data


def add_technical_features(df):
    """Add technical indicators as features for ML models"""
    df_features = df.copy()
    
    # Make sure we have the expected columns
    expected_columns = ['open', 'high', 'low', 'close', 'volume']
    actual_columns = set(df_features.columns.str.lower())
    
    # For debugging
    print("Available columns:", df_features.columns.tolist())
    
    # Check for 'close' or 'adj close' column
    price_column = None
    volume_column = None
    
    if 'close' in df_features.columns:
        price_column = 'close'
    elif 'adj close' in df_features.columns:
        price_column = 'adj close'
    elif 'adjclose' in df_features.columns:
        price_column = 'adjclose'
    else:
        # Try to find any column with 'close' in it
        close_cols = [col for col in df_features.columns if 'close' in col.lower()]
        if close_cols:
            price_column = close_cols[0]
        else:
            raise ValueError("Could not find a 'close' price column in the data")
    
    if 'volume' in df_features.columns:
        volume_column = 'volume'
    else:
        volume_cols = [col for col in df_features.columns if 'volume' in col.lower()]
        if volume_cols:
            volume_column = volume_cols[0]
        else:
            print("WARNING: Could not find a volume column in the data")
            # Create a dummy volume column
            df_features['volume'] = 1000000
            volume_column = 'volume'
    
    print(f"Using '{price_column}' as price column and '{volume_column}' as volume column")
    
    # Momentum factor: 5-day price change
    df_features['momentum_5'] = df_features[price_column] / df_features[price_column].shift(5) - 1
    
    # Volume factor: (5-day avg volume) / (10-day avg volume) - 1
    df_features['vol_ratio'] = (df_features[volume_column].rolling(5).mean() / 
                               df_features[volume_column].rolling(10).mean() - 1)
    
    # RSI (14-day)
    delta = df_features[price_column].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    # Avoid division by zero
    rs = gain / loss.replace(0, np.finfo(float).eps)
    df_features['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-day, 2 std)
    sma_20 = df_features[price_column].rolling(window=20).mean()
    std_20 = df_features[price_column].rolling(window=20).std()
    df_features['BB_upper'] = sma_20 + 2 * std_20
    df_features['BB_lower'] = sma_20 - 2 * std_20
    
    # Target variable: next day return
    df_features['future_ret_1d'] = df_features[price_column].pct_change().shift(-1)
    
    # Add 'close' column for compatibility if it doesn't exist
    if price_column != 'close':
        df_features['close'] = df_features[price_column]
    
    # Remove NaN values
    df_features.dropna(inplace=True)
    
    return df_features


def split_data(df, train_pct=0.6, val_pct=0.2):
    """Split data into training, validation, and test sets"""
    train_idx = int(len(df) * train_pct)
    val_idx = int(len(df) * (train_pct + val_pct))
    
    train_data = df.iloc[:train_idx].copy()
    val_data = df.iloc[train_idx:val_idx].copy()
    test_data = df.iloc[val_idx:].copy()
    
    print(f"Training set: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} samples)")
    print(f"Validation set: {val_data.index.min()} to {val_data.index.max()} ({len(val_data)} samples)")
    print(f"Test set: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} samples)")
    
    return train_data, val_data, test_data


def prepare_features_target(train_data, val_data, test_data, features):
    """Extract feature matrices and target vectors from dataframes"""
    X_train = train_data[features].values
    y_train = train_data['future_ret_1d'].values
    
    X_val = val_data[features].values
    y_val = val_data['future_ret_1d'].values
    
    X_test = test_data[features].values
    y_test = test_data['future_ret_1d'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_linear_model(X_train, y_train, X_val, y_val):
    """Train a linear regression model with hyperparameter tuning"""
    pipeline = Pipeline([('lr', LinearRegression())])
    
    param_grid = {'lr__fit_intercept': [True, False]}
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
            print(f"Linear model update: R² = {best_score:.4f}, Params = {best_params}")
    
    return best_model, best_score


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train a random forest model with hyperparameter tuning"""
    pipeline = Pipeline([('rf', RandomForestRegressor(random_state=42))])
    
    param_grid = {
        'rf__n_estimators': [500, 1000],
        'rf__max_depth': [3, 5, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': [0.3, 'sqrt']
    }
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
            print(f"Random Forest update: R² = {best_score:.4f}, Params = {best_params}")
    
    return best_model, best_score


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost model with hyperparameter tuning"""
    pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, verbosity=0))])
    
    param_grid = {
        'xgb__n_estimators': [100, 500, 1000],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__max_depth': [3, 5, 20],
        'xgb__subsample': [0.8, 1.0]
    }
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
            print(f"XGBoost update: R² = {best_score:.4f}, Params = {best_params}")
    
    return best_model, best_score


def train_mlp(X_train, y_train, X_val, y_val):
    """Train a neural network (MLP) model with hyperparameter tuning"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, max_iter=1000))
    ])
    
    param_grid = {
        'mlp__hidden_layer_sizes': [(64, 64), (128, 128), (256, 256)],
        'mlp__alpha': [1e-4, 1e-3, 1e-2],
        'mlp__learning_rate_init': [1e-4, 1e-3, 1e-2],
        'mlp__solver': ['adam', 'sgd']
    }
    
    best_score = float('-inf')
    best_params = None
    best_model = None
    
    for params in ParameterGrid(param_grid):
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        
        val_pred = pipeline.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        
        if val_r2 > best_score:
            best_score = val_r2
            best_params = params
            best_model = copy.deepcopy(pipeline)
            print(f"MLP update: R² = {best_score:.4f}, Params = {best_params}")
    
    return best_model, best_score


def optimize_ensemble_weights(models, X_val, y_val, sum_to_1=True, nonnegative=True, alpha_l1=0.0, alpha_l2=1e-3):
    """Optimize ensemble weights using a simple approach that doesn't require cvxpy"""
    # Get predictions from each model
    predictions = np.column_stack([model.predict(X_val) for model in models])
    N, M = predictions.shape
    
    try:
        # First try using cvxpy if available
        import cvxpy as cp
        
        # Define optimization variable
        if nonnegative:
            w = cp.Variable(M, nonneg=True)
        else:
            w = cp.Variable(M)
        
        # Define constraints
        constraints = []
        if sum_to_1:
            constraints.append(cp.sum(w) == 1)
        
        # Define objective function (MSE + regularization)
        residual = y_val - predictions @ w
        obj_mse = cp.sum_squares(residual)
        
        obj_reg = 0
        if alpha_l1 > 0:
            obj_reg += alpha_l1 * cp.norm1(w)
        if alpha_l2 > 0:
            obj_reg += alpha_l2 * cp.sum_squares(w)
            
        objective = cp.Minimize(obj_mse + obj_reg)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        
        # Get optimal weights
        w_opt = w.value
        
    except ImportError:
        print("CVXPY not available, using simple R² based weights instead")
        
        # Simple alternative: use R² scores as weights
        r2_scores = []
        for i in range(M):
            r2 = r2_score(y_val, predictions[:, i])
            # Handle negative R² values
            r2_scores.append(max(0.0001, r2))
        
        # Normalize weights to sum to 1
        w_opt = np.array(r2_scores) / sum(r2_scores) if sum(r2_scores) > 0 else np.ones(M) / M
    
    # Calculate final prediction and R²
    y_val_pred = predictions @ w_opt
    r2_val = r2_score(y_val, y_val_pred)
    
    print(f"Optimized weights: {[f'{w:.2f}' for w in w_opt]}")
    print(f"Validation R² = {r2_val:.4f}")
    
    return w_opt, r2_val


class VolumeIndicator(bt.Indicator):
    """Custom volume indicator for backtrader chart"""
    lines = ('vol',)
    plotinfo = dict(subplot=True, plotname='Volume')
    
    def __init__(self):
        self.lines.vol = self.data.volume


class MLEnsembleStrategy(bt.Strategy):
    """Trading strategy that uses ML ensemble predictions"""
    params = (
        ('target_percent', 0.98),  # Target position size
    )
    
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        
        # Turn off volume in main plot
        self.data.plotinfo.plotvolume = False
        
        # Add custom volume indicator
        self.vol_ind = VolumeIndicator(self.data)
        self.vol_5 = bt.indicators.SMA(self.vol_ind.vol, period=5)
        self.vol_10 = bt.indicators.SMA(self.vol_ind.vol, period=10)
        
        # Technical indicators
        self.momentum_5 = bt.indicators.PercentChange(self.data.close, period=5)
        self.rsi_14 = bt.indicators.RSI(self.data.close, period=14)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2)
        
        # Track portfolio value history
        self.value_history_dates = []
        self.value_history_values = []
    
    def next(self):
        # Calculate features
        momentum = self.momentum_5[0]
        vol_ratio = (self.vol_5[0] / self.vol_10[0] - 1) if self.vol_10[0] != 0 else 0
        rsi = self.rsi_14[0]
        bb_upper = self.bb.top[0]
        bb_lower = self.bb.bot[0]
        
        # Create feature vector
        X = np.array([[momentum, vol_ratio, rsi, bb_upper, bb_lower]])
        
        # Get ensemble prediction
        predictions = np.array([model.predict(X)[0] for model in self.models])
        pred_ret = np.sum(predictions * self.weights)
        
        # Get current position
        current_position = self.getposition().size
        
        # Execute trades based on predictions
        if pred_ret > 0 and current_position == 0:
            # Buy signal
            self.order_target_percent(target=self.p.target_percent)
            print(f"{self.data.datetime.date(0)} => BUY signal, pred_ret={pred_ret:.6f}")
        elif pred_ret <= 0 and current_position > 0:
            # Sell signal
            self.order_target_percent(target=0.0)
            print(f"{self.data.datetime.date(0)} => SELL signal, pred_ret={pred_ret:.6f}")
        
        # Record portfolio value
        self.value_history_dates.append(self.data.datetime.date(0))
        self.value_history_values.append(self.broker.getvalue())


class BuyAndHoldStrategy(bt.Strategy):
    """Simple buy and hold strategy for comparison"""
    def __init__(self):
        self.value_history_dates = []
        self.value_history_values = []
    
    def next(self):
        # Buy on first day
        if len(self) == 1:
            self.order_target_percent(target=0.98)
            print(f"{self.data.datetime.date(0)} => BUY and HOLD")
        
        # Record portfolio value
        self.value_history_dates.append(self.data.datetime.date(0))
        self.value_history_values.append(self.broker.getvalue())


def run_backtest(df, strategy_class, strategy_params=None, initial_cash=100000):
    """Run backtrader backtest with the given strategy"""
    cerebro = bt.Cerebro()
    
    # Add data feed
    data_feed = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Use index as datetime
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None
    )
    cerebro.adddata(data_feed)
    
    # Add strategy
    if strategy_params:
        cerebro.addstrategy(strategy_class, **strategy_params)
    else:
        cerebro.addstrategy(strategy_class)
    
    # Set initial cash and commission
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run backtest
    initial_value = cerebro.broker.getvalue()
    print(f"Initial portfolio value: ${initial_value:.2f}")
    
    results = cerebro.run()
    
    final_value = cerebro.broker.getvalue()
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Return: {(final_value/initial_value - 1) * 100:.2f}%")
    
    # Get analyzer results
    sharpe = results[0].analyzers.sharpe.get_analysis()
    drawdown = results[0].analyzers.drawdown.get_analysis()
    returns = results[0].analyzers.returns.get_analysis()
    
    # Print backtrader analyzer results
    print("\nBacktrader Analyzer Results:")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 0.0):.4f}")
    
    # Safely extract drawdown values
    max_dd = 0.0
    if 'drawdown' in drawdown:
        max_dd = drawdown['drawdown']
    elif 'max' in drawdown:
        max_dd = drawdown['max']
    
    max_dd_len = 0
    if 'len' in drawdown:
        max_dd_len = drawdown['len']
    elif 'maxlen' in drawdown:
        max_dd_len = drawdown['maxlen']
    
    print(f"Max Drawdown: {max_dd * 100:.2f}%")
    print(f"Max Drawdown Length: {max_dd_len} days")
    
    return results, cerebro


def visualize_data_splits(train_data, val_data, test_data):
    """Visualize the training, validation, and test data splits"""
    plt.figure(figsize=(15, 6))
    plt.plot(train_data.index, train_data['future_ret_1d'], label='Training', color='blue')
    plt.plot(val_data.index, val_data['future_ret_1d'], label='Validation', color='green')
    plt.plot(test_data.index, test_data['future_ret_1d'], label='Test', color='red')
    
    # Add split points
    split_date_1 = train_data.index[-1]
    split_date_2 = val_data.index[-1]
    plt.axvline(split_date_1, color='black', linestyle='--')
    plt.axvline(split_date_2, color='black', linestyle='--')
    
    plt.title('Data Split: Training, Validation, and Test Sets')
    plt.xlabel('Date')
    plt.ylabel('Next-Day Return')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_performance_metrics(strategy):
    """Calculate key performance metrics for a strategy
    
    Returns a dictionary with:
    - Total return
    - Annualized return
    - Sharpe ratio
    - Maximum drawdown
    - Calmar ratio
    """
    # Extract portfolio values and dates
    values = np.array(strategy.value_history_values)
    dates = strategy.value_history_dates
    
    # Calculate returns (daily)
    returns = np.diff(values) / values[:-1]
    
    # Calculate total and annualized returns
    total_return_pct = (values[-1] / values[0] - 1) * 100
    days = (dates[-1] - dates[0]).days
    ann_return_pct = ((1 + total_return_pct/100) ** (365/days) - 1) * 100
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
    # Annualized Sharpe = (Mean Daily Return / Daily Std Dev) * sqrt(252)
    daily_return_mean = np.mean(returns)
    daily_return_std = np.std(returns)
    sharpe_ratio = (daily_return_mean / daily_return_std) * np.sqrt(252) if daily_return_std > 0 else 0
    
    # Calculate maximum drawdown
    peak = values[0]
    max_drawdown = 0
    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    max_drawdown_pct = max_drawdown * 100
    
    # Calculate Calmar ratio (annualized return / maximum drawdown)
    calmar_ratio = ann_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')
    
    return {
        'total_return_pct': total_return_pct,
        'annualized_return_pct': ann_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown_pct,
        'calmar_ratio': calmar_ratio
    }

def plot_strategy_comparison(ml_strategy, bh_strategy):
    """Plot performance comparison between ML ensemble and buy-and-hold strategies"""
    # Calculate performance metrics
    ml_metrics = calculate_performance_metrics(ml_strategy)
    bh_metrics = calculate_performance_metrics(bh_strategy)
    
    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(ml_strategy.value_history_dates, ml_strategy.value_history_values, label='ML Ensemble Strategy')
    plt.plot(bh_strategy.value_history_dates, bh_strategy.value_history_values, label='Buy and Hold Strategy')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Strategy Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Display performance metrics
    print("\n=== Performance Metrics ===")
    metrics_table = pd.DataFrame({
        'ML Ensemble': [
            f"{ml_metrics['total_return_pct']:.2f}%",
            f"{ml_metrics['annualized_return_pct']:.2f}%",
            f"{ml_metrics['sharpe_ratio']:.2f}",
            f"{ml_metrics['max_drawdown_pct']:.2f}%",
            f"{ml_metrics['calmar_ratio']:.2f}"
        ],
        'Buy and Hold': [
            f"{bh_metrics['total_return_pct']:.2f}%",
            f"{bh_metrics['annualized_return_pct']:.2f}%",
            f"{bh_metrics['sharpe_ratio']:.2f}",
            f"{bh_metrics['max_drawdown_pct']:.2f}%",
            f"{bh_metrics['calmar_ratio']:.2f}"
        ]
    }, index=['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Maximum Drawdown', 'Calmar Ratio'])
    
    print(metrics_table)


def main():
    """Main function to run the ML trading pipeline"""
    # Data parameters
    ticker = 'TSLA'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    
    # Load stock data
    print(f"Getting data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    data = load_stock_data(ticker, start_date, end_date)
    
    # Add technical indicators as features
    df = add_technical_features(data)
    
    # Define feature set
    features = ['momentum_5', 'vol_ratio', 'RSI_14', 'BB_upper', 'BB_lower']
    
    # Display feature data
    print("\nFeature Data Preview:")
    print(df[['close'] + features].tail(5))
    
    # Target variable analysis
    print(f"\nTarget Variable (Next-Day Returns) Stats:")
    print(f"Mean: {df['future_ret_1d'].mean():.6f}")
    print(f"Std Dev: {df['future_ret_1d'].std():.6f}")
    
    # Visualize returns distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['future_ret_1d'], bins=50)
    plt.title('Next-Day Return Distribution')
    plt.xlabel('Return')
    plt.show()
    
    # Correlation analysis
    corr = df[['close', 'future_ret_1d'] + features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Split data into train/validation/test sets
    train_data, val_data, test_data = split_data(df)
    
    # Visualize data splits
    visualize_data_splits(train_data, val_data, test_data)
    
    # Prepare features and targets
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_features_target(
        train_data, val_data, test_data, features
    )
    
    # Train models with hyperparameter tuning
    print("\n=== Training Linear Regression Model ===")
    lr_model, lr_score = train_linear_model(X_train, y_train, X_val, y_val)
    
    print("\n=== Training Random Forest Model ===")
    rf_model, rf_score = train_random_forest(X_train, y_train, X_val, y_val)
    
    print("\n=== Training XGBoost Model ===")
    xgb_model, xgb_score = train_xgboost(X_train, y_train, X_val, y_val)
    
    print("\n=== Training MLP Model ===")
    mlp_model, mlp_score = train_mlp(X_train, y_train, X_val, y_val)
    
    # Evaluate individual models on test set
    models = [lr_model, rf_model, xgb_model, mlp_model]
    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'MLP']
    
    print("\n=== Test Set Performance ===")
    for name, model in zip(model_names, models):
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        print(f"{name}: MSE = {test_mse:.8f}, R² = {test_r2:.4f}")
    
    # Get feature importance from tree-based models
    print("\n=== Feature Importance (Random Forest) ===")
    rf_importances = rf_model.named_steps['rf'].feature_importances_
    for feature, importance in zip(features, rf_importances):
        print(f"{feature}: {importance:.4f}")
    
    # Sort feature importances
    sorted_idx = np.argsort(rf_importances)[::-1]
    print("\nSorted Feature Importances:")
    for idx in sorted_idx:
        print(f"{features[idx]} -> {rf_importances[idx]:.4f}")
    
    # Optimize ensemble weights
    print("\n=== Optimizing Ensemble Weights ===")
    weights, ensemble_val_r2 = optimize_ensemble_weights(
        models, X_val, y_val, sum_to_1=True, nonnegative=True, alpha_l1=0.0, alpha_l2=1e-3
    )
    
    # Evaluate ensemble on test set
    test_predictions = np.column_stack([model.predict(X_test) for model in models])
    ensemble_pred = test_predictions @ weights
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    print(f"Ensemble Test R² = {ensemble_r2:.4f}")
    
    # Run backtest for ensemble strategy
    print("\n=== Running ML Ensemble Strategy Backtest ===")
    ml_results, ml_cerebro = run_backtest(
        test_data,
        MLEnsembleStrategy,
        {'models': models, 'weights': weights, 'target_percent': 0.98},
        initial_cash=100000
    )
    
    # Run backtest for buy-and-hold strategy
    print("\n=== Running Buy and Hold Strategy Backtest ===")
    bh_results, bh_cerebro = run_backtest(
        test_data,
        BuyAndHoldStrategy,
        initial_cash=100000
    )
    
    # Compare strategy performance
    ml_strategy = ml_results[0]
    bh_strategy = bh_results[0]
    plot_strategy_comparison(ml_strategy, bh_strategy)
    
    # Plot detailed backtest results
    ml_cerebro.plot(style='candlestick')
    bh_cerebro.plot(style='candlestick')


if __name__ == "__main__":
    main()