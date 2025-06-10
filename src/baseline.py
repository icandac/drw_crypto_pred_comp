from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Baseline:
    """
    A simple forecasting baseline model with naive, drift, and mean methods.

    Attributes:
        training (pd.Series): Historical time series for model fitting.
        test (pd.Series): Time series to forecast.
        window (int): Rolling window size for drift and mean forecasts.
        forecast (pd.DataFrame): Container for forecasted values.
    """

    def __init__(
        self,
        training: pd.Series,
        test: pd.Series,
        window: int = 1
    ) -> None:
        self.training = training
        self.test = test
        self.window = window
        self.forecast = pd.DataFrame(index=test.index)

    def naive(self, noise_scale: float = 0.1) -> pd.DataFrame:
        """
        Naive forecast: repeat the last observed value plus Gaussian noise.

        Args:
            noise_scale (float): Standard deviation of additive Gaussian noise.

        Returns:
            pd.DataFrame: Forecast values with index matching the test set.
        """
        last_val = float(self.training.iloc[-1])
        noise = np.random.normal(
            loc=0.0,
            scale=noise_scale,
            size=len(self.test)
        )
        predictions = last_val + noise
        self.forecast = pd.DataFrame(
            {'y_pred': predictions},
            index=self.test.index
        )
        return self.forecast
        
    def drift(self) -> pd.DataFrame:
        start = float(self.training.iloc[-self.window].iloc[0])
        end = float(self.training.iloc[-1].iloc[0])

        raw_slope = (end - start) / (self.window - 1)
        periods = len(self.test)
    
        # Ensure min_val and max_val are scalars
        min_val = self.training.min()
        max_val = self.training.max()
        if hasattr(min_val, 'iloc'):
            min_val = min_val.iloc[0]
        if hasattr(max_val, 'iloc'):
            max_val = max_val.iloc[0]
    
        # Calculate allowed slopes
        max_slope = (max_val - end) / periods
        min_slope = (min_val - end) / periods
    
        # Clip slope
        slope = np.clip(raw_slope, min_slope, max_slope)
    
        # Calculate drift values
        drift_values = end + slope * np.arange(1, periods + 1)
    
        # Final forecast DataFrame
        self.forecast = pd.DataFrame({'y_pred': drift_values}, index=self.test.index)

    def mean(self) -> pd.DataFrame:
        """
        Mean forecast: use the rolling mean of the last 'window' observations.

        Returns:
            pd.DataFrame: Forecast where every value is the computed mean.
        """
        mean_val = float(self.training.iloc[-self.window:].mean())
        predictions = np.full(len(self.test), mean_val)
        self.forecast = pd.DataFrame(
            {'y_pred': predictions},
            index=self.test.index
        )
        return self.forecast
        
    def pearson_corr(self) -> float:
        """
        Compute the Pearson correlation coefficient between true values and forecasts.

        Returns:
            float: Pearson correlation coefficient.
        """
        y_true = self.test.squeeze()
        y_pred = self.forecast['y_pred'].reindex(y_true.index)
        return y_true.corr(y_pred)