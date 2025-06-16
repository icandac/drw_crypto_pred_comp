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
        last_val = float(self.training.iloc[-1].iloc[0])

        noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(self.test))
        predictions = last_val + noise
        self.forecast = pd.DataFrame({"prediction": predictions}, index=self.test.index)
        return self.forecast

    def random(self, scale: float = 0.1, random_state: None = None) -> pd.DataFrame:
        """
        Random baseline forecast:
        - Uses the mean of the last 'window' observations as the baseline.
        - Adds Gaussian noise scaled by the training data's standard deviation.
        - Sets a fixed random seed for reproducibility.
    
        Args:
            scale (float): Controls noise magnitude (default: 0.1).
            random_state (int): Seed for reproducibility (default: None).
    
        Returns:
            pd.DataFrame: Forecast with columns ['y_pred'], indexed to match self.test.
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Baseline: Mean of the last window (ensure scalar with .item())
        baseline = self.training.iloc[-self.window :].mean().item()

        # Noise: Scaled by training std (avoid zero with max(..., 1e-6))
        training_std = max(self.training.iloc[-self.window :].std().item(), 1e-6)
        noise = np.random.normal(loc=0, scale=scale * training_std, size=len(self.test))

        # Forecast: Baseline + noise (remove .cumsum() unless intentional)
        self.forecast = pd.DataFrame(
            # Replace with `noise` if no baseline desired
            {"prediction": baseline + noise.cumsum()},
            index=self.test.index,
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
        if hasattr(min_val, "iloc"):
            min_val = min_val.iloc[0]
        if hasattr(max_val, "iloc"):
            max_val = max_val.iloc[0]

        # Calculate allowed slopes
        max_slope = (max_val - end) / periods
        min_slope = (min_val - end) / periods

        # Clip slope
        slope = np.clip(raw_slope, min_slope, max_slope)

        # Calculate drift values
        drift_values = end + slope * np.arange(1, periods + 1)

        # Final forecast DataFrame
        self.forecast = pd.DataFrame(
            {"prediction": drift_values}, index=self.test.index
        )

    def mean(self) -> pd.DataFrame:
        """
        Mean forecast: use the rolling mean of the last 'window' observations.

        Returns:
            pd.DataFrame: Forecast where every value is the computed mean.
        """
        mean_val = float(self.training.iloc[-self.window :].mean())
        predictions = np.full(len(self.test), mean_val)
        self.forecast = pd.DataFrame({"y_pred": predictions}, index=self.test.index)
        return self.forecast

    def pearson_corr(self) -> float:
        """
        Compute the Pearson correlation coefficient between true values and forecasts.

        Returns:
            float: Pearson correlation coefficient.
        """
        y_true = self.test.squeeze()
        y_pred = self.forecast["prediction"].reindex(y_true.index)
        return y_true.corr(y_pred)

    def plot_forecast(self, start_date_str: str = "2024-02-29 18:00:00"):
        """
        Plot training, forecast, and test series with legend and x-axis
        limited from start_date.

        Args:
            start_date_str (str): Left limit of x-axis as a date string.
        """
        start_date = pd.to_datetime(start_date_str)

        fig, ax = plt.subplots(figsize=(10, 5))

        self.training["label"].plot(ax=ax, label="train")
        self.forecast["prediction"].plot(ax=ax, label="forecast")
        self.test["label"].plot(ax=ax, label="test")

        ax.legend()
        ax.set_xlim(left=start_date)
        ax.set_title("Training, Forecast, and Test Data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

        plt.show()
