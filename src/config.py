from pathlib import Path
import sys
from typing import Optional
import warnings

import pandas as pd

class ForecastConfig:
    """Handles configuration and data loading for forecasting tasks."""
    
    def __init__(self, 
                 data_dir: str = '../rawdata',
                 src_dir: str = '../src',
                 plot_start: str = None):
        self.DATA_DIR = Path(data_dir)
        self.SRC_DIR = Path(src_dir).resolve()
        self.PLOT_START_DATE = plot_start
        self._setup_environment()
        
    def _setup_environment(self):
        """Configure system paths and warnings."""
        warnings.filterwarnings('ignore', category=UserWarning)
        if str(self.SRC_DIR) not in sys.path:
            sys.path.append(str(self.SRC_DIR))
    
    def load_data(
        self, 
        train_cols: list = ['label'],
        test_cols: list = ['label'],
        use_actual_test: bool = True,
        test_size: Optional[float] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data with flexible options.
        
        Args:
            train_cols: Columns to load from training set
            test_cols: Columns to load from test set
            use_actual_test: If True, uses separate test file; if False, splits train data
            test_size: If splitting train data, fraction to use as test (0.0-1.0)
            
        Returns:
            Tuple of (training_data, testing_data)
        """
        train = pd.read_parquet(self.DATA_DIR / 'train.parquet', columns=train_cols)
        
        if use_actual_test:
            test = pd.read_parquet(self.DATA_DIR / 'test.parquet', columns=test_cols)
        else:
            if test_size is None:
                test_size = 0.2  # Default split ratio
                
            split_point = int(len(train) * (1 - test_size))
            test = train.iloc[split_point:]
            train = train.iloc[:split_point]
        
        return train, test