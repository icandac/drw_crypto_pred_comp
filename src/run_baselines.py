#!/usr/bin/env python3
"""Main forecasting evaluation pipeline."""

import sys
from pathlib import Path
import warnings
from typing import Optional

# Configuration
warnings.filterwarnings("ignore", category=FutureWarning)

# Local imports
src_path = Path("../src").resolve()
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from baseline import Baseline
from config import ForecastConfig

def evaluate_models(model: Baseline, models: Optional[dict] = None) -> None:
    """
    Evaluate forecasting models and display performance metrics.
    
    Args:
        model: Initialized Baseline model instance
        models: Dictionary of models to evaluate. Defaults to basic models.
    """
    default_models = {
        'Naive': {'method': 'naive', 'kwargs': {'noise_scale': 1}},
        'Drift': {'method': 'drift', 'kwargs': {}},
        'Random': {'method': 'random', 'kwargs': {'scale': 0.1, 'random_state': 42}},
        'Mean': {'method': 'mean', 'kwargs': {}}
    }
    models_to_test = models or default_models
    
    print("\nModel Evaluation Results:")
    print("-" * 30)
    for name, config in models_to_test.items():
        getattr(model, config['method'])(**config['kwargs'])
        print(f'{name:>6} PCC = {model.pearson_corr():.3f}')

def main() -> None:
    """Execute the full forecasting pipeline."""
    try:
        cfg = ForecastConfig(plot_start=None)
        train, test = cfg.load_data(
            train_cols=['label'],
            use_actual_test=False,
            test_size=0.5
        )

        model = Baseline(train, test, window=250000)
        evaluate_models(model)

        model.random(random_state=42)
        #model.plot_forecast(start_date_str=train.index[0])
        #model.save_forecast('submission_rnd.csv')

    except FileNotFoundError as e:
        print(f"\nData loading failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()