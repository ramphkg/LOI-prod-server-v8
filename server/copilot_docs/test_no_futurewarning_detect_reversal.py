import warnings
from contextlib import contextmanager
import pandas as pd, numpy as np
from TrendReversalDetectorFunction import detect_reversal_pro

@contextmanager
def assert_no_future_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield w
        for warn in w:
            if issubclass(warn.category, FutureWarning):
                raise AssertionError(f'Found unexpected FutureWarning: {warn.message}')


def test_detect_reversal_no_future_warning_datetime_index():
    dates = pd.date_range('2024-01-01', periods=40, freq='D')
    np.random.seed(1)
    prices = 100 + np.cumsum(np.random.randn(40))
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': np.random.randint(100,1000,40)
    })
    with assert_no_future_warning():
        detect_reversal_pro(df, verbose=False)


def test_detect_reversal_no_future_warning_range_index():
    dates = pd.date_range('2024-01-01', periods=40, freq='D')
    np.random.seed(1)
    prices = 100 + np.cumsum(np.random.randn(40))
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': prices + 1,
        'Low': prices - 1,
        'Close': prices,
        'Volume': np.random.randint(100,1000,40)
    }).reset_index(drop=True)
    with assert_no_future_warning():
        detect_reversal_pro(df, verbose=False)

if __name__ == '__main__':
    test_detect_reversal_no_future_warning_datetime_index()
    test_detect_reversal_no_future_warning_range_index()
    print('All no-FutureWarning tests passed')
