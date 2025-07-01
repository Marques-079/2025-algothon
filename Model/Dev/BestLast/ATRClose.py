import numpy as np
import pandas as pd
# Average True Range (ATR) (Close-to-Close Volatility approximates this)
def atr_close_to_close(close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Approximate Average True Range (ATR)
    Normally uses high/low/close; here approximated with abs diff of close.
    Measures volatility for detecting quiet vs explosive periods.
    """
    diff = np.abs(np.diff(close, prepend=np.nan))
    return pd.Series(diff).rolling(window).mean().to_numpy()
