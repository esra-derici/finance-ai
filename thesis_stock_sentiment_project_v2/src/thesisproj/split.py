
import pandas as pd
from typing import Tuple
def chrono_split(df: pd.DataFrame, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    return df.iloc[:-test_days].copy(), df.iloc[-test_days:].copy()
