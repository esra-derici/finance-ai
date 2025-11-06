
from dataclasses import dataclass, field
from typing import List
@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: ["THYAO","BIMAS","TUPRS"])
    data_dir: str = "data"
    results_dir: str = "results"
    test_days: int = 60
    random_state: int = 42
    n_estimators: int = 400
