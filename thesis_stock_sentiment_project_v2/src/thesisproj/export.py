
from __future__ import annotations
import pandas as pd
from pathlib import Path

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def save_table(df: pd.DataFrame, out_csv: Path, out_tex: Path, caption: str, label: str):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    # basic LaTeX export
    tex = df.to_latex(index=False, escape=True, caption=caption, label=label)
    out_tex.write_text(tex)
