import pandas as pd
from pathlib import Path

Path("data/health_study_dataset.csv")

def load_data(path: str = "data/health_study_dataset.csv") -> pd.DataFrame:

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Kan inte hitta datafilen: {p.resolve()}")
    
    return pd.read_csv(p)