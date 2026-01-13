import pandas as pd
from typing import List


def make_alerts(doc_ids: List[str], drift_scores, threshold: float):
    df = pd.DataFrame({
        "doc_id": doc_ids,
        "drift_score": drift_scores,
        "threshold": threshold,
        "alert": drift_scores > threshold,
    })
    return df
