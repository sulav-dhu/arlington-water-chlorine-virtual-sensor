"""Equal-weight average of member models (point predictions and quantiles alike)."""
from __future__ import annotations

import pandas as pd

from ..evaluation.models import Model


class Ensemble(Model):
    def __init__(self, cfg, code, name, members: list[Model]):
        super().__init__(cfg)
        self.code, self.name, self.members = code, name, members
        self.uses_site_history = any(m.uses_site_history for m in members)

    def bind(self, pool, panel):
        for m in self.members:
            m.bind(pool, panel)

    def fit(self, train) -> "Ensemble":
        for m in self.members:
            m.fit(train)
        return self

    def predict(self, df) -> pd.DataFrame:
        preds = [m.predict(df) for m in self.members]
        cols = ["pred"] + [f"q{int(round(q * 100))}" for q in self.cfg.quantiles]
        out = sum(p[cols] for p in preds) / len(preds)       # quantile averaging (Vincentization)
        # Pass through the regression-kriging member's predictive SD (used to scale conformal intervals).
        sd = next((p["sd"] for p in preds if "sd" in p), None)
        if sd is not None:
            out["sd"] = sd
        return out
