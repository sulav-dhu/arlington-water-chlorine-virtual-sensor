"""Paths and parameters for the cleaning and evaluation pipelines. Every tunable choice lives here."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CleaningConfig:
    version: str = "v1"
    raw_xlsx: Path = ROOT / "Chlorine_residual_2015-2027.xlsx"
    # SHA-256 of the export this version was built and verified against; a mismatch is reported.
    expected_raw_sha256: str = "7d2377bb6e9b363ea30134f0ab297ab820fc453edd25847dd7e075e0c2246460"
    gis_dir: Path = ROOT / "data" / "external" / "arlington_gis"
    reference_dir: Path = ROOT / "config"
    out_root: Path = ROOT / "data" / "processed"

    # Modeling scope: routine bacteriological + daily chlorine programs.
    scope_projects: tuple[str, ...] = ("Monthly Bact", "Monthly Cl")

    # Location resolution. Two addresses are the same physical place when their street names
    # are similar and house numbers are within a few lots (~50-150 m).
    same_place_max_number_gap: int = 30
    same_street_min_similarity: float = 0.85
    # A per-code address cluster with at least this many rows is a real location, not a typo.
    location_min_rows: int = 5
    # A small cluster this far outside a code's active period is a later re-use of the code.
    code_reuse_min_gap_days: int = 180

    # Plant on/off inference from gaps in plant sampling.
    plant_gap_min_days: int = 28
    dip_flank_days: int = 56
    jk_dip_supported_mg_l: float = -0.25

    # Value plausibility for total chlorine (mg/L). Plant effluent tops out ~5.1.
    chlorine_max_mg_l: float = 6.0

    # Geocoding and hydrant-derived features.
    max_nearest_number_gap: int = 40
    hydrant_k: int = 7
    neighborhood_radius_m: float = 500.0
    suspicious_hydrant_distance_m: float = 300.0

    @property
    def out_dir(self) -> Path:
        return self.out_root / self.version


@dataclass(frozen=True)
class EvalConfig:
    data_version: str = "v1"
    processed_root: Path = ROOT / "data" / "processed"
    out_root: Path = ROOT / "data" / "eval"

    # Time splits (inclusive ends). Validation selects models; the test period is touched only with
    # --split test, and every such run is logged to data/eval/test_access_log.jsonl.
    train_end: str = "2022-12-31"
    val_end: str = "2023-12-31"

    # Unseen-site protocol: place groups split into folds, stratified by pressure plane.
    n_folds: int = 5
    seed: int = 20261012

    # Reference-model settings, fixed a priori (not tuned on validation or test).
    idw_length_km: float = 3.0        # exponential distance kernel exp(-d / L) for neighbor weighting
    blend_k_places: float = 2.0       # group-neighbor mean vs. prior: n / (n + k), n = neighbor places
    blend_k_weight: float = 1.0       # same for kernel-weighted means: W / (W + k)
    offset_shrink_rows: float = 20.0  # empirical-Bayes shrinkage of group offsets toward their parent
    scale_min_rows: int = 30          # min residuals for a site-specific spread in prediction intervals

    # Metrics.
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    low_thresholds: tuple[float, ...] = (2.5, 3.0)   # "low residual" events for tail metrics, mg/L
    min_site_eval_samples: int = 10                  # for site-level ranking metrics
    bootstrap_reps: int = 1000                       # cluster bootstrap over place groups

    @property
    def data_dir(self) -> Path:
        return self.processed_root / self.data_version
