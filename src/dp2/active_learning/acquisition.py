"""Pure acquisition policies. No masks, ground truth or error metrics accepted."""
import numpy as np
import pandas as pd

SCORE_COLUMNS = ("image_id", "method", "inference_seed", "bald", "image_uncertainty",
                 "image_uncertainty_valid", "vcdr_uncertainty", "vcdr_uncertainty_valid",
                 "empty_foreground", "roi_pixels", "vcdr_missing_cup_fraction")


class AllScoresInvalid(ValueError):
    pass


def _valid_flags(values):
    mapping = {True: True, False: False, "True": True, "False": False,
               "true": True, "false": False}
    if not values.isin(list(mapping)).all():
        raise ValueError("Invalid boolean validity flag")
    return values.map(mapping).astype(bool)


def select_images(pool_ids, strategy, count, seed, scores=None):
    """Return selected IDs plus a full seeded ranking; ties ignore input row order."""
    ids = list(pool_ids)
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Pool IDs must be unique nonempty strings")
    if type(count) is not int or count < 1:
        raise ValueError("Acquisition size must be a positive integer")
    if strategy not in ("random", "bald", "vcdr"):
        raise ValueError("Unknown strategy")
    if not ids:
        raise ValueError("Cannot acquire from an empty pool")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid acquisition seed")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(ids)).tolist()
    ranked = pd.DataFrame({"image_id": shuffled, "tie_order": range(len(ids))})
    if strategy == "random":
        if scores is not None:
            raise ValueError("Random acquisition does not accept scores")
        ranked["score"] = np.nan
        ranked["score_valid"] = False
        ranked["fallback"] = False
    else:
        if scores is None or set(scores.columns) - set(SCORE_COLUMNS):
            raise ValueError("Scoring table missing or contains forbidden annotation/evaluation columns")
        column = "bald" if strategy == "bald" else "vcdr_uncertainty"
        flag = "image_uncertainty_valid" if strategy == "bald" else "vcdr_uncertainty_valid"
        if not {"image_id", column, flag}.issubset(scores.columns):
            raise ValueError("Missing score or validity column")
        if scores.image_id.duplicated().any() or set(scores.image_id) != set(ids):
            raise ValueError("Scores must cover the current pool exactly once")
        scores = scores.set_index("image_id").loc[shuffled]
        ranked["score"] = pd.to_numeric(scores[column], errors="raise").to_numpy()
        ranked["score_valid"] = _valid_flags(scores[flag]).to_numpy() & np.isfinite(ranked.score)
        if (ranked.loc[ranked.score_valid, "score"] < 0).any():
            raise ValueError("Uncertainty scores cannot be negative")
        if not ranked.score_valid.any():
            raise AllScoresInvalid("All pool scores are invalid; inspect the predictor before acquiring")
        ranked["fallback"] = ~ranked.score_valid
        ranked["sort_score"] = ranked.score.where(ranked.score_valid, -np.inf)
        ranked = ranked.sort_values(["score_valid", "sort_score", "tie_order"],
                                     ascending=[False, False, True], kind="stable").drop(columns="sort_score")
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["strategy"] = strategy
    ranked["acquisition_seed"] = seed
    ranked["selected"] = ranked["rank"] <= min(count, len(ids))
    return ranked[ranked.selected].copy(), ranked
