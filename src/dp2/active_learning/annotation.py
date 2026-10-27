"""Simulated annotation boundary: reveal only explicitly selected training IDs."""
import pandas as pd


class AnnotationStore:
    def __init__(self, records, training_ids):
        if set(records.columns) != {"image_id", "mask_path"} or records.image_id.duplicated().any():
            raise ValueError("Invalid private annotation store")
        self._records = records.set_index("image_id")
        self._training_ids = set(training_ids)
        if self._training_ids != set(self._records.index):
            raise ValueError("Annotation store must contain exactly the training population")

    def reveal(self, selected_ids):
        ids = list(selected_ids)
        if len(ids) != len(set(ids)) or not set(ids).issubset(self._training_ids):
            raise ValueError("Reveal requires unique training IDs; validation/test IDs forbidden")
        return self._records.loc[ids].reset_index()


def reveal_selection(experiment, state_directory, selected_ids, output, smoke=False, acquisition=None):
    """Explicit manual stage, not an automatic active-learning controller."""
    from .state import load_state, write_state
    state, labeled, pool, annotations = load_state(experiment, state_directory)
    ids = list(selected_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(set(pool.image_id)):
        raise ValueError("Selection must contain unique, currently unlabeled IDs")
    selected = pool.set_index("image_id").loc[ids].reset_index()
    new_labeled = pd.concat([labeled, selected], ignore_index=True)
    new_pool = pool[~pool.image_id.isin(ids)].reset_index(drop=True)
    revealed = pd.concat([annotations, experiment.annotations.reveal(ids)], ignore_index=True)
    return write_state(output, experiment, new_labeled, new_pool, revealed,
                       state["repetition"], state["round"] + 1,
                       smoke=smoke or state.get("smoke", False), acquisition=acquisition)
