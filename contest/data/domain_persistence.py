import json
import os
from pathlib import Path
import numpy as np

from contest.data import event_log as el


DOMAIN_FILENAME = "domain.json"
RESOURCE_MATRIX_FILENAME = "resource_activity_matrix.npy"


def save_domain(weights_dir: str, dataset_name: str = None) -> None:
   
    p = Path(weights_dir)
    p.mkdir(parents=True, exist_ok=True)

    domain = {
        "dataset_name": dataset_name,
        "activities": list(el.ACTIVITIES),
        "resources": list(el.RESOURCES),
        "scalar_names": list(el._SCALAR_NAMES),
        "cat_names": list(el._CAT_NAMES),
        "cat_vocabularies": {k: dict(v) for k, v in el.CAT_VOCABULARIES.items()},
        "numeric_min_max": {k: list(v) for k, v in el.NUMERIC_MIN_MAX.items()},
        "event_dim": int(el.EVENT_DIM),
        "case_invariant_features": list(el.CASE_INVARIANT_FEATURES),
        "activity_determined_categoricals": list(el.ACTIVITY_DETERMINED_CATEGORICALS),
        "activity_to_category_maps": {
            k: v.tolist() for k, v in el.ACTIVITY_TO_CATEGORY_MAPS.items()
        },
    }
    with open(p / DOMAIN_FILENAME, "w") as f:
        json.dump(domain, f, indent=2)

    if el.RESOURCE_ACTIVITY_MATRIX.size > 0:
        np.save(p / RESOURCE_MATRIX_FILENAME, el.RESOURCE_ACTIVITY_MATRIX)

    tag = f" dataset={dataset_name}" if dataset_name else ""
    print(f"  Domain snapshot saved → {p / DOMAIN_FILENAME}{tag}")
    print(f"    activities={len(domain['activities'])} resources={len(domain['resources'])} "
          f"event_dim={domain['event_dim']}")


def load_domain(weights_dir: str, expected_dataset: str = None) -> None:
    
    p = Path(weights_dir)
    domain_path = p / DOMAIN_FILENAME
    if not domain_path.exists():
        raise FileNotFoundError(
            f"No domain snapshot found at {domain_path}. "
            f"This checkpoint was saved before domain persistence was added — "
            f"re-run training with save_domain() to produce one, otherwise "
            f"eval/inference is not guaranteed to align with these weights."
        )

    with open(domain_path) as f:
        domain = json.load(f)

    snapshot_dataset = domain.get("dataset_name")
    if expected_dataset is not None:
        if snapshot_dataset is None:
            print(f"  WARNING: domain snapshot at {domain_path} has no dataset_name tag "
                  f"(saved before dataset tagging was added) — cannot verify it is "
                  f"'{expected_dataset}'. Proceeding, but re-save with save_domain(..., "
                  f"dataset_name=...) to close this gap.")
        elif snapshot_dataset != expected_dataset:
            raise RuntimeError(
                f"Domain mismatch: expected dataset '{expected_dataset}' but snapshot at "
                f"{domain_path} was saved for dataset '{snapshot_dataset}'. Refusing to load "
                f"— this weights_dir belongs to a different event log."
            )

    el.ACTIVITIES.clear(); el.ACTIVITIES.extend(domain["activities"])
    el.RESOURCES.clear(); el.RESOURCES.extend(domain["resources"])
    el.ACT2IDX.clear(); el.ACT2IDX.update({a: i for i, a in enumerate(el.ACTIVITIES)})
    el.RES2IDX.clear(); el.RES2IDX.update({r: i for i, r in enumerate(el.RESOURCES)})
    el.IDX2ACT.clear(); el.IDX2ACT.update({i: a for a, i in el.ACT2IDX.items()})
    el.IDX2RES.clear(); el.IDX2RES.update({i: r for r, i in el.RES2IDX.items()})

    el._SCALAR_NAMES.clear(); el._SCALAR_NAMES.extend(domain["scalar_names"])
    el._CAT_NAMES.clear(); el._CAT_NAMES.extend(domain["cat_names"])

    el.CAT_VOCABULARIES.clear()
    el.CAT_VOCABULARIES.update({k: dict(v) for k, v in domain["cat_vocabularies"].items()})

    el.NUMERIC_MIN_MAX.clear()
    el.NUMERIC_MIN_MAX.update({k: tuple(v) for k, v in domain["numeric_min_max"].items()})

    el.EVENT_DIM = domain["event_dim"]
    el.CASE_INVARIANT_FEATURES = list(domain["case_invariant_features"])

    el.ACTIVITY_DETERMINED_CATEGORICALS = list(domain.get("activity_determined_categoricals", []))
    el.ACTIVITY_TO_CATEGORY_MAPS = {
        k: np.array(v, dtype=np.float32)
        for k, v in domain.get("activity_to_category_maps", {}).items()
    }

    matrix_path = p / RESOURCE_MATRIX_FILENAME
    if matrix_path.exists():
        el.RESOURCE_ACTIVITY_MATRIX = np.load(matrix_path)

    n_names = len(el.feature_names())
    if n_names != el.EVENT_DIM:
        raise RuntimeError(
            f"Domain snapshot inconsistent: feature_names() produces {n_names} "
            f"columns but EVENT_DIM={el.EVENT_DIM}. Snapshot may be corrupted."
        )

    tag = f" dataset={snapshot_dataset}" if snapshot_dataset else ""
    print(f"  Domain restored from {domain_path}{tag}")
    print(f"    activities={len(el.ACTIVITIES)} resources={len(el.RESOURCES)} "
          f"event_dim={el.EVENT_DIM}")
