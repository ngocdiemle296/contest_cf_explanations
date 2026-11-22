from dataclasses import dataclass, field, fields
from typing import Iterable, List, Dict, Optional, Any
import numpy as np
import torch


ACTIVITIES: List[str] = []
RESOURCES: List[str] = []
ACT2IDX: Dict[str, int] = {}
RES2IDX: Dict[str, int] = {}
IDX2ACT: Dict[int, str] = {}
IDX2RES: Dict[int, str] = {}
EVENT_DIM: int = 0

AMT_MIN = 0.0
DENOM = 1.0

_SCALAR_NAMES: List[str] = []
_CAT_NAMES: List[str] = []

NUMERIC_MIN_MAX: Dict[str, tuple] = {}
CAT_VOCABULARIES: Dict[str, Dict[str, int]] = {}


import numpy as np
from typing import List

RESOURCE_ACTIVITY_MATRIX: np.ndarray = np.array([])
RESOURCE_ACTIVITY_COUNTS = None

def build_resource_activity_matrix(cases: "List[Case]", min_observations: int = 1) -> np.ndarray:
    global RESOURCE_ACTIVITY_MATRIX
    global RESOURCE_ACTIVITY_COUNTS

    counts = np.zeros((len(RESOURCES), len(ACTIVITIES)), dtype=np.int32)

    for case in cases:
        for event in case.events:
            act_idx = ACT2IDX.get(event.activity, -1)
            res_idx = RES2IDX.get(event.resource, -1)
            if act_idx >= 0 and res_idx >= 0:
                counts[res_idx, act_idx] += 1

    RESOURCE_ACTIVITY_MATRIX = (counts >= min_observations).astype(np.float32)
    RESOURCE_ACTIVITY_COUNTS = counts
    return RESOURCE_ACTIVITY_MATRIX, RESOURCE_ACTIVITY_COUNTS


def pm_row_entropy_floor(PM: torch.Tensor, cases: List["Case"], n_activities: int) -> float:
    visit_counts = torch.zeros(n_activities)
    for c in cases:
        for e in c.events[:-1]:
            visit_counts[ACT2IDX[e.activity]] += 1
    visit_freq = visit_counts / visit_counts.sum().clamp(min=1e-8)

    row_entropy = -(PM * torch.log(PM.clamp(min=1e-8))).sum(dim=1)
    return (visit_freq * row_entropy).sum().item()

def is_feasible_pair(activity: str, resource: str) -> bool:
    act_idx = ACT2IDX.get(activity, -1)
    res_idx = RES2IDX.get(resource, -1)
    if act_idx < 0 or res_idx < 0:
        return False
    if RESOURCE_ACTIVITY_MATRIX.size == 0:
        return True
    return bool(RESOURCE_ACTIVITY_MATRIX[res_idx, act_idx])

DERIVED_FEATURE_PREFIXES = ("freq_",)

def is_derived_feature(name: str) -> bool:
    return name.startswith(DERIVED_FEATURE_PREFIXES)

CASE_INVARIANT_FEATURES: List[str] = []


ACTIVITY_DETERMINED_CATEGORICALS: List[str] = []
ACTIVITY_TO_CATEGORY_MAPS: Dict[str, np.ndarray] = {}

def categorical_block_columns(cat_name: str) -> List[str]:
    vocab = CAT_VOCABULARIES.get(cat_name, {})
    if not vocab:
        return []
    return [f"{cat_name}_{v}" for v in sorted(vocab.keys())]


def discover_activity_determined_categoricals(cases: "List[Case]") -> List[str]:
    global ACTIVITY_DETERMINED_CATEGORICALS

    determined = []
    for cat_name in _CAT_NAMES:
        activity_to_values: Dict[str, set] = {}
        for case in cases:
            for ev in case.events:
                val = ev.attributes.get(cat_name)
                if val is None:
                    continue
                activity_to_values.setdefault(ev.activity, set()).add(val)
        if activity_to_values and all(len(v) == 1 for v in activity_to_values.values()):
            determined.append(cat_name)

    ACTIVITY_DETERMINED_CATEGORICALS = determined
    return determined


def build_activity_to_category_maps(cases: "List[Case]") -> Dict[str, np.ndarray]:
    global ACTIVITY_TO_CATEGORY_MAPS

    maps: Dict[str, np.ndarray] = {}
    for cat_name in ACTIVITY_DETERMINED_CATEGORICALS:
        vocab = CAT_VOCABULARIES.get(cat_name, {})
        if not vocab:
            continue
        M = np.zeros((len(ACTIVITIES), len(vocab)), dtype=np.float32)
        seen = set()
        for case in cases:
            for ev in case.events:
                a = ev.activity
                if a in seen or a not in ACT2IDX:
                    continue
                val = ev.attributes.get(cat_name)
                if val in vocab:
                    M[ACT2IDX[a], vocab[val]] = 1.0
                    seen.add(a)
        maps[cat_name] = M

    ACTIVITY_TO_CATEGORY_MAPS = maps
    return maps


def discover_case_invariant_features(cases: "List[Case]", tol: float = 1e-9) -> List[str]:
    global CASE_INVARIANT_FEATURES

    names = feature_names()
    n_features = len(names)

    event_level_exact = set(ACTIVITIES) | set(RESOURCES)

    is_constant = np.ones(n_features, dtype=bool)
    any_multi_event_case = False

    for case in cases:
        if len(case.events) < 2:
            continue
        any_multi_event_case = True
        mat = case.to_matrix()
        col_range = mat.max(axis=0) - mat.min(axis=0)
        is_constant &= (col_range <= tol)

    invariant_names = []
    if any_multi_event_case:
        for i, name in enumerate(names):
            if name in event_level_exact:
                continue
            if is_derived_feature(name):
                continue
            if is_constant[i]:
                invariant_names.append(name)

    CASE_INVARIANT_FEATURES = invariant_names
    return invariant_names


def denormalize_feature(feature_name: str, value: float) -> float:
    if feature_name in NUMERIC_MIN_MAX:
        amin, amax = NUMERIC_MIN_MAX[feature_name]
        denom = amax - amin if amax > amin else 1.0
        return value * denom + amin
    if feature_name == "AMOUNT_REQ" and "AMOUNT_REQ" not in NUMERIC_MIN_MAX:
        return value * DENOM + AMT_MIN
    return value


def set_domain(activities: List[str],
               resources: List[str],
               cases_log: Optional[List[Any]] = None,
               discovered_vocabularies: Optional[Dict[str, List[str]]] = None) -> None:
    ACTIVITIES.clear()
    ACTIVITIES.extend(list(activities))
    RESOURCES.clear()
    RESOURCES.extend(list(resources))

    ACT2IDX.clear()
    ACT2IDX.update({a: i for i, a in enumerate(ACTIVITIES)})
    RES2IDX.clear()
    RES2IDX.update({r: i for i, r in enumerate(RESOURCES)})

    IDX2ACT.clear()
    IDX2ACT.update({i: a for a, i in ACT2IDX.items()})
    IDX2RES.clear()
    IDX2RES.update({i: r for r, i in RES2IDX.items()})

    global _SCALAR_NAMES, _CAT_NAMES, EVENT_DIM, CAT_VOCABULARIES
    _SCALAR_NAMES.clear()
    _CAT_NAMES.clear()
    CAT_VOCABULARIES.clear()

    if cases_log and len(cases_log) > 0 and len(cases_log[0].events) > 0:
        sample_event = cases_log[0].events[0]
        for key, val in sample_event.attributes.items():
            if key in ["activity", "resource", "remaining_time"]:
                continue

            if isinstance(val, (int, float)) and not np.isnan(val) and not isinstance(val, bool):
                _SCALAR_NAMES.append(key)
            elif isinstance(val, str):
                _CAT_NAMES.append(key)

        _SCALAR_NAMES.sort()
        _CAT_NAMES.sort()

    if discovered_vocabularies:
        for attr_name, categories in discovered_vocabularies.items():
            if attr_name in _CAT_NAMES:
                CAT_VOCABULARIES[attr_name] = {cat: idx for idx, cat in enumerate(sorted(categories))}

    total_cat_dim = sum(len(vocab) for vocab in CAT_VOCABULARIES.values())

    EVENT_DIM = len(ACTIVITIES) + len(RESOURCES) + total_cat_dim + len(_SCALAR_NAMES) + len(ACTIVITIES)


def feature_names() -> List[str]:
    names = [f"{a}" for a in ACTIVITIES] + [f"{r}" for r in RESOURCES]
    for attr in _CAT_NAMES:
        for cat in sorted(CAT_VOCABULARIES.get(attr, {}).keys()):
            names.append(f"{attr}_{cat}")
    names.extend(_SCALAR_NAMES)
    names.extend([f"freq_{a}" for a in ACTIVITIES])
    return names

def compute_trace_frequencies(events_up_to_t: Iterable) -> np.ndarray:
    freq_vector = np.zeros(len(ACTIVITIES), dtype=np.float32)
    for ev in events_up_to_t:
        if ev.activity in ACT2IDX:
            freq_vector[ACT2IDX[ev.activity]] += 1.0
    return freq_vector / max(1.0, float(len(list(events_up_to_t))))

def build_activity_transition_matrix(cases, n_activities):
    counts = torch.zeros(n_activities, n_activities)
    for c in cases:
        act_idxs = [ACT2IDX[e.activity] for e in c.events]
        for a_t, a_t1 in zip(act_idxs[:-1], act_idxs[1:]):
            counts[a_t, a_t1] += 1.0
    row_sums = counts.sum(dim=1, keepdim=True).clamp(min=1e-8)
    PM = counts / row_sums
    return PM


@dataclass
class Event:
    attributes: Dict[str, Any] = field(default_factory=dict)
    remaining_time: Optional[float] = None

    def __getattr__(self, name: str) -> Any:
        if name in self.attributes:
            return self.attributes[name]

        aliases = {
            "loan_amount":    ["AMOUNT_REQ", "case:AMOUNT_REQ",
                                "RequestedAmount", "case:RequestedAmount",
                                "loan_amount"],
            "execution_time": ["event_duration", "execution_time"],
            "workload_res":   ["resource_workload", "workload_res"]
        }

        if name in aliases:
            for potential_key in aliases[name]:
                if potential_key in self.attributes:
                    return self.attributes[potential_key]

        raise AttributeError(f"'Event' object has no attribute '{name}'")

    @property
    def activity(self) -> str:
        return self.attributes.get("activity", "Unknown")

    @property
    def resource(self) -> str:
        return self.attributes.get("resource", "Resource_unknown")

    def to_vector(self, history_events: List) -> np.ndarray:
        if not ACTIVITIES or not RESOURCES:
            raise ValueError("Domain not initialized; call set_domain first.")

        vectors = []

        act_oh = np.zeros(len(ACTIVITIES), dtype=np.float32)
        if self.activity in ACT2IDX:
            act_oh[ACT2IDX[self.activity]] = 1.0
        vectors.append(act_oh)

        res_oh = np.zeros(len(RESOURCES), dtype=np.float32)
        if self.resource in RES2IDX:
            res_oh[RES2IDX[self.resource]] = 1.0
        vectors.append(res_oh)

        for attr in _CAT_NAMES:
            vocab = CAT_VOCABULARIES.get(attr, {})
            oh = np.zeros(len(vocab), dtype=np.float32)
            val = self.attributes.get(attr)
            if val in vocab:
                oh[vocab[val]] = 1.0
            vectors.append(oh)

        scalar_list = []
        for name in _SCALAR_NAMES:
            val = float(self.attributes.get(name, 0.0))
            if name in NUMERIC_MIN_MAX:
                amin, amax = NUMERIC_MIN_MAX[name]
                denom = amax - amin if amax > amin else 1.0
                val = (val - amin) / denom
            scalar_list.append(val)
        vectors.append(np.array(scalar_list, dtype=np.float32))

        vectors.append(compute_trace_frequencies(history_events))

        return np.concatenate(vectors)


@dataclass
class Case:
    case_id: str
    events: List[Event] = field(default_factory=list)

    def partial_trace(self, t: int) -> List[Event]:
        return self.events[:t]

    def to_matrix(self, t: Optional[int] = None) -> np.ndarray:
        events = self.events if t is None else self.events[:t]
        if not events:
            return np.zeros((1, EVENT_DIM), dtype=np.float32)

        vectors = []
        for idx, ev in enumerate(events):
            vectors.append(ev.to_vector(events[:idx]))
        return np.stack(vectors)
