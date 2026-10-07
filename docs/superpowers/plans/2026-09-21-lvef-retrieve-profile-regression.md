# LVEF 과제 구현 계획: Retrieve, Profile, 회귀 출력

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MI-CDM에서 ECG와 LVEF 라벨을 꺼내 4개 모델로 분류와 회귀를 같은 파이프라인에서 돌린다.

**Architecture:** 새 stage 2개(retrieve, profile)는 기존 `Stage` 계약을 그대로 따르고 파일로만 연결된다. 회귀는 새 run_key 축 없이 `label_def=value`로 구분하며, `label_value`와 `pred_value` 두 컬럼이 스키마에 추가된다. torch adapter는 카드의 `builder` 선언으로 모델을 만들고, 카드 `runtime.returns`로 module 반환 튜플의 의미를 읽는다.

**Tech Stack:** Python 3.9+ (core), pandas/pyarrow, psycopg 3, pydicom 3, torch, numpy. 서버 env `/data/mi-val/envs/mival`에 pydicom 3.0.2와 psycopg 3.3.4가 이미 있다.

**Spec:** `docs/superpowers/specs/2026-09-21-lvef-retrieve-profile-regression-design.md`

## Global Constraints

- `src/mival/` 식별자와 문자열에 모델명, 기관명, 데이터셋명을 쓰지 않는다 (`tests/test_no_proper_nouns.py`). 주석과 독스트링은 허용
- 문서에 em dash, en dash를 쓰지 않는다
- 백엔드(torch, tensorflow, pandas, psycopg, pydicom, sklearn)는 모듈 상단에서 import하지 않는다. 함수 안에서 import한다
- `tests/golden/expected.json`의 기존 스냅샷은 바뀌지 않아야 한다
- 코드 스타일: 함수 이름이 문서, docstring 한 줄, 주석은 "왜"만, pandas로 처리, 목표 분량 retrieve 120줄 / profile 80줄 / DICOM loader 40줄
- 커밋 메시지 끝에 다음 두 줄:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TQGri9wdZCYpXCeQDxvjTc
  ```
- 테스트 실행: `cd /Users/minseongkim/Desktop/youlab/mi-va && python -m pytest tests/<file> -q`. 백엔드 테스트(`torch`, `weights` 마커)는 서버 `/data/mi-val/mi-va`에서 해당 env로 돌린다

## 확정된 데이터 사실 (2026-09-21 실측)

- `cdm.image_occurrence`: `image_occurrence_id bigint, person_id bigint, local_path varchar, image_occurrence_date date` (datetime 컬럼 없음). 1,011,623행, 165,418명
- `cdm.measurement`: `measurement_date date, measurement_datetime timestamp, value_as_number numeric, measurement_concept_id int`
- `local_path` 예: `/home/ubuntu/dryou_mount/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/files/p1000/p10000032/s40689238/40689238.dcm`
- DICOM: SOP `12-lead ECG Waveform Storage`, `WaveformSequence` 1개, 12 채널, 5000 샘플, 500 Hz, 16-bit SS. `ds.waveform_array(0)`은 `(5000, 12)` float64 **mV** (sensitivity 0.005 적용됨). 채널 순서는 I, II, III, aVR, **aVF, aVL**, V1..V6 (aVF가 aVL보다 앞). `ChannelSourceSequence[0].CodeValue`는 MDC 코드 (`2:1`=I, `2:2`=II, `2:61`=III, `2:62`=aVR, `2:63`=aVL, `2:64`=aVF, `2:3`..`2:8`=V1..V6). 단위 `ChannelSensitivityUnitsSequence[0].CodeValue == "mV"`
- xECG forward는 `(cls, out)` 튜플을 낸다. `cls`가 `(B, 1024)` pooled representation
- HeartWise 회귀 module은 `(B, 1)` 텐서 하나를 낸다 (LVEF %). binary jit module도 `(B, 1)` logit 하나
- ECGFounder `Net1D(return_features=True)`는 `(logits, features)` 튜플

## 파일 구조

| 파일 | 책임 | 작업 |
|---|---|---|
| `src/mival/adapters/torch_adapter.py` | builder 선언으로 module 생성, `runtime.returns` 해석, 회귀 forward/finetune | 1, 4 |
| `registry/models/ecgfounder.json` | builder 선언 추가 | 1 |
| `src/mival/stages/retrieve.py`, `src/mival/stages/sql/retrieve_ecg_label.sql` | cohort_index 생성 | 2 |
| `src/mival/stages/profile.py` | cohort_split 동결, event-count gate | 3 |
| `src/mival/gates.py` | `EventCountError` | 3 |
| `src/mival/pipeline/stage.py` | retrieve, profile 등록 | 2, 3 |
| `src/mival/adapters/_training.py`, `base.py`, `keras_adapter.py` | `objective: mse`, `LinearHead.link` | 4 |
| `src/mival/stages/models.py` | `label_value`, `pred_value`, 회귀 arm | 5 |
| `src/mival/metrics/regression.py`, `metrics/__init__.py` | mae, rmse, r2, auroc_below | 6 |
| `src/mival/stages/_predictions.py`, `evaluate.py`, `misclassify.py`, `figures.py` | 회귀 arm 평가, 산점도, 6단계 건너뛰기 | 6 |
| `src/mival/ops.py`, `contract.py`, `compiler.py` | `gain` op | 7 |
| `src/mival/stages/preprocess.py` | `load_dicom_record`, loader 선택 | 7 |
| `registry/models/{xecg,heartwise-lvef-binary,heartwise-lvef-regression}.json`, `studies/lvef/study.yaml` | 카드와 study | 8 |
| `docs/decisions/adaptations.md`, `docs/plans/README.md`, `docs/operations/` | 기록 | 8, 9 |

## 병렬 운영 작업: DICOM 다운로드

코드 작업이 아니다. Task 7 전에 끝나 있어야 한다. 서버에는 S3 자격증명이 없고 노트북에만 있으므로, 사람이 다음 중 하나를 고른다.

- (a) 노트북에서 `aws sts get-session-token --duration-seconds 43200`으로 12시간 임시 토큰을 만들어 서버 셸 env에만 넣고 `aws s3 sync s3://<bucket>/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/files /scratch/mi-val/dicom/files --only-show-errors`를 `nohup`으로 돌린다. 끝나면 env를 지운다. 토큰은 파일에 남기지 않는다
- (b) 노트북에서 `aws s3 sync`로 외장 디스크에 받은 뒤 `rsync -e "ssh -p 2022 -i <key>.pem"`으로 서버 `/scratch/mi-val/dicom/files`에 올린다

어느 쪽이든 끝나면 `find /scratch/mi-val/dicom/files -name '*.dcm' | wc -l`이 1,011,623에 가까운지 확인하고 `docs/operations/dicom-cache.md`에 날짜, 파일 수, 용량, 방법을 적는다. `/scratch`는 인스턴스 stop 후 비워지므로 그 사실도 적는다.

---

### Task 1: torch adapter builder 선언 (A-4)

**Files:**
- Modify: `src/mival/adapters/torch_adapter.py:95-145` (`load`), `:52-70` (`_module_outputs`, `_module_features`)
- Modify: `registry/models/ecgfounder.json`
- Test: `tests/test_torch_builder.py` (신규), `tests/test_torch_adapter.py` (기존, weights 필요)

**Interfaces:**
- Consumes: `ModelCard.raw["x-mival"]` (`code_path`, `weights_format`, `builder`, `state_dict_key`, `rename_keys`), `card.output.weight_key`, `card.runtime.returns`
- Produces: `TorchAdapter.load(card) -> TorchHandle`, `_module_outputs(handle, tensor) -> (logits | None, features | None)`, `build_module(card) -> torch.nn.Module`

카드 선언 형식:

```json
"weights_format": "checkpoint",
"state_dict_key": "state_dict",
"builder": {"module": "net1d:Net1D", "kwargs": {"in_channels": 12, "...": "..."}},
"rename_keys": {"classifier.fc1.": "classifier.3."},
"output": {"type": "logits", "n_outputs": 150, "positive_index": null, "weight_key": "dense.weight"},
"runtime": {"returns": ["logits", "features"]}
```

`weights_format`: `state_dict` (파일이 곧 state_dict), `checkpoint` (dict 안의 `state_dict_key`), `safetensors`, `jit` (builder 불필요). `runtime.returns`는 module 반환값의 각 자리 이름이며 `logits`, `features`, `ignore` 중 하나. 텐서 하나를 반환하면 첫 이름이 적용된다. 기본값 `["logits", "features"]`는 지금 동작과 같다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_torch_builder.py`:

```python
"""Builder declarations on the ModelCard replace the hardcoded module constructor."""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.torch

from mival.adapters import get_adapter  # noqa: E402
from mival.modelcard import load_card  # noqa: E402

TINY_MODULE = '''
import torch

class Tiny(torch.nn.Module):
    def __init__(self, n_in, n_out, with_features=True):
        super().__init__()
        self.dense = torch.nn.Linear(n_in, n_out)
        self.with_features = with_features
    def forward(self, x):
        flat = x.flatten(1)
        out = self.dense(flat)
        return (out, flat) if self.with_features else out
'''


def write_code(tmp_path):
    (tmp_path / "tinymod.py").write_text(TINY_MODULE)
    return str(tmp_path)


def write_card(tmp_path, weights, weights_format, builder=None, extra=None, returns=None):
    body = {
        "model_id": "tiny",
        "Name": "tiny",
        "x-mival": {
            "adapter": "torch",
            "weights": [{"uri": str(weights), "sha256": "0" * 64, "role": "backbone"}],
            "weights_format": weights_format,
            "code_path": write_code(tmp_path),
            "input_contract": {
                "leads": ["I", "II"], "sampling_rate_hz": 2, "duration_s": 1,
                "unit": "mV", "scaling": "none", "layout": "lead_time", "dtype": "float32",
            },
            "output": {"type": "logits", "n_outputs": 3, "positive_index": 0, "weight_key": "dense.weight"},
            "runtime": {"returns": returns or ["logits", "features"]},
            "training_modes_supported": ["inference_only"],
            "pretraining_corpora": [],
        },
    }
    if builder is not None:
        body["x-mival"]["builder"] = builder
    body["x-mival"].update(extra or {})
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(body))
    return load_card(path)


def batch():
    return np.ones((2, 2, 2), dtype=np.float32)


def test_state_dict_weights_are_built_from_the_declared_module(tmp_path):
    import torch
    torch.manual_seed(0)
    reference = torch.nn.Linear(4, 3)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}})
    handle = get_adapter("torch").load(card)
    assert torch.equal(handle.module.dense.weight.cpu(), reference.weight)


def test_checkpoint_weights_use_state_dict_key_and_rename_keys(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 3)
    torch.save({"epoch": 1, "state_dict": {"fc1.weight": reference.weight, "fc1.bias": reference.bias}},
               tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "checkpoint",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}},
                      extra={"state_dict_key": "state_dict", "rename_keys": {"fc1.": "dense."}})
    handle = get_adapter("torch").load(card)
    assert torch.equal(handle.module.dense.bias.cpu(), reference.bias)


def test_jit_weights_need_no_builder(tmp_path):
    import torch
    import sys
    sys.path.insert(0, write_code(tmp_path))
    from tinymod import Tiny
    scripted = torch.jit.script(Tiny(4, 3, with_features=False))
    scripted.save(str(tmp_path / "w.pt"))
    card = write_card(tmp_path, tmp_path / "w.pt", "jit", returns=["logits"])
    handle = get_adapter("torch").load(card)
    probs = get_adapter("torch").forward(handle, batch())
    assert probs.shape == (2,)


def test_returns_declaration_names_each_output(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 3)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}},
                      returns=["ignore", "features"])
    adapter = get_adapter("torch")
    features = adapter.features(adapter.load(card), batch())
    assert features.shape == (2, 4)


def test_n_outputs_mismatch_is_rejected(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 5)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 5}})
    with pytest.raises(ValueError, match="declares 3 outputs"):
        get_adapter("torch").load(card)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_torch_builder.py -q` (torch가 있는 env에서)
Expected: FAIL, `net1d` import 오류 또는 `KeyError: 'code_path'`

- [ ] **Step 3: `load`와 `_module_outputs` 구현**

`src/mival/adapters/torch_adapter.py`의 `_module_outputs`, `_module_features`를 다음으로 교체한다. 호출부(`features`, `_published_probabilities`, `_probability_tensor`, `_run_finetune`, `_feature_width`, `_probabilities`)는 `_module_outputs(handle.module, tensor)` 대신 `_module_outputs(handle, tensor)`, `_module_features(handle.module, tensor)` 대신 `_module_features(handle, tensor)`로 바꾼다. `_run_finetune`와 `_feature_width`, `_probabilities`는 deepcopy한 `module`을 쓰므로 `_module_features(handle, tensor, module=module)` 형태로 module을 넘긴다.

```python
DEFAULT_RETURNS = ("logits", "features")


def _module_outputs(handle: "TorchHandle", tensor: Any, module: Any = None) -> Tuple[Optional[Any], Optional[Any]]:
    """``(logits, features)`` read off the module's return value by the card's ``runtime.returns``."""
    output = (module or handle.module)(tensor)
    outputs = output if isinstance(output, tuple) else (output,)
    names = tuple(handle.card.runtime.get("returns", DEFAULT_RETURNS))
    if len(outputs) > len(names):
        raise TrainingError(
            f"{handle.card.model_id!r} returned {len(outputs)} tensors but runtime.returns "
            f"names {len(names)}: {list(names)}"
        )
    named = dict(zip(names, outputs))
    return named.get("logits"), named.get("features")


def _module_features(handle: "TorchHandle", tensor: Any, module: Any = None) -> Any:
    features = _module_outputs(handle, tensor, module)[1]
    if features is None:
        raise TrainingError(
            f"{handle.card.model_id!r} names no 'features' in runtime.returns; "
            "there is no representation to probe"
        )
    return features
```

`load`를 다음으로 교체한다 (`from net1d import Net1D`와 하드코딩된 생성자 인자를 삭제):

```python
    def load(self, card: ModelCard) -> TorchHandle:
        import torch

        module = build_module(card)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TorchHandle(module=module.to(device).eval(), device=device, card=card)


def build_module(card: ModelCard) -> Any:
    """Construct the module the card declares and load its weights into it."""
    import importlib
    import sys

    import torch

    ext = card.raw["x-mival"]
    uri = card.weights[0]["uri"]
    weights_format = ext.get("weights_format", "checkpoint")
    code_path = ext.get("code_path")
    if code_path and code_path not in sys.path:
        sys.path.insert(0, code_path)

    if weights_format == "jit":
        return torch.jit.load(uri, map_location="cpu")

    builder = ext.get("builder")
    if not builder:
        raise ValueError(f"{card.model_id!r}: weights_format {weights_format!r} needs x-mival.builder")
    module_name, _, attribute = builder["module"].partition(":")
    constructor = getattr(importlib.import_module(module_name), attribute)
    module = constructor(*builder.get("args", ()), **builder.get("kwargs", {}))

    state = _read_state_dict(uri, weights_format, ext.get("state_dict_key"))
    for old, new in ext.get("rename_keys", {}).items():
        state = {key.replace(old, new, 1) if key.startswith(old) else key: value for key, value in state.items()}
    _check_n_outputs(card, state)
    module.load_state_dict(state, strict=True)
    return module


def _read_state_dict(uri: str, weights_format: str, state_dict_key: Optional[str]) -> Dict[str, Any]:
    import torch

    if weights_format == "safetensors":
        from safetensors.torch import load_file

        return load_file(uri)
    # Pickle stays off. The two NumPy scalar types are what checkpoints of
    # this project have been seen to carry as metadata.
    safe_globals = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), (np.dtype, "numpy.dtype"), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe_globals):
        loaded = torch.load(uri, map_location="cpu", weights_only=True)
    if weights_format == "checkpoint":
        return loaded[state_dict_key or "state_dict"]
    if weights_format == "state_dict":
        return loaded
    raise ValueError(f"unknown weights_format {weights_format!r}; use state_dict, checkpoint, safetensors or jit")


def _check_n_outputs(card: ModelCard, state: Dict[str, Any]) -> None:
    """A card that disagrees with its own checkpoint would silently score the wrong head."""
    key = card.output.get("weight_key")
    declared = card.output.get("n_outputs")
    if key is None or declared is None:
        return
    actual = int(state[key].shape[0])
    if actual != int(declared):
        raise ValueError(f"card declares {declared} outputs but the checkpoint has {actual}")
```

`_positive_probability`는 `handle.card.output["positive_index"]`를 쓰므로 그대로 둔다.

- [ ] **Step 4: ECGFounder 카드에 builder 선언 추가**

`registry/models/ecgfounder.json`의 `x-mival`에 추가:

```json
"weights_format": "checkpoint",
"state_dict_key": "state_dict",
"builder": {
  "module": "net1d:Net1D",
  "kwargs": {"in_channels": 12, "base_filters": 64, "ratio": 1,
             "filter_list": [64, 160, 160, 400, 400, 1024, 1024],
             "m_blocks_list": [2, 2, 2, 3, 3, 4, 4], "kernel_size": 16, "stride": 2,
             "groups_width": 16, "n_classes": 150, "use_bn": false, "use_do": false,
             "return_features": true, "verbose": false}
},
```

`output`에 `"weight_key": "dense.weight"`를, `runtime`에 `"returns": ["logits", "features"]`를 추가한다. `configs/aws/*.yaml`은 손대지 않는다.

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest tests/test_torch_builder.py tests/test_torch_adapter.py tests/test_torch_adapter_fit.py tests/test_torch_attribution.py tests/test_modelcard.py tests/test_no_proper_nouns.py -q`
Expected: 전부 PASS. `test_torch_adapter.py`는 ECGFounder weights가 있는 서버에서 돌리고, `features.shape == (2, 1024)`가 그대로 통과해야 한다.

- [ ] **Step 6: 커밋**

```bash
git add src/mival/adapters/torch_adapter.py registry/models/ecgfounder.json tests/test_torch_builder.py
git commit -m "feat: torch adapter builds the module the ModelCard declares (A-4)"
```

---

### Task 2: Retrieve stage

**Files:**
- Create: `src/mival/stages/retrieve.py`, `src/mival/stages/sql/retrieve_ecg_label.sql`
- Modify: `src/mival/pipeline/stage.py:33-46` (`_STAGES`, `_UNBUILT`), `pyproject.toml` (optional extra `db`), `setuptools` package data
- Test: `tests/test_stage_retrieve.py`

**Interfaces:**
- Consumes: `Stage`, `StageContext`, `StageResult`, `ExclusionLedger.record`, `write_table`
- Produces: `cohort_index.parquet` with `COHORT_INDEX_COLUMNS = ("image_occurrence_id", "person_id", "local_path", "index_datetime", "label_datetime", "label_delta_days", "label_value", "label_primary", "label_sens1", "label_sens2", "label_sens3")`; `RetrieveStage(query=None)` where `query(sql: str, params: Mapping) -> pandas.DataFrame`; `resolve_local_path(cdm_path, root) -> str | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage_retrieve.py`:

```python
"""Retrieve stage (spec §4.1) against an injected query, so no database is needed."""

from datetime import date, datetime

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from mival.pipeline.ledger import read_exclusions  # noqa: E402
from mival.pipeline.layout import exclusions_root  # noqa: E402
from mival.pipeline.stage import execute, get_stage, prepare  # noqa: E402
from mival.pipeline.tables import read_table  # noqa: E402
from mival.stages.retrieve import (  # noqa: E402
    COHORT_INDEX,
    COHORT_INDEX_COLUMNS,
    SUMMARY,
    RetrieveStage,
    resolve_local_path,
)

CDM_PREFIX = "/elsewhere/mimic-iv-ecg-dcm/files/p1000/p10000032"


def row(image_id, person, day, value, delta, path=None):
    return {
        "image_occurrence_id": image_id,
        "person_id": person,
        "local_path": path or f"{CDM_PREFIX}/s{image_id}/{image_id}.dcm",
        "index_datetime": date(2180, 7, day),
        "label_datetime": None if value is None else datetime(2180, 7, day + delta, 9, 0),
        "label_delta_days": None if value is None else delta,
        "label_value": value,
    }


def fake_query(rows_by_window):
    """Return the frame for the window the SQL was run with."""
    calls = []

    def query(sql, params):
        calls.append((sql, dict(params)))
        return pd.DataFrame(rows_by_window[params["window_days"]])

    query.calls = calls
    return query


def make_spec(tmp_path, **overrides):
    spec = {
        "dsn_env": "/nonexistent/micdm.env",
        "schema": "cdm",
        "label_concept_id": 3027172,
        "window_days": 7,
        "window_days_sens2": 30,
        "implausible_below": 5,
        "local_path_root": str(tmp_path / "dicom"),
        "require_local_file": False,
    }
    spec.update(overrides)
    return spec


def run(tmp_path, query, **overrides):
    stage = RetrieveStage(query=query)
    ctx = prepare(stage, "s", "site-a", make_spec(tmp_path, **overrides), {}, tmp_path / "runs", seed=1)
    execute(stage, ctx)
    return ctx, read_table(ctx.layout.artifact(COHORT_INDEX))


def test_local_path_is_rerooted_under_the_site_root():
    cdm = f"{CDM_PREFIX}/s1/1.dcm"
    assert resolve_local_path(cdm, "/scratch/dicom") == "/scratch/dicom/files/p1000/p10000032/s1/1.dcm"
    assert resolve_local_path(cdm, "/other/root") == "/other/root/files/p1000/p10000032/s1/1.dcm"
    assert resolve_local_path("/no/files/segment.dcm", "/scratch/dicom") is None


def test_labels_are_thresholded_and_sens2_uses_the_wide_window(tmp_path):
    narrow = [row(1, 10, 1, 35.0, 2), row(2, 11, 1, 55.0, -3), row(3, 12, 1, None, None)]
    wide = [row(1, 10, 1, 35.0, 2), row(2, 11, 1, 55.0, -3), row(3, 12, 1, 45.0, 20)]
    _, index = run(tmp_path, fake_query({7: narrow, 30: wide}))
    assert list(index.columns) == list(COHORT_INDEX_COLUMNS)
    assert index["image_occurrence_id"].tolist() == [1, 2]
    assert index["label_primary"].tolist() == [1, 0]
    assert index["label_sens1"].tolist() == [1, 0]
    assert index["label_sens2"].tolist() == [1, 0]
    assert index["label_sens3"].isna().all()
    assert index["label_value"].tolist() == [35.0, 55.0]


def test_exclusions_are_recorded_in_order(tmp_path):
    rows = [row(1, 10, 1, None, None), row(2, 11, 1, 0.0, 0), row(3, 12, 1, 50.0, 1, path="/no/segment.dcm"), row(4, 13, 1, 50.0, 1)]
    ctx, index = run(tmp_path, fake_query({7: rows, 30: rows}))
    ledger = read_exclusions(exclusions_root(tmp_path / "runs", "s"))
    assert dict(zip(ledger["image_occurrence_id"], ledger["reason_code"])) == {
        "1": "label_missing", "2": "label_implausible", "3": "path_unresolvable"}
    assert index["image_occurrence_id"].tolist() == [4]


def test_missing_files_are_excluded_only_when_required(tmp_path):
    rows = [row(1, 10, 1, 30.0, 0), row(2, 11, 1, 30.0, 0)]
    present = tmp_path / "dicom" / "files" / "p1000" / "p10000032" / "s1" / "1.dcm"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"")
    _, index = run(tmp_path, fake_query({7: rows, 30: rows}), require_local_file=True)
    assert index["image_occurrence_id"].tolist() == [1]
    _, index = run(tmp_path, fake_query({7: rows, 30: rows}), require_local_file=False)
    assert len(index) == 2


def test_no_labels_at_all_fails_loudly(tmp_path):
    rows = [row(1, 10, 1, None, None)]
    with pytest.raises(ValueError, match="no label"):
        run(tmp_path, fake_query({7: rows, 30: rows}))


def test_sql_is_copied_and_summary_written(tmp_path):
    rows = [row(1, 10, 1, 30.0, 0)]
    query = fake_query({7: rows, 30: rows})
    ctx, _ = run(tmp_path, query)
    assert query.calls[0][1]["window_days"] == 7 and query.calls[1][1]["window_days"] == 30
    assert "cdm.image_occurrence" in query.calls[0][0]
    assert (ctx.layout.artifact("sql", "window_7.sql")).is_file()
    summary = ctx.layout.artifact(SUMMARY)
    assert summary.is_file() and '"n_out": 1' in summary.read_text()


def test_stage_is_registered():
    assert get_stage("retrieve").name == "retrieve"
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_stage_retrieve.py -q`
Expected: FAIL, `ModuleNotFoundError: mival.stages.retrieve`

- [ ] **Step 3: SQL 파일 작성**

`src/mival/stages/sql/retrieve_ecg_label.sql`:

```sql
-- Every ECG with the nearest label within the window, or nulls when none.
-- Ties on day distance go to the earlier measurement. {schema} is filled by
-- the stage after an identifier check; the other values are bound parameters.
select
    i.image_occurrence_id,
    i.person_id,
    i.local_path,
    i.image_occurrence_date                          as index_datetime,
    m.measurement_datetime                           as label_datetime,
    m.measurement_date - i.image_occurrence_date     as label_delta_days,
    m.value_as_number                                as label_value
from {schema}.image_occurrence i
left join lateral (
    select m.measurement_datetime, m.measurement_date, m.value_as_number
    from {schema}.measurement m
    where m.person_id = i.person_id
      and m.measurement_concept_id = %(label_concept_id)s
      and m.value_as_number is not null
      and abs(m.measurement_date - i.image_occurrence_date) <= %(window_days)s
    order by abs(m.measurement_date - i.image_occurrence_date), m.measurement_datetime
    limit 1
) m on true
order by i.image_occurrence_id
```

- [ ] **Step 4: retrieve.py 작성**

```python
"""Retrieve stage (spec §4.1): every ECG in the CDM with its nearest label."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import write_table

COHORT_INDEX = "cohort_index.parquet"
SUMMARY = "retrieve_summary.json"
SQL_DIR = "sql"
SQL_FILE = Path(__file__).with_name("sql") / "retrieve_ecg_label.sql"

COHORT_INDEX_COLUMNS = (
    "image_occurrence_id", "person_id", "local_path", "index_datetime",
    "label_datetime", "label_delta_days", "label_value",
    "label_primary", "label_sens1", "label_sens2", "label_sens3",
)
REASON_CODES = frozenset({"label_missing", "label_implausible", "path_unresolvable", "file_missing"})

Query = Callable[[str, Mapping[str, Any]], Any]
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class RetrieveSpec:
    dsn_env: str
    schema: str
    label_concept_id: int
    window_days: int
    window_days_sens2: int
    implausible_below: float
    local_path_root: str
    require_local_file: bool
    primary_cutoff: float = 40.0   # label_primary = value <= primary_cutoff
    sens1_cutoff: float = 50.0     # label_sens1 = value < sens1_cutoff

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "RetrieveSpec":
        for key in ("dsn_env", "local_path_root"):
            if not body.get(key):
                raise ValueError(f"retrieve.{key} is required")
        schema = str(body.get("schema", "cdm"))
        if not _IDENTIFIER.match(schema):
            raise ValueError(f"retrieve.schema {schema!r} is not a plain identifier")
        return cls(
            dsn_env=str(body["dsn_env"]),
            schema=schema,
            label_concept_id=int(body.get("label_concept_id", 3027172)),
            window_days=int(body.get("window_days", 7)),
            window_days_sens2=int(body.get("window_days_sens2", 30)),
            implausible_below=float(body.get("implausible_below", 5)),
            local_path_root=str(body["local_path_root"]),
            require_local_file=bool(body.get("require_local_file", True)),
            primary_cutoff=float(body.get("primary_cutoff", 40)),
            sens1_cutoff=float(body.get("sens1_cutoff", 50)),
        )

    def params(self, window_days: int) -> dict:
        return {"label_concept_id": self.label_concept_id, "window_days": window_days}


class RetrieveStage(Stage):
    name = "retrieve"
    reason_codes = REASON_CODES

    def __init__(self, query: Optional[Query] = None) -> None:
        self._query = query  # injected in tests; None means PostgreSQL via dsn_env

    def run(self, ctx: StageContext) -> StageResult:
        spec = RetrieveSpec.from_mapping(ctx.spec or {})
        query = self._query or (lambda sql, params: query_postgres(sql, params, spec.dsn_env))
        sql = SQL_FILE.read_text(encoding="utf-8").format(schema=spec.schema)

        narrow = query(sql, spec.params(spec.window_days))
        wide = query(sql, spec.params(spec.window_days_sens2))
        if narrow["label_value"].notna().sum() == 0:
            raise ValueError(
                f"no label within {spec.window_days} days for any ECG; check retrieve.label_concept_id"
            )
        frame = narrow.merge(
            wide[["image_occurrence_id", "label_value"]].rename(columns={"label_value": "label_value_sens2"}),
            on="image_occurrence_id", how="left",
        )
        frame["local_path"] = frame["local_path"].map(lambda p: resolve_local_path(p, spec.local_path_root))

        kept = exclude(frame, spec, ctx.ledger)
        kept = attach_labels(kept, spec)
        index_path = write_table(kept[list(COHORT_INDEX_COLUMNS)].to_dict("records"),
                                 ctx.layout.artifact(COHORT_INDEX), COHORT_INDEX_COLUMNS)
        sql_paths = [write_sql(ctx, sql, spec.params(days), days) for days in (spec.window_days, spec.window_days_sens2)]
        summary_path = write_summary(ctx, frame, kept)
        return StageResult(outputs=[index_path, summary_path, *sql_paths],
                           counts={"in": len(frame), "out": len(kept)})


def resolve_local_path(cdm_path: Any, root: str) -> Optional[str]:
    """Re-root the part after 'files/' under this site's DICOM root (ledger A-1)."""
    _, sep, tail = str(cdm_path).partition("/files/")
    return str(Path(root) / "files" / tail) if sep else None


def exclude(frame, spec: RetrieveSpec, ledger):
    """Drop rows in the spec's order, so each record is excluded for one reason only."""
    def drop(frame, mask, code, detail):
        for row in frame[mask].itertuples(index=False):
            ledger.record(row.image_occurrence_id, row.person_id, code, detail(row))
        return frame[~mask]

    frame = drop(frame, frame["label_value"].isna(), "label_missing",
                 lambda r: f"no label within {spec.window_days} days")
    frame = drop(frame, frame["label_value"] <= spec.implausible_below, "label_implausible",
                 lambda r: f"label_value={r.label_value}")
    frame = drop(frame, frame["local_path"].isna(), "path_unresolvable",
                 lambda r: "no 'files/' segment in the CDM local_path")
    if spec.require_local_file:
        frame = drop(frame, ~frame["local_path"].map(lambda p: Path(p).is_file()), "file_missing",
                     lambda r: r.local_path)
    return frame


def attach_labels(frame, spec: RetrieveSpec):
    import numpy as np

    value, wide = frame["label_value"], frame["label_value_sens2"]
    # Nullable labels are float with NaN: parquet round-trips that, and the
    # models stage already reads NaN as "no label".
    return frame.assign(
        label_primary=(value <= spec.primary_cutoff).astype("int64"),
        label_sens1=(value < spec.sens1_cutoff).astype("int64"),
        label_sens2=(wide <= spec.primary_cutoff).astype("float64").where(wide.notna()),
        label_sens3=np.nan,
    )


def write_sql(ctx: StageContext, sql: str, params: Mapping[str, Any], days: int) -> Path:
    filled = sql
    for key, value in params.items():
        filled = filled.replace(f"%({key})s", str(value))
    path = ctx.layout.artifact(SQL_DIR, f"window_{days}.sql")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(filled, encoding="utf-8")
    return path


def write_summary(ctx: StageContext, frame, kept) -> Path:
    quantiles = kept["label_value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    payload = {
        "n_ecg": int(len(frame)),
        "n_labelled": int(frame["label_value"].notna().sum()),
        "n_out": int(len(kept)),
        "n_persons_out": int(kept["person_id"].nunique()),
        "excluded": ctx.ledger.counts(),
        "positives": {name: int(kept[name].sum()) for name in ("label_primary", "label_sens1", "label_sens2")},
        "label_value_quantiles": {f"p{int(q * 100)}": float(v) for q, v in quantiles.items()},
        "label_delta_days": {str(k): int(v) for k, v in kept["label_delta_days"].value_counts().sort_index().items()},
    }
    path = ctx.layout.artifact(SUMMARY)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def query_postgres(sql: str, params: Mapping[str, Any], dsn_env: str):
    """Run one query with libpq settings read from an env file; nothing is logged."""
    import pandas
    import psycopg

    for line in Path(dsn_env).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().startswith("PG"):
            os.environ[key.strip()] = value.strip().strip("'\"")
    with psycopg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [column.name for column in cursor.description]
        return pandas.DataFrame(cursor.fetchall(), columns=columns)
```

`src/mival/stages/sql/__init__.py`는 만들지 않는다. 대신 `pyproject.toml`에 SQL을 패키지 데이터로 포함한다:

```toml
[tool.setuptools.package-data]
mival = ["stages/sql/*.sql"]
```

그리고 optional extra 추가:

```toml
db = ["psycopg[binary]>=3.1"]
```

- [ ] **Step 5: stage 등록**

`src/mival/pipeline/stage.py`에서 `_STAGES`에 `"retrieve": "mival.stages.retrieve:RetrieveStage"`를 맨 앞에 넣고, `_UNBUILT`에서 `retrieve` 항목을 지운다.

- [ ] **Step 6: 테스트 통과 확인**

Run: `python -m pytest tests/test_stage_retrieve.py tests/test_stage.py tests/test_study_and_cli.py tests/test_no_proper_nouns.py -q`
Expected: PASS. `test_stage.py`에 `_UNBUILT`를 검사하는 테스트가 있으면 `retrieve`가 빠진 것에 맞춰 고친다.

- [ ] **Step 7: 커밋**

```bash
git add src/mival/stages/retrieve.py src/mival/stages/sql/retrieve_ecg_label.sql src/mival/pipeline/stage.py pyproject.toml tests/test_stage_retrieve.py
git commit -m "feat: retrieve stage builds cohort_index from the CDM with a nearest label"
```

---

### Task 3: Profile stage

**Files:**
- Create: `src/mival/stages/profile.py`
- Modify: `src/mival/gates.py` (`EventCountError`), `src/mival/pipeline/stage.py` (등록)
- Test: `tests/test_stage_profile.py`

**Interfaces:**
- Consumes: `cohort_index.parquet` (Task 2 컬럼)
- Produces: `cohort_split.parquet` (`person_id`, `split`, `fold`), `profile_summary.json`; `assign_splits(persons, stratify_on, test_fraction, n_folds, seed) -> DataFrame`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage_profile.py`:

```python
"""Profile stage (spec §4.2): a frozen person-level split and the event-count gate."""

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from mival.gates import EventCountError  # noqa: E402
from mival.pipeline.stage import execute, get_stage, prepare  # noqa: E402
from mival.pipeline.tables import read_table  # noqa: E402
from mival.stages.profile import COHORT_SPLIT, SUMMARY, ProfileStage, assign_splits  # noqa: E402


def cohort(n_persons=100, ecgs_per_person=2, positive_every=4):
    rows = []
    for person in range(n_persons):
        for recording in range(ecgs_per_person):
            value = 30.0 if person % positive_every == 0 else 60.0
            rows.append({
                "image_occurrence_id": person * 10 + recording, "person_id": person,
                "local_path": f"/x/{person}_{recording}.dcm", "index_datetime": None,
                "label_datetime": None, "label_delta_days": 0, "label_value": value,
                "label_primary": int(value <= 40), "label_sens1": int(value < 50),
                "label_sens2": int(value <= 40), "label_sens3": None,
            })
    return pd.DataFrame(rows)


def run(tmp_path, frame, spec=None, seed=7):
    index = tmp_path / "cohort_index.parquet"
    frame.to_parquet(index, index=False)
    stage = ProfileStage()
    ctx = prepare(stage, "s", "site-a", spec or {"min_test_positives": 5}, {"cohort_index": index}, tmp_path / "runs", seed=seed)
    execute(stage, ctx)
    return ctx, read_table(ctx.layout.artifact(COHORT_SPLIT))


def test_split_is_person_level_and_stratified(tmp_path):
    frame = cohort()
    _, split = run(tmp_path, frame)
    assert set(split.columns) == {"person_id", "split", "fold"}
    assert split["person_id"].is_unique and len(split) == 100
    test = split[split["split"] == "test"]
    assert len(test) == 20
    positives = frame.groupby("person_id")["label_primary"].max()
    assert positives[test["person_id"]].sum() == 5          # 25 positive persons, 20% of them
    assert split[split["split"] == "dev"]["fold"].nunique() == 5
    assert split[split["split"] == "test"]["fold"].isna().all()


def test_same_seed_reproduces_and_another_seed_differs(tmp_path):
    a = assign_splits(pd.DataFrame({"person_id": range(50), "label_primary": [i % 3 == 0 for i in range(50)]}), "label_primary", 0.2, 5, seed=1)
    b = assign_splits(pd.DataFrame({"person_id": range(50), "label_primary": [i % 3 == 0 for i in range(50)]}), "label_primary", 0.2, 5, seed=1)
    c = assign_splits(pd.DataFrame({"person_id": range(50), "label_primary": [i % 3 == 0 for i in range(50)]}), "label_primary", 0.2, 5, seed=2)
    assert a.equals(b) and not a.equals(c)


def test_event_count_gate_fails_the_run(tmp_path):
    with pytest.raises(EventCountError, match="test split holds"):
        run(tmp_path, cohort(), spec={"min_test_positives": 1000})


def test_summary_reports_counts(tmp_path):
    ctx, _ = run(tmp_path, cohort())
    text = ctx.layout.artifact(SUMMARY).read_text()
    assert '"test"' in text and '"n_positive_ecgs"' in text


def test_stage_is_registered():
    assert get_stage("profile").name == "profile"
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_stage_profile.py -q`
Expected: FAIL, `ImportError: EventCountError` 또는 `ModuleNotFoundError`

- [ ] **Step 3: gate와 stage 구현**

`src/mival/gates.py` 끝에 추가:

```python
class EventCountError(RuntimeError):
    """Spec §2.4 gate 2: too few events in the held-out split to report anything."""
```

`src/mival/stages/profile.py`:

```python
"""Profile stage (spec §4.2): freeze the person-level split and check the event count."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mival.gates import EventCountError
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table

COHORT_SPLIT = "cohort_split.parquet"
SUMMARY = "profile_summary.json"
COHORT_SPLIT_COLUMNS = ("person_id", "split", "fold")


@dataclass(frozen=True)
class ProfileSpec:
    test_fraction: float
    n_folds: int
    stratify_on: str
    min_test_positives: int

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "ProfileSpec":
        return cls(
            test_fraction=float(body.get("test_fraction", 0.2)),
            n_folds=int(body.get("n_folds", 5)),
            stratify_on=str(body.get("stratify_on", "label_primary")),
            min_test_positives=int(body.get("min_test_positives", 100)),
        )


class ProfileStage(Stage):
    name = "profile"
    reason_codes = frozenset()  # profile describes; it never drops a record

    def required_inputs(self) -> tuple:
        return ("cohort_index",)

    def run(self, ctx: StageContext) -> StageResult:
        spec = ProfileSpec.from_mapping(ctx.spec or {})
        cohort = read_table(ctx.input_path("cohort_index"))
        # A person is positive if any of their ECGs is, so the split is stratified on people.
        persons = cohort.groupby("person_id")[spec.stratify_on].max().reset_index()
        split = assign_splits(persons, spec.stratify_on, spec.test_fraction, spec.n_folds, ctx.seed)

        test_positives = int(cohort[cohort["person_id"].isin(split.loc[split["split"] == "test", "person_id"])][spec.stratify_on].sum())
        if test_positives < spec.min_test_positives:
            raise EventCountError(
                f"test split holds {test_positives} positive ECGs; profile.min_test_positives is {spec.min_test_positives}"
            )

        split_path = write_table(split.to_dict("records"), ctx.layout.artifact(COHORT_SPLIT), COHORT_SPLIT_COLUMNS)
        summary_path = write_summary(ctx, cohort, split, spec, test_positives)
        return StageResult(outputs=[split_path, summary_path], counts={"in": len(cohort), "out": len(split)})


def assign_splits(persons, stratify_on: str, test_fraction: float, n_folds: int, seed):
    """Stratified person-level test split, then stratified folds over dev. Same seed, same split."""
    import pandas

    rng = np.random.default_rng(seed)
    parts = []
    for _, stratum in persons.sort_values("person_id").groupby(stratify_on):
        order = stratum.iloc[rng.permutation(len(stratum))]
        n_test = int(round(len(order) * test_fraction))
        n_dev = len(order) - n_test
        parts.append(pandas.DataFrame({
            "person_id": order["person_id"].to_numpy(),
            "split": ["test"] * n_test + ["dev"] * n_dev,
            # NaN rather than a nullable integer: parquet round-trips it and
            # the models stage reads NaN as "not cross-validated".
            "fold": [np.nan] * n_test + list((np.arange(n_dev) % n_folds).astype(float)),
        }))
    return pandas.concat(parts, ignore_index=True)


def write_summary(ctx: StageContext, cohort, split, spec: ProfileSpec, test_positives: int) -> Path:
    joined = cohort.merge(split, on="person_id")
    by_split = {
        name: {
            "n_persons": int(group["person_id"].nunique()),
            "n_ecgs": int(len(group)),
            "n_positive_ecgs": int(group[spec.stratify_on].sum()),
        }
        for name, group in joined.groupby("split")
    }
    payload = {
        "splits": by_split,
        "gate": {"min_test_positives": spec.min_test_positives, "test_positives": test_positives, "passed": True},
        "label_value_quantiles": {f"p{int(q * 100)}": float(v) for q, v in cohort["label_value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()},
        "label_delta_days": {str(k): int(v) for k, v in cohort["label_delta_days"].value_counts().sort_index().items()},
        "ecgs_per_person": {str(k): int(v) for k, v in cohort.groupby("person_id").size().value_counts().sort_index().items()},
    }
    path = ctx.layout.artifact(SUMMARY)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
```

`src/mival/pipeline/stage.py`: `_STAGES`에 `"profile": "mival.stages.profile:ProfileStage"`를 retrieve 다음에 넣고 `_UNBUILT`를 빈 dict로 둔다 (변수와 `get_stage`의 검사는 남긴다).

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_stage_profile.py tests/test_stage.py tests/test_gates.py -q`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/mival/stages/profile.py src/mival/gates.py src/mival/pipeline/stage.py tests/test_stage_profile.py
git commit -m "feat: profile stage freezes the person-level split and gates on event count"
```

---

### Task 4: adapter 회귀 (objective mse, LinearHead link, regression forward)

**Files:**
- Modify: `src/mival/adapters/_training.py` (`FitSplit`, `FitData`, `parse_fit_data`, `_split`, `LinearHead`, `fit_linear_head`, `validation_score`, `parse_hparams`)
- Modify: `src/mival/adapters/base.py` (`fit`, `_fit_linear_probe`)
- Modify: `src/mival/adapters/torch_adapter.py` (`forward`, `_published_probabilities`, `_positive_probability`, `_run_finetune`, `_probabilities`)
- Modify: `src/mival/adapters/keras_adapter.py` (`forward`)
- Test: `tests/test_adapter_training.py`, `tests/test_torch_adapter_fit.py`, `tests/test_keras_adapter.py`

**Interfaces:**
- Consumes: fit payload from Task 5 with key `"objective": "bce" | "mse"` (default `"bce"`)
- Produces: `FitData.objective`, `LinearHead.link in ("logistic", "identity")`, `LinearHead.predict(features)`, `fit_linear_head(..., objective="bce")`, `validation_score(pred, labels, metric, context, objective="bce")`; `Adapter.forward` returns LVEF values for a card with `output.type == "regression"`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_adapter_training.py` 끝에 추가:

```python
def make_regression_data(n=200, seed=0, n_validation=60, batch_size=16):
    """y = 3*x0 - 2*x1 + 50 + noise, so a linear head must recover the coefficients."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, (n, 2))
    y = 3.0 * signal[:, 0] - 2.0 * signal[:, 1] + 50.0 + rng.normal(0, 0.1, n)
    store = {f"rec-{i}": signal[i] for i in range(n)}

    def load_batch(paths):
        return np.stack([np.repeat(store[p][:, None], 4, axis=1) for p in paths])

    split = n - n_validation
    return {
        "tensor_paths": [f"rec-{i}" for i in range(split)], "y": y[:split],
        "person_id": [f"p-{i}" for i in range(split)], "fold": [0] * split,
        "label_column": "label_value", "objective": "mse",
        "load_batch": load_batch, "batch_size": batch_size,
        "validation": {"tensor_paths": [f"rec-{i}" for i in range(split, n)], "y": y[split:],
                       "person_id": [f"p-{i}" for i in range(split, n)]},
    }


def test_mse_objective_accepts_continuous_labels():
    data = parse_fit_data(make_regression_data())
    assert data.objective == "mse"
    assert data.train.y.dtype == np.float64


def test_bce_objective_still_rejects_continuous_labels():
    body = make_regression_data()
    body["objective"] = "bce"
    with pytest.raises(TrainingError, match="labels must be 0 or 1"):
        parse_fit_data(body)


def test_linear_probe_with_mse_recovers_the_coefficients():
    adapter = FakeAdapter()
    fitted = adapter.fit("handle", make_regression_data(), "linear_probe", {"epochs": 2000, "lr": 0.05, "patience": 200})
    head = fitted["head"]
    assert head.link == "identity"
    x = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    predicted = head.predict(x)
    assert np.allclose(predicted, [53.0, 48.0, 50.0], atol=0.5)
    assert fitted["fit_record"]["objective"] == "mse"
    assert fitted["fit_record"]["validation_metric"] == "mae"


def test_validation_score_for_mse_is_negative_mae():
    score = validation_score(np.array([1.0, 3.0]), np.array([2.0, 2.0]), "auroc", "t", objective="mse")
    assert score.primary == -1.0
```

`tests/test_torch_adapter_fit.py`에 추가 (기존 fixture의 tiny module 방식을 따른다; 파일 상단의 헬퍼로 module과 card를 만드는 함수가 있으면 그대로 쓴다):

```python
def test_full_finetune_with_mse_uses_identity_head(tmp_path):
    """The fine-tuned head predicts values, not probabilities."""
    import torch
    from mival.modelcard import load_card

    (tmp_path / "tinymod.py").write_text(
        "import torch\n"
        "class Tiny(torch.nn.Module):\n"
        "    def __init__(self, n_in, n_out):\n"
        "        super().__init__(); self.dense = torch.nn.Linear(n_in, n_out)\n"
        "    def forward(self, x):\n"
        "        flat = x.flatten(1); return self.dense(flat), flat\n"
    )
    reference = torch.nn.Linear(4, 3)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card_body = {
        "model_id": "tiny", "Name": "tiny",
        "x-mival": {
            "adapter": "torch", "code_path": str(tmp_path), "weights_format": "state_dict",
            "weights": [{"uri": str(tmp_path / "w.pt"), "sha256": "0" * 64, "role": "backbone"}],
            "builder": {"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}},
            "input_contract": {"leads": ["I", "II"], "sampling_rate_hz": 2, "duration_s": 1, "unit": "mV",
                               "scaling": "none", "layout": "lead_time", "dtype": "float32"},
            "output": {"type": "logits", "n_outputs": 3, "positive_index": 0},
            "runtime": {"returns": ["logits", "features"]},
            "training_modes_supported": ["full_finetune"], "pretraining_corpora": [],
        },
    }
    (tmp_path / "tiny.json").write_text(json.dumps(card_body))
    card = load_card(tmp_path / "tiny.json")

    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(0, 1, (n, 2, 2)).astype(np.float32)          # flattens to 4 features
    y = 50.0 + 3.0 * x[:, 0, 0] - 2.0 * x[:, 1, 1] + rng.normal(0, 0.1, n)
    store = {f"r{i}": x[i] for i in range(n)}
    data = {
        "tensor_paths": [f"r{i}" for i in range(100)], "y": y[:100],
        "person_id": [f"p{i}" for i in range(100)], "fold": [0] * 100,
        "label_column": "label_value", "objective": "mse",
        "load_batch": lambda paths: np.stack([store[p] for p in paths]), "batch_size": 16,
        "validation": {"tensor_paths": [f"r{i}" for i in range(100, n)], "y": y[100:],
                       "person_id": [f"p{i}" for i in range(100, n)]},
    }
    adapter = get_adapter("torch")
    fitted = adapter.fit(adapter.load(card), data, "full_finetune", {"epochs": 3, "seed": 0, "lr": 0.01})
    assert fitted.head.link == "identity"
    out = adapter.forward(fitted, x[:5])
    assert out.shape == (5,) and np.all(out > 1.5)   # values near 50, not probabilities
```

(파일 상단에 `import json`이 없으면 추가한다.)

`tests/test_torch_builder.py`에 추가:

```python
def test_regression_card_forward_returns_the_value_column(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 1)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 1, "with_features": False}},
                      returns=["logits"])
    card.output.update({"type": "regression", "n_outputs": 1, "value_index": 0, "positive_index": None})
    adapter = get_adapter("torch")
    out = adapter.forward(adapter.load(card), batch())
    expected = reference(torch.ones(2, 4)).detach().numpy().ravel()
    assert np.allclose(out, expected, atol=1e-5)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_adapter_training.py -q`
Expected: FAIL, `parse_fit_data`가 `objective`를 모르고 float 라벨을 거부

- [ ] **Step 3: `_training.py` 수정**

`FitData`에 `objective: str = "bce"` 필드를 추가한다. `parse_fit_data`:

```python
    objective = str(data.get("objective", "bce"))
    if objective not in ("bce", "mse"):
        raise TrainingError(f"objective must be 'bce' or 'mse', got {objective!r}")
    train = _split(data, "train", objective)
    if not len(train):
        raise TrainingError("fit was given no training records")
    if objective == "bce" and (train.n_pos == 0 or train.n_neg == 0):
        raise TrainingError(...)  # 기존 메시지 그대로
    ...
        validation = _split(body, "validation", objective)
    return FitData(train=train, validation=validation, load_batch=loader, batch_size=batch_size,
                   label_column=str(data.get("label_column", "")), objective=objective)
```

`_split(body, where, objective="bce")`: `objective == "mse"`면 0/1 검사를 건너뛰고 `y=labels.astype(np.float64)`로 만든다. `bce`면 기존대로 int64.

`LinearHead`에 `link: str = "logistic"` 필드와 메서드 추가:

```python
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Values for an identity head, probabilities for a logistic one."""
        return self.logits(features) if self.link == "identity" else self.probabilities(features)
```

`LinearHead.of(weights, bias, link="logistic")`.

`fit_linear_head(features, labels, hp, validation=None, context="linear_probe", objective="bce")`:
- bias 초기값: `mse`면 `float(y.mean())`, 아니면 기존 log-odds
- `sample_weight`: `mse`면 전부 1.0
- 루프 안 예측: `mse`면 `p = Z @ weights + bias`, residual은 `sample_weight * (p - y)`. 같은 식이므로 `p` 계산만 분기한다
- 검증 점수: `validation_score(head.predict(...), labels, hp.early_stopping_metric, context, objective)`
- 반환 head: `LinearHead(..., link="identity" if objective == "mse" else "logistic")`
- record: `"head": "linear" if mse else "logistic"`, `"objective": objective`, `"validation_metric": ("mae" if objective == "mse" else hp.early_stopping_metric) if validation is not None else None`, `"train_loss": mean_squared_error(...)` (아래 함수)

```python
def mean_squared_error(pred: np.ndarray, labels: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    return float(np.mean(diff * diff))


def validation_score(pred: np.ndarray, labels: np.ndarray, metric: str, context: str, objective: str = "bce") -> Score:
    """The early-stopping score, oriented so that higher is always better."""
    if objective == "mse":
        residual = np.asarray(pred, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
        return Score(primary=-float(np.abs(residual).mean()), tiebreak=-float((residual ** 2).mean()))
    ...  # 기존 본문
```

`parse_hparams`는 그대로 둔다. `pos_weight: balanced`가 `mse`와 함께 오면 `n_pos`가 0일 수 있으므로 `base.fit`에서 `mse`일 때 `n_pos=1, n_neg=1`을 넘긴다.

- [ ] **Step 4: `base.py` 수정**

```python
    def fit(self, handle, data, mode, hparams):
        payload = parse_fit_data(data)
        counts = (payload.train.n_pos, payload.train.n_neg) if payload.objective == "bce" else (1, 1)
        hp = parse_hparams(hparams, mode, *counts)
        ...
```

`_fit_linear_probe`에서 `fit_linear_head(..., context=..., objective=data.objective)`. `combined`에 `"objective": data.objective` 추가.

- [ ] **Step 5: torch adapter 수정**

`forward`: `handle.head.probabilities(...)` → `handle.head.predict(...)`. `_published_probabilities`를 `_published_scores`로 이름을 바꾸고 `_positive_probability`를 `_score_from_outputs`로 바꾼다:

```python
    def _score_from_outputs(self, handle: TorchHandle, outputs: Any) -> Any:
        """Positive-class probability, or the value column for a regression card."""
        import torch

        kind = handle.card.output.get("type", "logits")
        if kind == "regression":
            return outputs[:, int(handle.card.output["value_index"])]
        index = int(handle.card.output["positive_index"])
        if kind == "softmax":
            return torch.softmax(outputs, dim=-1)[:, index]
        if kind in ("logits", "sigmoid"):
            return torch.sigmoid(outputs[:, index])
        raise ValueError(f"unknown output.type {kind!r}; adapter reads 'softmax', 'logits', 'sigmoid' or 'regression'")
```

`_published_scores`의 "declares no output.positive_index" 검사는 `kind != "regression"`일 때만 한다. `_probability_tensor`(attribution)는 회귀 카드나 identity head를 만나면 `NotImplementedError("attribution is defined for probabilities; regression arms are skipped in stage 6")`를 낸다.

`_run_finetune`:
- `criterion = torch.nn.MSELoss() if data.objective == "mse" else torch.nn.BCEWithLogitsLoss(pos_weight=...)`
- head bias 초기값: `mse`면 `float(data.train.y.mean())`
- targets dtype float32 (이미 그렇다)
- `_probabilities`를 `_outputs`로 바꾸고 `mse`면 sigmoid를 취하지 않는다
- `validation_score(outputs, data.validation.y, hp.early_stopping_metric, f"{mode} fit", data.objective)`
- `lifted = LinearHead.of(weight, bias, link="identity" if data.objective == "mse" else "logistic")`
- record에 `"objective": data.objective`, `"head": "linear" if mse else "logistic"`, `"validation_metric": "mae" if mse else ...`

- [ ] **Step 6: keras adapter 수정**

`forward`의 두 분기에서 `head.probabilities` → `head.predict`. 카드 없는 head 경로에서 `handle.card.output["positive_index"]` 대신:

```python
        column = handle.card.output["value_index"] if handle.card.output.get("type") == "regression" else handle.card.output["positive_index"]
        return pool_ensemble(outputs, self._pooling(handle), column)
```

`_as_two_class`는 identity head에도 그대로 쓴다. 열 1이 값이므로 `pool_ensemble`이 멤버 평균을 낸다는 사실은 같다. 한 줄 주석으로 적는다.

- [ ] **Step 7: 테스트 통과 확인**

Run: `python -m pytest tests/test_adapter_training.py tests/test_torch_adapter_fit.py tests/test_torch_builder.py tests/test_keras_adapter.py tests/test_keras_pooling.py tests/test_no_proper_nouns.py -q`
Expected: PASS (keras 테스트는 prophecg env에서)

- [ ] **Step 8: 커밋**

```bash
git add src/mival/adapters tests/test_adapter_training.py tests/test_torch_adapter_fit.py tests/test_torch_builder.py
git commit -m "feat: adapters fit and score a regression objective"
```

---

### Task 5: models stage 회귀 arm

**Files:**
- Modify: `src/mival/stages/models.py` (`PREDICTION_COLUMNS`, `Arm`, `Record`, `_labels_from`, `_fit_payload`, `_run_group`, `_fit_with_internal_cv`, `_rows`)
- Test: `tests/test_stage_models.py`

**Interfaces:**
- Consumes: Task 4 `objective`, `LinearHead.link`
- Produces: `PREDICTION_COLUMNS` with `label_value` (after `label_sens3`) and `pred_value` (after `logit`); `REGRESSION_LABEL_DEF = "value"`; `Arm.is_regression`, `Arm.objective`; train_log entry `thresholds == {}` and `selection_metric == "neg_mae"` for regression arms

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_stage_models.py` 끝에 추가. 파일의 `FakeAdapter`, `build_fixture`류 헬퍼(예: `write_fixture(tmp_path, ...)`와 `run_stage(...)`)의 실제 이름을 열어 확인하고 그 이름으로 맞춘다. 아래는 fixture가 `cohort_index`에 `label_value` 컬럼을 갖도록 확장한 뒤의 테스트다.

```python
def test_regression_arm_writes_values_and_no_thresholds(tmp_path):
    """label_def=value: pred_value filled, prob and logit null, no threshold fit."""
    adapter = FakeAdapter()
    ctx, result = run_stage(
        tmp_path, adapter,
        arms=[{"model_id": "m1", "training_mode": "inference_only", "label_def": "value"}],
        card_output={"type": "regression", "n_outputs": 1, "value_index": 0},
        label_value=lambda score: 100.0 * score,
    )
    files = sorted(ctx.layout.artifact("predictions").glob("*.parquet"))
    assert all("label_def=value" in f.name for f in files)
    frame = read_table(files[0])
    assert frame["prob"].isna().all() and frame["logit"].isna().all()
    assert frame["pred_value"].notna().all()
    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    entry = json.loads(ctx.layout.artifact("train_log.jsonl").read_text().splitlines()[0])
    assert entry["thresholds"] == {} and entry["primary_threshold_policy"] is None


def test_regression_arm_on_a_classification_card_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="output.type"):
        run_stage(tmp_path, FakeAdapter(),
                  arms=[{"model_id": "m1", "training_mode": "inference_only", "label_def": "value"}],
                  card_output={"type": "logits", "n_outputs": 2, "positive_index": 1})


def test_linear_probe_regression_selects_by_neg_mae(tmp_path):
    adapter = FakeAdapter()
    ctx, _ = run_stage(tmp_path, adapter,
                       arms=[{"model_id": "m1", "training_mode": "linear_probe", "label_def": "value",
                              "hparam_grid": {"lr": [0.1, 0.01]}}],
                       label_value=lambda score: 100.0 * score)
    assert all(fit["data"]["objective"] == "mse" for fit in adapter.fits)
    entry = json.loads(ctx.layout.artifact("train_log.jsonl").read_text().splitlines()[0])
    assert entry["training"]["internal_cv"]["selection_metric"] == "neg_mae"
    assert all("neg_mae" in c for c in entry["training"]["internal_cv"]["candidates"])
```

`FakeAdapter.fit`이 받은 `data`를 `self.fits`에 기록하도록 (이미 `mode`, `hparams`를 기록한다면 `"data": {"objective": data.get("objective", "bce")}`를 추가) 고친다. `FakeAdapter.forward`는 분류 카드면 score, 회귀 카드면 `100 * score`를 내게 한다:

```python
    def forward(self, handle, batch):
        scores = np.asarray(batch, dtype=np.float64).mean(axis=(1, 2))
        self.forward_calls.append([round(float(score), 6) for score in scores])
        values = 1.0 - scores if handle.flip else scores
        return 100.0 * values if handle.card.output.get("type") == "regression" else values
```

기존 fixture 작성 함수에 `label_value` 컬럼(기본값: `None`이면 `100 * prob`)과 `card_output` 인자를 추가한다.

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_stage_models.py -q -k regression`
Expected: FAIL, `label_def 'value' names no label column`

- [ ] **Step 3: 구현**

`PREDICTION_COLUMNS`를 다음으로 바꾼다:

```python
PREDICTION_COLUMNS = (
    "image_occurrence_id", "person_id", "split", "fold",
    "label_primary", "label_sens1", "label_sens2", "label_sens3", "label_value",
    "prob", "logit", "pred_value",
    "model_id", "training_mode", "recipe_id", "perturbation_id", "seed", "site",
)

#: The label_def of a regression arm (spec: no new run_key axis; the label names the task).
REGRESSION_LABEL_DEF = "value"
SELECTION_METRIC_REGRESSION = "neg_mae"
```

`LABEL_COLUMNS`는 파생이므로 `label_value`가 자동으로 들어간다.

`Arm`에 추가:

```python
    @property
    def is_regression(self) -> bool:
        return self.label_def == REGRESSION_LABEL_DEF

    @property
    def objective(self) -> str:
        return "mse" if self.is_regression else "bce"
```

`Record.labels: Mapping[str, Optional[float]]`. `_labels_from`:

```python
def _labels_from(frame):
    labels = {}
    for row in frame.to_dict("records"):
        labels[str(row["image_occurrence_id"])] = {
            column: (_as_float if column == "label_" + REGRESSION_LABEL_DEF else _as_int)(row.get(column))
            for column in LABEL_COLUMNS
        }
    return labels


def _as_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)
```

`_label_source`의 `cohort_index` 요구 컬럼 검사는 `LABEL_COLUMNS` 전체를 요구하므로 `label_value`가 없는 옛 cohort_index는 거부된다. 그것이 의도다 (스키마가 바뀌었다).

`_fit_payload`: `dtype = np.float64 if arm.is_regression else np.int64`를 y 두 곳에 쓰고 `"objective": arm.objective`를 추가한다.

`_run_group`:

```python
        if arm.training_mode == "inference_only":
            kind = card.output.get("type", "logits")
            if arm.is_regression != (kind == "regression"):
                raise ValueError(
                    f"{context}: label_def {arm.label_def!r} needs output.type "
                    f"{'regression' if arm.is_regression else 'logits/softmax'}, but the card declares {kind!r}"
                )
            ...
        if arm.is_regression:
            thresholds = {}
        else:
            scores = _scores_of(dev, dev_probs, arm.label_column)
            thresholds = fit_thresholds(THRESHOLD_POLICIES, scores, card.threshold, card.model_id)
```

entry: `"primary_threshold_policy": None if arm.is_regression else PRIMARY_THRESHOLD_POLICY`.

`_fit_with_internal_cv`: 후보 점수를

```python
            score = _selection_score(oof, [record.labels[arm.label_column] for record in dev], arm)
            selection.append({"hparams": dict(hparams), _selection_metric(arm): score})
```

로 바꾸고 `"selection_metric": _selection_metric(arm)`를 기록한다:

```python
def _selection_metric(arm: Arm) -> str:
    return SELECTION_METRIC_REGRESSION if arm.is_regression else SELECTION_METRIC


def _selection_score(predicted: np.ndarray, labels, arm: Arm) -> float:
    if arm.is_regression:
        return -float(np.mean(np.abs(predicted - np.asarray(labels, dtype=np.float64))))
    return auroc(predicted, labels)
```

`_rows`:

```python
def _rows(records, predicted, arm, ctx):
    logits = [None] * len(records) if arm.is_regression else _logits(predicted)
    rows = []
    for position, record in enumerate(records):
        value = float(predicted[position])
        row = {
            ..., 
            "prob": None if arm.is_regression else value,
            "logit": logits[position],
            "pred_value": value if arm.is_regression else None,
            ...
        }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_stage_models.py tests/test_golden.py tests/test_no_proper_nouns.py -q`
Expected: PASS. golden이 `PREDICTION_COLUMNS`를 스냅샷하고 있으면 새 컬럼 두 개 추가만 다르고 값은 같아야 한다. 값이 바뀌면 원인을 찾고, 컬럼 추가만이면 `expected.json`을 갱신하고 커밋 메시지에 그 이유를 적는다.

- [ ] **Step 5: 커밋**

```bash
git add src/mival/stages/models.py tests/test_stage_models.py tests/golden
git commit -m "feat: models stage runs regression arms on label_def=value"
```

---

### Task 6: evaluate 회귀 지표, 산점도, misclassify 건너뛰기

**Files:**
- Create: `src/mival/metrics/regression.py`
- Modify: `src/mival/metrics/__init__.py`, `src/mival/stages/_predictions.py`, `src/mival/stages/evaluate.py`, `src/mival/stages/misclassify.py`, `src/mival/figures.py`
- Test: `tests/test_metrics_regression.py`, `tests/test_stage_evaluate.py`, `tests/test_stage_misclassify.py`

**Interfaces:**
- Consumes: predictions with `label_value`, `pred_value`, run_key `label_def=value`
- Produces: metrics `mae`, `rmse`, `r2`, `auroc_below@<cut>` in category `regression`; `is_regression(key)`; figure `regression_scatter.png`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_metrics_regression.py`:

```python
import numpy as np
import pytest

pytest.importorskip("sklearn")

from mival.metrics import category_of
from mival.metrics.regression import auroc_below, mae, r2, regression_metrics, rmse


def test_point_metrics():
    y = np.array([40.0, 50.0, 60.0])
    p = np.array([42.0, 50.0, 56.0])
    assert mae(y, p) == pytest.approx(2.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(20 / 3))
    assert r2(y, p) == pytest.approx(1 - 20 / 200)


def test_auroc_below_scores_low_predictions_as_positive():
    y = np.array([30.0, 35.0, 55.0, 60.0])
    p = np.array([32.0, 45.0, 50.0, 58.0])
    assert auroc_below(y, p, 40.0) == pytest.approx(1.0)


def test_regression_metrics_are_registered():
    values = regression_metrics(np.array([30.0, 60.0]), np.array([35.0, 55.0]), cuts=(40.0,))
    assert set(values) == {"mae", "rmse", "r2", "auroc_below@40"}
    assert category_of("auroc_below@40") == "regression"
```

`tests/test_stage_evaluate.py`에 추가. `build_predictions`에 `regression=False` 인자를 두어 `True`면 `label_def="value"`, `prob`/`logit` None, `label_value`와 `pred_value`를 채우도록 확장한다 (`label_value = 60 - 20*label + noise`, `pred_value = label_value + noise`). 모든 row dict에 `label_value`, `pred_value` 키를 추가한다 (분류는 `pred_value=None`, `label_value`는 그래도 채워도 된다).

```python
def test_regression_arm_reports_regression_metrics_only(tmp_path):
    _, _, table = run_stage(tmp_path, BASE_SPEC, regression=True)
    reg = table[table["label_def"] == "value"]
    assert set(reg["metric"]) == {"mae", "rmse", "r2", "auroc_below@40"}
    assert set(reg["category"]) == {"regression"}
    assert reg["ci_lo"].notna().any()
    assert not (table[table["label_def"] == "primary"]["category"] == "regression").any()


def test_regression_cut_comes_from_the_spec(tmp_path):
    _, _, table = run_stage(tmp_path, {**BASE_SPEC, "regression_cuts": [35, 50]}, regression=True)
    reg = table[table["label_def"] == "value"]
    assert {"auroc_below@35", "auroc_below@50"} <= set(reg["metric"])
```

`tests/test_stage_misclassify.py`에 추가 (기존 fixture 작성 함수를 열어 회귀 run_key 파일 하나를 더 쓰는 인자를 추가한다):

```python
def test_regression_predictions_are_skipped_with_a_warning(tmp_path):
    ctx, result = run_stage(tmp_path, with_regression=True)
    assert any("regression" in w for w in result.warnings)
    cases = read_table(ctx.layout.artifact("cases.parquet"))  # 실제 산출물 이름으로 맞춘다
    assert not (cases["label_def"] == "value").any()
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_metrics_regression.py -q`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: metrics/regression.py**

```python
"""Regression metrics (category 'regression'): error, fit, and the derived classification."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .discrimination import auroc

METRICS = ("mae", "rmse", "r2")
METRIC_STEMS = ("auroc_below",)


def _arrays(y_true: Sequence, y_pred: Sequence):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mae(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - y)))


def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    total = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if total == 0 else 1.0 - float(np.sum((p - y) ** 2)) / total


def auroc_below(y_true: Sequence, y_pred: Sequence, cut: float) -> float:
    """AUROC for 'value <= cut', scoring by how far below the cut the prediction falls."""
    y, p = _arrays(y_true, y_pred)
    return auroc((y <= cut).astype(int), cut - p)


def regression_metrics(y_true: Sequence, y_pred: Sequence, cuts: Sequence[float]) -> Dict[str, float]:
    values = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}
    for cut in cuts:
        values[f"auroc_below@{cut:g}"] = auroc_below(y_true, y_pred, float(cut))
    return values
```

`auroc`가 단일 클래스에서 예외를 내면 `auroc_below`에서 `float("nan")`로 바꾼다 (`mival.metrics.discrimination.auroc`의 동작을 열어 확인).

`metrics/__init__.py`: `from . import regression as _regression`, `REGRESSION = "regression"`, `METRIC_CATEGORY.update({name: REGRESSION for name in _regression.METRICS + _regression.METRIC_STEMS})`, `__all__`에 `REGRESSION`, `regression_metrics` 추가, `from .regression import regression_metrics`.

- [ ] **Step 4: `_predictions.py`**

```python
from mival.stages.models import PREDICTION_COLUMNS, PRIMARY_THRESHOLD_POLICY, REGRESSION_LABEL_DEF, THRESHOLD_POLICIES


def is_regression(key: RunKey) -> bool:
    return key.label_def == REGRESSION_LABEL_DEF
```

`fit_operating_thresholds`: 루프 첫머리에 `if is_regression(key): thresholds[arm] = float("nan"); continue`. `recorded_thresholds_and_contamination`: `fit is None`으로 경고를 내기 전에 `entry.get("primary_threshold_policy") is None`이면 경고 없이 건너뛴다 (회귀 arm은 threshold가 없는 것이 정상).

- [ ] **Step 5: evaluate.py**

- `CATEGORIES = ("discrimination", "calibration", "clinical_utility", "interpretation", "regression")`
- `_Settings`에 `self.regression_cuts = tuple(float(c) for c in spec.get("regression_cuts", (40.0,)))`
- `_rows_for_slice` 첫머리:

```python
    if is_regression(key):
        return _regression_rows(key, frame, label_column, outcome, subgroup, settings, contaminated)
```

```python
def _regression_rows(key, frame, label_column, outcome, subgroup, settings, contaminated):
    import numpy as np
    from mival.metrics import category_of, regression_metrics
    from mival.metrics.bootstrap import bootstrap_ci, event_strata

    usable = frame[frame[label_column].notna() & frame["pred_value"].notna()]
    y = usable[label_column].to_numpy(dtype=float)
    p = usable["pred_value"].to_numpy(dtype=float)
    n = int(y.size)
    if n == 0:
        return []
    # Events for stratified resampling and the suppression floor: the primary cut.
    events = (y <= settings.regression_cuts[0]).astype(float)
    n_events = int(events.sum())

    def statistic(indices):
        return regression_metrics(y[indices], p[indices], settings.regression_cuts)

    point = statistic(np.arange(n))
    intervals = {}
    if settings.bootstrap and settings.bootstrap_replicates > 0:
        groups = usable[BOOTSTRAP_UNIT].to_numpy() if BOOTSTRAP_UNIT in usable.columns else None
        intervals = bootstrap_ci(statistic, n_rows=n, n_replicates=settings.bootstrap_replicates, seed=settings.seed,
                                 groups=groups, strata=event_strata(events, groups), alpha=settings.bootstrap_alpha, point=point)
    rows = []
    base = key.to_dict()
    for metric, value in point.items():
        interval = intervals.get(metric)
        row = dict(base)
        row.update(dict(zip(REPORT_AXES, (subgroup, outcome))))
        row.update({"category": category_of(metric), "metric": metric, "value": _finite(value),
                    "ci_lo": _finite(interval.ci_lo) if interval is not None else None,
                    "ci_hi": _finite(interval.ci_hi) if interval is not None else None,
                    "n": n, "n_events": n_events, "suppressed": bool(n_events < settings.min_events),
                    "contaminated": bool(contaminated)})
        rows.append(row)
    return rows
```

`run`의 curve 수집 블록: `if subgroup == SUBGROUP_ALL and outcome == settings.primary_outcome:` 안에서 `is_regression(key)`면 `curves` 대신 `scatter.setdefault(outcome, []).append((_curve_label(key), y, pred_value))`를 모은다. `_draw_figures(ctx, rows, curves, scatter, settings, n_analysed, warnings)`에 인자를 하나 늘리고 다음을 추가:

```python
    points = scatter.get(settings.primary_outcome, [])
    if points:
        written.append(figure_module.regression_scatter(points, directory / "regression_scatter.png"))
```

`figures.py`에 추가 (기존 `_pyplot`, `_save` 사용):

```python
def regression_scatter(series: Sequence[Tuple[str, Any, Any]], path: Union[str, Path]) -> Path:
    """Predicted against observed, one panel per arm, with the identity line."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 4), squeeze=False)
    for axis, (label, y, p) in zip(axes[0], series):
        axis.scatter(y, p, s=4, alpha=0.3)
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        axis.plot([lo, hi], [lo, hi], color="gray", linewidth=1)
        axis.set_title(label, fontsize=8)
        axis.set_xlabel("observed")
        axis.set_ylabel("predicted")
    return _save(figure, path)
```

`_comparison_rows`가 회귀 arm을 만나면 (`is_regression(key)`) 경고를 적고 건너뛴다. 비교 지표는 확률 위에 정의돼 있다.

- [ ] **Step 6: misclassify.py**

`run`에서 `predictions = load_predictions(...)` 직후:

```python
        regression = predictions["label_def"] == REGRESSION_LABEL_DEF
        if regression.any():
            warnings.append(
                f"{int(regression.sum())} regression prediction rows (label_def={REGRESSION_LABEL_DEF!r}) "
                "were skipped: the selectors are defined on probabilities"
            )
            predictions = predictions[~regression]
```

`from mival.stages.models import LABEL_COLUMNS, REGRESSION_LABEL_DEF`.

- [ ] **Step 7: 테스트 통과 확인**

Run: `python -m pytest tests/test_metrics_regression.py tests/test_metrics_registry.py tests/test_stage_evaluate.py tests/test_stage_misclassify.py tests/test_golden.py tests/test_no_proper_nouns.py -q`
Expected: PASS

- [ ] **Step 8: 커밋**

```bash
git add src/mival/metrics src/mival/stages/_predictions.py src/mival/stages/evaluate.py src/mival/stages/misclassify.py src/mival/figures.py tests/test_metrics_regression.py tests/test_stage_evaluate.py tests/test_stage_misclassify.py
git commit -m "feat: evaluation reports regression arms; audit skips them"
```

---

### Task 7: `gain` op와 DICOM loader

**Files:**
- Modify: `src/mival/ops.py` (`Gain`), `src/mival/contract.py` (`gain` field), `src/mival/compiler.py` (append `Gain`)
- Modify: `src/mival/stages/preprocess.py` (`load_dicom_record`, `LOADERS`, spec `loader`)
- Modify: `pyproject.toml` (extra `dicom = ["pydicom>=3.0"]`)
- Test: `tests/test_compiler.py`, `tests/test_dicom_loader.py`, `tests/test_stage_preprocess.py`

**Interfaces:**
- Produces: `InputContract.gain: float = 1.0`; `Gain(factor)` op applied after unit conversion and before `Normalize`; `load_dicom_record(path) -> (np.ndarray, SourceMetadata)`; preprocess spec key `loader: "npz" | "dicom"` (default `npz`)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_compiler.py`에 추가:

```python
def test_gain_is_applied_after_unit_conversion_and_before_normalize():
    contract = InputContract.from_dict({**ECGFOUNDER_DICT, "gain": 208.3333, "scaling": "none"})
    meta = mimic_source(unit="uV")
    chain = compile_recipe(contract, meta)
    names = [op["op"] for op in chain.describe()]
    assert names.index("scale_unit") < names.index("gain") < names.index("normalize")
    out = chain.apply(make_signal(meta))
    assert np.isclose(out.data[0, 0], make_signal(meta).data[0, 0] * 1e-3 * 208.3333, rtol=1e-5)
```

(`ECGFOUNDER_DICT`는 파일 상단의 dict를 변수로 빼서 쓴다. `describe()`의 키 이름이 `"op"`이 아니면 실제 키로 맞춘다.)

`tests/test_dicom_loader.py`:

```python
"""load_dicom_record reads a 12-lead ECG Waveform Storage file into the canonical layout."""

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from mival.signal import LEADS_12  # noqa: E402
from mival.stages.preprocess import load_dicom_record  # noqa: E402

# MDC codes in the channel order MIMIC files use: aVF comes before aVL.
CHANNELS = [("2:1", "Lead I"), ("2:2", "Lead II"), ("2:61", "Lead III"), ("2:62", "aVR, augmented voltage, right"),
            ("2:64", "aVF, augmented voltage, foot"), ("2:63", "aVL, augmented voltage, left"),
            ("2:3", "Lead V1"), ("2:4", "Lead V2"), ("2:5", "Lead V3"), ("2:6", "Lead V4"), ("2:7", "Lead V5"), ("2:8", "Lead V6")]


def write_dicom(path, data_mv, fs=500.0, sensitivity=0.005):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "ECG"

    item = Dataset()
    item.MultiplexGroupTimeOffset = 0
    item.WaveformOriginality = "ORIGINAL"
    item.NumberOfWaveformChannels = len(CHANNELS)
    item.NumberOfWaveformSamples = data_mv.shape[1]
    item.SamplingFrequency = fs
    item.WaveformBitsAllocated = 16
    item.WaveformSampleInterpretation = "SS"
    channels = []
    for code, meaning in CHANNELS:
        source = Dataset(); source.CodeValue = code; source.CodingSchemeDesignator = "MDC"; source.CodeMeaning = meaning
        unit = Dataset(); unit.CodeValue = "mV"; unit.CodingSchemeDesignator = "UCUM"; unit.CodeMeaning = "millivolt"
        channel = Dataset()
        channel.ChannelSourceSequence = [source]
        channel.ChannelSensitivity = sensitivity
        channel.ChannelSensitivityUnitsSequence = [unit]
        channel.ChannelBaseline = 0.0
        channel.ChannelSampleSkew = 0
        channel.WaveformBitsStored = 16
        channels.append(channel)
    item.ChannelDefinitionSequence = channels
    item.WaveformData = np.round(data_mv.T / sensitivity).astype("<i2").tobytes()
    ds.WaveformSequence = [item]
    ds.save_as(str(path), enforce_file_format=True)
    return path


def test_loader_returns_mv_with_normalised_lead_names(tmp_path):
    data = np.zeros((12, 100), dtype=np.float32)
    data[4, 10] = 1.0   # the file's fifth channel is aVF
    path = write_dicom(tmp_path / "x.dcm", data)
    array, source = load_dicom_record(path)
    assert array.shape == (12, 100) and array.dtype == np.float32
    assert source.leads == ("I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6")
    assert set(source.leads) == set(LEADS_12)
    assert source.sampling_rate_hz == 500.0 and source.n_samples == 100 and source.unit == "mV"
    assert np.isclose(array[4, 10], 1.0, atol=0.005)
```

`tests/test_stage_preprocess.py`에 추가:

```python
def test_loader_is_chosen_by_name_from_the_spec(tmp_path):
    stage = PreprocessStage()
    assert stage.resolve_loader({"loader": "npz"}) is load_npz_record
    with pytest.raises(ValueError, match="loader"):
        stage.resolve_loader({"loader": "xml"})
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_compiler.py tests/test_dicom_loader.py -q -k "gain or loader"`
Expected: FAIL

- [ ] **Step 3: Gain op와 contract**

`ops.py`에 `Normalize` 앞에:

```python
@dataclass(frozen=True)
class Gain(Op):
    """Multiply by a constant the model expects on top of its unit (a card-declared quirk)."""

    factor: float
    name: str = field(default="gain", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"factor": self.factor}

    def apply(self, sig: Signal) -> Signal:
        return Signal(data=np.ascontiguousarray(sig.data * np.float32(self.factor), dtype=np.float32),
                      leads=sig.leads, sampling_rate_hz=sig.sampling_rate_hz, unit=sig.unit)
```

`contract.py`: `InputContract`에 `gain: float = 1.0`, `from_dict`에서 `gain=float(body.get("gain", 1.0))`. `compiler.py` 6단계 앞에:

```python
    if contract.gain != 1.0:
        ops.append(Gain(contract.gain))
```

- [ ] **Step 4: DICOM loader와 선택**

`preprocess.py`의 `load_npz_record` 아래:

```python
#: MDC lead codes (ChannelSourceSequence CodeValue) to the names this project uses.
MDC_LEADS = {"2:1": "I", "2:2": "II", "2:61": "III", "2:62": "aVR", "2:63": "aVL", "2:64": "aVF",
             "2:3": "V1", "2:4": "V2", "2:5": "V3", "2:6": "V4", "2:7": "V5", "2:8": "V6"}


def load_dicom_record(path: Path) -> Tuple[np.ndarray, SourceMetadata]:
    """Read a 12-lead ECG Waveform Storage file; pydicom applies sensitivity, so values are in the channel unit."""
    import pydicom

    dataset = pydicom.dcmread(str(path))
    item = dataset.WaveformSequence[0]
    channels = item.ChannelDefinitionSequence
    leads = tuple(_lead_name(channel) for channel in channels)
    units = {channel.ChannelSensitivityUnitsSequence[0].CodeValue
             for channel in channels if "ChannelSensitivityUnitsSequence" in channel}
    data = np.ascontiguousarray(dataset.waveform_array(0).T, dtype=np.float32)
    source = SourceMetadata(leads=leads, sampling_rate_hz=float(item.SamplingFrequency),
                            n_samples=int(data.shape[1]), unit=units.pop() if len(units) == 1 else None)
    return data, source


def _lead_name(channel) -> str:
    code = channel.ChannelSourceSequence[0]
    return MDC_LEADS.get(str(code.CodeValue), str(code.CodeMeaning))


LOADERS: Dict[str, Loader] = {"npz": load_npz_record, "dicom": load_dicom_record}
```

`PreprocessStage`:

```python
    def __init__(self, loader: Optional[Loader] = None) -> None:
        self._loader = loader  # a test's injection wins over the spec

    def resolve_loader(self, spec) -> Loader:
        if self._loader is not None:
            return self._loader
        name = str((spec or {}).get("loader", "npz"))
        if name not in LOADERS:
            raise ValueError(f"preprocess.loader {name!r} is not one of {sorted(LOADERS)}")
        return LOADERS[name]
```

`run`에서 `self.loader(...)` → `loader = self.resolve_loader(spec)` 한 번 구한 뒤 `loader(...)`. `self.loader` 속성을 참조하는 테스트가 있으면 `resolve_loader`로 바꾼다.

`pyproject.toml` extra: `dicom = ["pydicom>=3.0"]`.

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest tests/test_compiler.py tests/test_contract.py tests/test_ops.py tests/test_dicom_loader.py tests/test_stage_preprocess.py tests/test_golden.py -q`
Expected: PASS. `pydicom`이 로컬에 없으면 `pip install pydicom` 후 실행.

- [ ] **Step 6: 실제 파일로 확인**

노트북 scratchpad에 받아 둔 `sample.dcm`으로:

```bash
python -c "
from pathlib import Path
from mival.stages.preprocess import load_dicom_record
data, src = load_dicom_record(Path('/private/tmp/claude-503/-Users-minseongkim-Desktop-youlab-mi-va/706c1703-6545-4a8e-ac70-fb02adab3d14/scratchpad/sample.dcm'))
print(data.shape, src)"
```

Expected: `(12, 5000)`, leads `('I','II','III','aVR','aVF','aVL','V1',...)`, 500.0, unit `mV`.

- [ ] **Step 7: 커밋**

```bash
git add src/mival/ops.py src/mival/contract.py src/mival/compiler.py src/mival/stages/preprocess.py pyproject.toml tests/test_compiler.py tests/test_dicom_loader.py tests/test_stage_preprocess.py
git commit -m "feat: DICOM loader and a card-declared gain op"
```

---

### Task 8: ModelCard 3장, study.yaml, 문서

**Files:**
- Create: `registry/models/xecg.json`, `registry/models/heartwise-lvef-binary.json`, `registry/models/heartwise-lvef-regression.json`, `studies/lvef/study.yaml`
- Modify: `docs/decisions/adaptations.md`, `docs/plans/README.md`, `docs/models/README.md`, `docs/superpowers/specs/2026-09-21-lvef-retrieve-profile-regression-design.md` (§7.4 지표 이름, §9 gain)
- Test: `tests/test_modelcard.py` (registry 전체 로드), 서버에서 `tests/test_torch_builder.py` + 카드별 smoke

**Interfaces:**
- Consumes: Task 1 builder 선언, Task 7 `gain`
- Produces: 카드 3장이 `load_registry`를 통과하고, 서버에서 `get_adapter("torch").load(card)`와 `forward`/`features`가 동작

- [ ] **Step 1: 카드 작성**

`registry/models/xecg.json`:

```json
{
  "model_id": "xecg",
  "Name": "xECG base (xLSTM foundation)",
  "Summary": "xLSTM-based ECG foundation model; pooled 1024-d representation, no task head.",
  "Link": "https://huggingface.co/riccardolunelli/xECG_base_model_v1",
  "Descriptors": {"Version": "b06053253a9b", "References": ["arXiv:2509.10151"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 100 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": "/data/mi-val/models/xecg/model.safetensors", "sha256": "812dec69ac0fbf13f39e50bde4f35f85435a66966d8785baec65c2b5c70e722c", "role": "backbone"}],
    "weights_format": "safetensors",
    "code_path": "/data/mi-val/models/xecg",
    "builder": {
      "module": "xECG:xECG",
      "kwargs": {
        "cls_type": "avg",
        "config": {"activation_fn": "gelu", "batch_size": 32, "cls_normalization": null, "cls_type": "avg",
                   "drop_path_prob": 0.0, "dropout": 0.0, "embedding_size": 1024, "num_heads": 4, "patch_size": 25,
                   "proj_factor": 2, "sampling_freq": 100, "xlstm_config": ["s", "s", "m", "m", "s", "s", "m", "m", "s"]}
      }
    },
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 100, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "none", "layout": "time_lead", "dtype": "float32"
    },
    "output": {"type": "logits", "n_outputs": null, "positive_index": null},
    "feature_layer": "pooled",
    "feature_width": 1024,
    "threshold": {},
    "runtime": {"framework": "torch==2.8.0+cu128", "python": "3.12.13", "device": "cuda",
                "env": "/data/mi-val/envs/xecg", "slstm_backend": "cuda",
                "returns": ["features", "ignore"],
                "compile_env": {"CUDA_HOME": "/usr/local/cuda-12.8", "CC": "gcc-14", "CXX": "g++-14",
                                "NVCC_PREPEND_FLAGS": "-ccbin g++-14",
                                "TORCH_EXTENSIONS_DIR": "/data/mi-val/cache/torch_extensions",
                                "TORCH_CUDA_ARCH_LIST": "8.9"}},
    "training_modes_supported": ["linear_probe", "partial_unfreeze", "full_finetune"],
    "pretraining_corpora": ["code", "chapman-ningbo", "incart"],
    "code_commit": "13e57523e418d8cddced73f81b27ba541413f4f3",
    "notes": "Forward returns (pooled, patches); only the pooled representation is used. Zero-valued samples are treated as padding by the model, so scaling must stay 'none'. Downsampled 500 to 100 Hz by the resample op."
  }
}
```

`registry/models/heartwise-lvef-binary.json`:

```json
{
  "model_id": "heartwise-lvef-binary",
  "Name": "HeartWise DeepECG-SL EfficientNetV2, LVEF <= 40",
  "Summary": "Supervised single-task binary classifier for LVEF <= 40%.",
  "Link": "https://huggingface.co/heartwise/EfficientNetV2_LVEF_equal_under_40",
  "Descriptors": {"Version": "973b356566b0", "References": ["doi:10.1093/eurheartj/ehaf1119"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 250 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": "/data/mi-val/models/heartwise-lvef/lvef_under_equal_40.pt", "sha256": "605cabad817067c225602e70158b15a65a5bfdd8fc94fdec1f54fc3c228541e8", "role": "full"}],
    "weights_format": "jit",
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 250, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "none", "gain": 208.3333, "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "logits", "n_outputs": 1, "positive_index": 0},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": "torch==2.13.0+cu130", "python": "3.10.20", "device": "cuda",
                "env": "/data/mi-val/envs/ecgfounder", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["mhi"],
    "code_commit": "cabf6c06b74731c86b8c5a27ae7101f14db5ec38",
    "notes": "gain 208.3333 is the authors' 1/0.0048 factor applied before the model (efficientnet_wrapper.mhi_factor). The authors' cohort-level spectral scaling (ledger L2-2) is not applied. TorchScript exposes no representation, so only inference_only."
  }
}
```

`registry/models/heartwise-lvef-regression.json`: 위와 같되 `model_id: heartwise-lvef-regression`, `Name`에 "regression", `Link`를 `https://huggingface.co/heartwise/deepecg-sl_finetuned_LVEF_MHI`, `Version` `1ee6698a3759`, weights `LVEF_MSE_SL.pt` sha `cc46c6e95f7e9e138f1460747ce3545db9e932a2431963bd478561b65f4926b7`, 그리고:

```json
    "weights_format": "checkpoint",
    "state_dict_key": "state_dict",
    "code_path": "/opt/deepecg-docker/notebooks",
    "builder": {"module": "EfficientNetv2:EfficientNet1DV2", "kwargs": {"num_classes": 1, "expansion_factors": [1, 2, 2, 2, 2, 2, 2]}},
    "rename_keys": {"classifier.fc1.": "classifier.3."},
    "output": {"type": "regression", "n_outputs": 1, "value_index": 0, "unit": "%"},
    "runtime": {"framework": "torch==2.13.0+cu130", "python": "3.10.20", "device": "cuda",
                "env": "/data/mi-val/envs/ecgfounder", "returns": ["logits"]},
    "notes": "Outputs LVEF in percent directly (checkpoint metrics: MAE 7.55, R2 0.46 on the authors' validation set). Same gain caveat as the binary card."
```

`EfficientNetv2.py`가 module로 import 가능한지(`/opt/deepecg-docker/notebooks/EfficientNetv2.py`, 대소문자 포함) 서버에서 확인하고 `builder.module`을 실제 파일명에 맞춘다.

- [ ] **Step 2: study.yaml**

`studies/lvef/study.yaml`:

```yaml
study_id: lvef
site: dicom-miva
seed: 20260921

stages:
  retrieve:
    dsn_env: /data/mi-val/secrets/micdm.env
    schema: cdm
    label_concept_id: 3027172
    window_days: 7
    window_days_sens2: 30
    implausible_below: 5
    primary_cutoff: 40
    sens1_cutoff: 50
    local_path_root: /scratch/mi-val/dicom
    require_local_file: true

  profile:
    test_fraction: 0.2
    n_folds: 5
    stratify_on: label_primary
    min_test_positives: 100

  preprocess:
    loader: dicom
    registry: registry/models
    allow_upsample: false
    pad_policy: reject

  models:
    registry: registry/models
    cohort_sources: [mimic-iv-ecg]
    batch_size: 64
    arms:
      - {model_id: heartwise-lvef-binary, training_mode: inference_only, label_def: primary}
      - {model_id: heartwise-lvef-regression, training_mode: inference_only, label_def: value}
      - {model_id: ecgfounder, training_mode: linear_probe, label_def: primary, hparam_grid: {lr: [0.01, 0.001]}}
      - {model_id: ecgfounder, training_mode: linear_probe, label_def: value, hparam_grid: {lr: [0.01, 0.001]}}
      - {model_id: xecg, training_mode: linear_probe, label_def: primary, hparam_grid: {lr: [0.01, 0.001]}}
      - {model_id: xecg, training_mode: linear_probe, label_def: value, hparam_grid: {lr: [0.01, 0.001]}}

  evaluate:
    outcomes: {primary: null}
    regression_cuts: [40]
    bootstrap_replicates: 2000
    figures: true

  misclassify: {}
```

`evaluate.outcomes`의 키를 `primary`로 두는 이유는 `OUTCOME_DIAGNOSTIC`의 기본값이 `stemi`(A-2)이기 때문이다. 이 study는 기본값을 쓰지 않는다.

- [ ] **Step 3: registry 로드 테스트**

`tests/test_modelcard.py`에 이미 registry 전체를 로드하는 테스트가 있으면 그대로 실행한다. 없으면 추가:

```python
def test_every_registry_card_loads(registry_dir):
    from mival.modelcard import load_registry
    cards = load_registry(registry_dir)
    assert {"ecgfounder", "prophecg-stemi", "xecg", "heartwise-lvef-binary", "heartwise-lvef-regression"} <= set(cards)
```

Run: `python -m pytest tests/test_modelcard.py tests/test_contract.py -q`
Expected: PASS

- [ ] **Step 4: 서버 smoke**

서버 `/data/mi-val/mi-va`에 브랜치를 올린 뒤:

```bash
# ecgfounder env: ECGFounder, HeartWise 2종
/data/mi-val/envs/ecgfounder/bin/python -c "
import numpy as np
from mival.adapters import get_adapter
from mival.modelcard import load_card
a = get_adapter('torch')
for cid, shape in [('ecgfounder', (2,12,5000)), ('heartwise-lvef-binary', (2,12,2500)), ('heartwise-lvef-regression', (2,12,2500))]:
    card = load_card(f'registry/models/{cid}.json'); h = a.load(card)
    x = np.random.default_rng(0).normal(0, 0.5, shape).astype(np.float32)
    print(cid, a.features(h, x).shape if card.feature_layer else a.forward(h, x))
"
# xecg env (compile env vars from the card's runtime.compile_env exported first)
/data/mi-val/envs/xecg/bin/python -c "
import numpy as np
from mival.adapters import get_adapter
from mival.modelcard import load_card
a = get_adapter('torch'); card = load_card('registry/models/xecg.json'); h = a.load(card)
print(a.features(h, np.random.default_rng(0).normal(0, 0.5, (2,12,1000)).astype(np.float32)).shape)
"
```

Expected: ECGFounder `(2, 1024)`, HeartWise binary는 0..1 값 2개, HeartWise 회귀는 LVEF 범위 값 2개, xECG `(2, 1024)`.

- [ ] **Step 5: 문서**

- `docs/decisions/adaptations.md`: A-1 "상태: retrieve 구현으로 종료", A-4 "상태: 종료. 카드 `weights_format`/`builder`/`runtime.returns`", A-5 "상태: 종료. `output.type: regression`, 연구 결정은 회귀·분류 병행". 2026-09-21 섹션에 A-7 추가: "L1 · 라벨이 이진이라는 가정. `label_value`, `pred_value` 컬럼 추가. 회귀 테스트: 분류 golden 불변". A-8 추가: "L0 · `input_contract.gain`. HeartWise의 1/0.0048 상수. L2-2의 임시 조치"
- `docs/plans/README.md`: Plan 6 행을 "구현 완료 (2026-09-21, LVEF 과제)"로, 남은 작업에서 Plan 6 제거, 이 계획 문서 링크 추가
- `docs/models/README.md`: 카드 파일명을 각 모델 옆에 적는다
- spec §7.4의 `auroc_at_40`을 `auroc_below@40`으로, §9에 `gain` 필드를 적는다

- [ ] **Step 6: 커밋**

```bash
git add registry/models studies/lvef docs tests/test_modelcard.py
git commit -m "feat: LVEF study with four ModelCards and the ledger updated"
```

---

### Task 9: 서버 end-to-end (retrieve, profile, 그리고 첫 preprocess)

**Files:**
- Create: `docs/operations/lvef-first-run.md`

**Interfaces:**
- Consumes: Tasks 2, 3, 7, 8, DICOM 캐시

- [ ] **Step 1: retrieve 실행 (DICOM 없이도 가능)**

DICOM이 아직 없으면 `require_local_file: false`로 임시 실행해 라벨 분포부터 본다.

```bash
cd /data/mi-val/mi-va && /data/mi-val/envs/mival/bin/mival run retrieve --study studies/lvef --runs-root /data/mi-val/runs
```

Expected: `counts.in == 1011623`, `excluded.label_missing`이 약 770,000, `retrieve_summary.json`의 `positives.label_primary`가 수만 건. 요약 파일을 열어 `label_delta_days` 분포와 `label_value_quantiles`를 문서에 옮긴다.

- [ ] **Step 2: profile 실행**

```bash
mival run profile --study studies/lvef --runs-root /data/mi-val/runs --input cohort_index=/data/mi-val/runs/lvef/retrieve/<hash>/artifacts/cohort_index.parquet
```

Expected: gate 통과, `profile_summary.json`의 split별 수치.

- [ ] **Step 3: DICOM 캐시가 준비되면 preprocess를 1,000건 샘플로**

`studies/lvef-smoke/study.yaml`을 복사해 `retrieve`에 `sample_limit`이 없으므로, cohort_index를 pandas로 1,000행 잘라 `cohort_index_sample.parquet`로 저장한 뒤 preprocess를 돌린다.

```bash
mival run preprocess --study studies/lvef --runs-root /data/mi-val/runs --input cohort_index=/data/mi-val/runs/lvef/_sample/cohort_index_sample.parquet
```

Expected: 5개 카드 전부 recipe 컴파일. `unit_missing`, `rate_unsupported` 0건. `recipes.json`에 xECG용 resample 500→100과 HeartWise용 250 Hz + gain이 보인다.

- [ ] **Step 4: 기록**

`docs/operations/lvef-first-run.md`에 날짜, 명령, config_hash, 요약 수치, 발견한 문제를 적는다. 협업자에게 물을 것(0.0 값, 중복 4쌍)은 `micdm-database.md`의 목록에 그대로 둔다.

- [ ] **Step 5: 커밋**

```bash
git add docs/operations/lvef-first-run.md
git commit -m "docs: first LVEF retrieve and profile run"
```

---

## 실행 순서와 병렬성

- Task 1, 2, 3은 서로 독립이다. 병렬 가능
- Task 4 → 5 → 6은 순서대로
- Task 7은 독립. DICOM 다운로드(운영)와 병렬
- Task 8은 1, 4, 7 뒤
- Task 9는 전부 뒤. Step 1, 2는 DICOM 없이 가능
