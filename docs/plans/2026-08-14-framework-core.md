# MI-VAL Framework Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모델을 ModelCard 하나로 등록하면 전처리 recipe가 모델의 입력 계약에서 자동 컴파일되고, backend에 무관하게 forward/feature 추출이 가능한 core 패키지를 만든다.

**Architecture:** `mival` core 패키지는 numpy와 scipy에만 의존한다(torch/tensorflow를 module level에서 import하지 않는다). 모델 backend는 lazy import하는 adapter 뒤에 숨고, 전처리는 사람이 쓰는 YAML이 아니라 `compile_recipe(contract, source)`가 생성하는 op chain이다. 컴파일 실패가 곧 input contract gate다.

**Tech Stack:** Python, numpy, scipy, jsonschema, pytest. 모델 실행은 PyTorch 2.13(ECGFounder) / TensorFlow-Keras 2.7 CPU(PROPHECG) 두 환경.

## Global Constraints

- **core는 두 환경 모두에서 import되어야 한다.** `src/mival/` 아래 `adapters/` 를 제외한 어떤 모듈도 `torch` 또는 `tensorflow`를 module level에서 import하지 않는다. adapter는 함수 내부에서 lazy import한다.
- **환경 2종**: `torch` = Python 3.10.20 / PyTorch 2.13.0+cu130. `keras27` = Python 3.9.25 / TensorFlow CPU 2.7.4 / Keras 2.7.0. core 코드는 Python 3.9 문법 호환이어야 한다(`X | Y` 타입 표기 금지, `typing.Optional`/`Union` 사용).
- **정규 내부 신호 표현**: `numpy.ndarray`, shape `(n_leads, n_samples)`, `dtype=float32`, 단위 `mV`.
- **op 적용 순서 고정**: `scale_unit → filters → resample → select/reconstruct_leads → crop/pad → normalize`. layout 변환은 adapter가 담당한다.
- **reason_code 어휘 고정**: `lead_unavailable`, `upsample_required`, `duration_short`, `unit_missing`, `rate_unsupported`. 새 코드를 임의로 추가하지 않는다.
- **ModelCard**: RSNA ATLAS ROADMAP 필드 + `x-mival` 확장. `x-mival.pretraining_corpora`는 **필수**다.
- **결정론**: op에 난수를 쓰지 않는다. 합성 신호는 RNG 없이 닫힌 수식으로 생성한다.
- **weights 경로**: `/data/mi-val/models/ecgfounder/12_lead_ECGFounder.pth`, `/data/mi-val/models/prophecg-stemi/*_bestmodel.h5` (DICOM-MIVA 인스턴스). weights가 필요한 테스트는 `@pytest.mark.weights`로 표시하고 파일 부재 시 skip한다.
- **이 plan의 범위 밖**: `fit`(학습), `attribute`(설명), perturbation, CDM/DICOM 접근. adapter interface에 `fit`/`attribute` 자리를 두되 구현은 Plan 4에서 한다.

## File Structure

| 파일 | 책임 |
|---|---|
| `pyproject.toml` | 패키지 정의, 의존성, pytest marker |
| `src/mival/signal.py` | `Signal`, `SourceMetadata`, 표준 lead 상수 |
| `src/mival/ops.py` | op 원시 연산과 `OpChain` |
| `src/mival/contract.py` | `InputContract`, `CompileError`, reason code |
| `src/mival/compiler.py` | `compile_recipe()` — contract + source → OpChain |
| `src/mival/modelcard.py` | ModelCard 로딩과 검증 |
| `src/mival/adapters/base.py` | `Adapter` protocol |
| `src/mival/adapters/__init__.py` | `get_adapter(name)` |
| `src/mival/adapters/torch_adapter.py` | PyTorch backend |
| `src/mival/adapters/keras_adapter.py` | Keras backend, 평균 앙상블 |
| `src/mival/synthetic.py` | 결정론적 합성 ECG 생성 |
| `registry/models/*.json` | ModelCard 인스턴스 (데이터) |
| `tests/*` | 위 각 모듈에 대응 |

---

### Task 1: 패키지 스캐폴딩과 신호 타입

**Files:**
- Create: `pyproject.toml`
- Create: `src/mival/__init__.py`
- Create: `src/mival/signal.py`
- Create: `tests/test_signal.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `mival.signal.Signal(data: np.ndarray, leads: Tuple[str, ...], sampling_rate_hz: float, unit: str)` — frozen dataclass. `.n_leads -> int`, `.n_samples -> int`, `.duration_s -> float`
  - `mival.signal.SourceMetadata(leads: Tuple[str, ...], sampling_rate_hz: float, n_samples: int, unit: Optional[str])` — frozen dataclass
  - `mival.signal.LEADS_12: Tuple[str, ...]` = `("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")`
  - `mival.signal.DERIVABLE_LEADS: Dict[str, ...]` — 유도 가능한 lead의 계수

- [ ] **Step 1: 저장소가 git repo가 아니면 초기화**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-val
git rev-parse --is-inside-work-tree 2>/dev/null || git init
```

- [ ] **Step 2: `pyproject.toml` 작성**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "mival"
version = "0.1.0"
description = "MI-VAL: model-agnostic ECG validation framework on OMOP MI-CDM"
requires-python = ">=3.9"
dependencies = ["numpy>=1.21", "scipy>=1.7", "jsonschema>=4.0"]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "weights: requires model weight files on disk",
    "torch: requires PyTorch",
    "keras: requires TensorFlow/Keras 2.7",
]
```

- [ ] **Step 3: 실패하는 테스트 작성**

`tests/test_signal.py`:

```python
import numpy as np
import pytest

from mival.signal import LEADS_12, Signal, SourceMetadata


def test_signal_reports_shape_and_duration():
    sig = Signal(
        data=np.zeros((12, 5000), dtype=np.float32),
        leads=LEADS_12,
        sampling_rate_hz=500.0,
        unit="mV",
    )
    assert sig.n_leads == 12
    assert sig.n_samples == 5000
    assert sig.duration_s == pytest.approx(10.0)


def test_signal_rejects_lead_count_mismatch():
    with pytest.raises(ValueError, match="lead count"):
        Signal(
            data=np.zeros((3, 100), dtype=np.float32),
            leads=("I", "II"),
            sampling_rate_hz=500.0,
            unit="mV",
        )


def test_signal_rejects_non_2d_data():
    with pytest.raises(ValueError, match="2-D"):
        Signal(
            data=np.zeros((2, 3, 4), dtype=np.float32),
            leads=("I", "II"),
            sampling_rate_hz=500.0,
            unit="mV",
        )


def test_source_metadata_allows_missing_unit():
    meta = SourceMetadata(
        leads=LEADS_12, sampling_rate_hz=500.0, n_samples=5000, unit=None
    )
    assert meta.unit is None
    assert meta.duration_s == pytest.approx(10.0)


def test_leads_12_is_standard_order():
    assert LEADS_12[:6] == ("I", "II", "III", "aVR", "aVL", "aVF")
    assert LEADS_12[6:] == ("V1", "V2", "V3", "V4", "V5", "V6")
```

- [ ] **Step 4: 테스트가 실패하는지 확인**

```bash
pip install -e ".[dev]"
pytest tests/test_signal.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.signal'`

- [ ] **Step 5: 최소 구현**

`src/mival/__init__.py`:

```python
"""MI-VAL: model-agnostic ECG validation framework."""

__version__ = "0.1.0"
```

`src/mival/signal.py`:

```python
"""Canonical in-memory ECG representation.

Internal layout is always (n_leads, n_samples), float32, millivolts.
Model-specific layouts are produced by adapters, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

LEADS_12: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

# Einthoven and Goldberger relations. Each entry maps a derivable lead to the
# coefficients applied to leads I and II.
DERIVABLE_LEADS: Dict[str, Dict[str, float]] = {
    "III": {"I": -1.0, "II": 1.0},
    "aVR": {"I": -0.5, "II": -0.5},
    "aVL": {"I": 1.0, "II": -0.5},
    "aVF": {"I": -0.5, "II": 1.0},
}


@dataclass(frozen=True)
class Signal:
    data: np.ndarray
    leads: Tuple[str, ...]
    sampling_rate_hz: float
    unit: str

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"signal data must be 2-D, got {self.data.ndim}-D")
        if self.data.shape[0] != len(self.leads):
            raise ValueError(
                f"lead count mismatch: data has {self.data.shape[0]} rows "
                f"but {len(self.leads)} lead names were given"
            )

    @property
    def n_leads(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_rate_hz


@dataclass(frozen=True)
class SourceMetadata:
    """What the Profile stage observed about a record, before preprocessing."""

    leads: Tuple[str, ...]
    sampling_rate_hz: float
    n_samples: int
    unit: Optional[str]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_rate_hz
```

- [ ] **Step 6: 테스트 통과 확인**

```bash
pytest tests/test_signal.py -v
```

Expected: PASS (5 passed)

- [ ] **Step 7: 커밋**

```bash
git add pyproject.toml src/mival/__init__.py src/mival/signal.py tests/test_signal.py
git commit -m "feat: add canonical Signal and SourceMetadata types"
```

---

### Task 2: Op 라이브러리 — resample, crop, pad

**Files:**
- Create: `src/mival/ops.py`
- Create: `tests/test_ops.py`

**Interfaces:**
- Consumes: `mival.signal.Signal`
- Produces:
  - `mival.ops.Op` — 추상 기반. 각 op은 `.name: str`, `.params: Dict[str, Any]`, `.apply(sig: Signal) -> Signal`
  - `mival.ops.Resample(target_hz: float)`
  - `mival.ops.Crop(n_samples: int, anchor: str = "start")`
  - `mival.ops.Pad(n_samples: int, mode: str = "zero", anchor: str = "start")`
  - `mival.ops.OpChain(ops: Tuple[Op, ...])` — `.apply(sig) -> Signal`, `.describe() -> List[Dict[str, Any]]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_ops.py`:

```python
import numpy as np
import pytest

from mival.ops import Crop, OpChain, Pad, Resample
from mival.signal import Signal


def make_signal(n_leads=2, n_samples=1000, fs=500.0, unit="mV"):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(n_leads)]
    return Signal(
        data=np.stack(rows).astype(np.float32),
        leads=tuple(f"L{i}" for i in range(n_leads)),
        sampling_rate_hz=fs,
        unit=unit,
    )


def test_resample_halves_sample_count():
    out = Resample(250.0).apply(make_signal(n_samples=1000, fs=500.0))
    assert out.sampling_rate_hz == 250.0
    assert out.n_samples == 500
    assert out.data.dtype == np.float32


def test_resample_is_identity_at_same_rate():
    sig = make_signal()
    out = Resample(500.0).apply(sig)
    np.testing.assert_array_equal(out.data, sig.data)


def test_resample_preserves_lead_names():
    out = Resample(125.0).apply(make_signal(n_leads=3))
    assert out.leads == ("L0", "L1", "L2")


def test_crop_from_start_keeps_leading_samples():
    sig = make_signal(n_samples=1000)
    out = Crop(400).apply(sig)
    assert out.n_samples == 400
    np.testing.assert_array_equal(out.data, sig.data[:, :400])


def test_crop_center_is_symmetric():
    sig = make_signal(n_samples=1000)
    out = Crop(400, anchor="center").apply(sig)
    np.testing.assert_array_equal(out.data, sig.data[:, 300:700])


def test_crop_rejects_longer_than_source():
    with pytest.raises(ValueError, match="shorter"):
        Crop(2000).apply(make_signal(n_samples=1000))


def test_pad_zero_appends_at_end():
    sig = make_signal(n_samples=100)
    out = Pad(150).apply(sig)
    assert out.n_samples == 150
    np.testing.assert_array_equal(out.data[:, :100], sig.data)
    assert np.all(out.data[:, 100:] == 0.0)


def test_pad_rejects_shorter_than_source():
    with pytest.raises(ValueError, match="longer"):
        Pad(50).apply(make_signal(n_samples=100))


def test_opchain_applies_in_order_and_describes_itself():
    chain = OpChain((Resample(250.0), Crop(300)))
    out = chain.apply(make_signal(n_samples=1000, fs=500.0))
    assert out.n_samples == 300
    assert out.sampling_rate_hz == 250.0
    assert chain.describe() == [
        {"name": "resample", "params": {"target_hz": 250.0}},
        {"name": "crop", "params": {"n_samples": 300, "anchor": "start"}},
    ]
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/test_ops.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.ops'`

- [ ] **Step 3: 최소 구현**

`src/mival/ops.py`:

```python
"""Preprocessing primitives.

Ops are deterministic and side-effect free. The canonical application order is
fixed by the compiler, not by callers: scale_unit -> filters -> resample ->
lead selection -> crop/pad -> normalize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal as sps

from mival.signal import Signal


class Op:
    """Base class. Subclasses set `name` and implement `apply`."""

    name: str = "op"

    @property
    def params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def apply(self, sig: Signal) -> Signal:
        raise NotImplementedError


@dataclass(frozen=True)
class Resample(Op):
    target_hz: float
    name: str = field(default="resample", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"target_hz": self.target_hz}

    def apply(self, sig: Signal) -> Signal:
        if sig.sampling_rate_hz == self.target_hz:
            return sig
        ratio = Fraction(self.target_hz / sig.sampling_rate_hz).limit_denominator(1000)
        resampled = sps.resample_poly(
            sig.data, ratio.numerator, ratio.denominator, axis=1
        )
        return Signal(
            data=np.ascontiguousarray(resampled, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=self.target_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class Crop(Op):
    n_samples: int
    anchor: str = "start"
    name: str = field(default="crop", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"n_samples": self.n_samples, "anchor": self.anchor}

    def apply(self, sig: Signal) -> Signal:
        if sig.n_samples < self.n_samples:
            raise ValueError(
                f"cannot crop to {self.n_samples}: source is shorter "
                f"({sig.n_samples} samples)"
            )
        if self.anchor == "center":
            start = (sig.n_samples - self.n_samples) // 2
        elif self.anchor == "start":
            start = 0
        else:
            raise ValueError(f"unknown crop anchor: {self.anchor}")
        return Signal(
            data=np.ascontiguousarray(
                sig.data[:, start : start + self.n_samples], dtype=np.float32
            ),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class Pad(Op):
    n_samples: int
    mode: str = "zero"
    anchor: str = "start"
    name: str = field(default="pad", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"n_samples": self.n_samples, "mode": self.mode, "anchor": self.anchor}

    def apply(self, sig: Signal) -> Signal:
        if sig.n_samples > self.n_samples:
            raise ValueError(
                f"cannot pad to {self.n_samples}: source is longer "
                f"({sig.n_samples} samples)"
            )
        if self.mode != "zero":
            raise ValueError(f"unknown pad mode: {self.mode}")
        deficit = self.n_samples - sig.n_samples
        before, after = (0, deficit) if self.anchor == "start" else (deficit, 0)
        padded = np.pad(sig.data, ((0, 0), (before, after)), mode="constant")
        return Signal(
            data=np.ascontiguousarray(padded, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class OpChain:
    ops: Tuple[Op, ...]

    def apply(self, sig: Signal) -> Signal:
        for op in self.ops:
            sig = op.apply(sig)
        return sig

    def describe(self) -> List[Dict[str, Any]]:
        return [{"name": op.name, "params": op.params} for op in self.ops]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_ops.py -v
```

Expected: PASS (9 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/mival/ops.py tests/test_ops.py
git commit -m "feat: add resample, crop, pad ops and OpChain"
```

---

### Task 3: Op 라이브러리 — 단위, lead 선택/유도, 정규화, 필터

**Files:**
- Modify: `src/mival/ops.py`
- Modify: `tests/test_ops.py`

**Interfaces:**
- Consumes: `mival.signal.Signal`, `mival.signal.DERIVABLE_LEADS`, Task 2의 `Op`, `OpChain`
- Produces:
  - `mival.ops.ScaleUnit(source_unit: str, target_unit: str)` — `uV`↔`mV` 변환
  - `mival.ops.SelectLeads(order: Tuple[str, ...])` — 존재하는 lead만 재정렬 선택
  - `mival.ops.ReconstructLeads(order: Tuple[str, ...])` — I, II에서 III/aVR/aVL/aVF 유도 후 선택
  - `mival.ops.Normalize(method: str)` — `none` | `global_zscore` | `per_lead_zscore`
  - `mival.ops.BandFilter(kind: str, cutoff_hz, order: int = 4)` — `highpass` | `lowpass` | `notch`

- [ ] **Step 1: 실패하는 테스트를 `tests/test_ops.py`에 추가**

```python
from mival.ops import BandFilter, Normalize, ReconstructLeads, ScaleUnit, SelectLeads
from mival.signal import LEADS_12


def make_named(leads, n_samples=1000, fs=500.0, unit="mV"):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(len(leads))]
    return Signal(
        data=np.stack(rows).astype(np.float32),
        leads=tuple(leads),
        sampling_rate_hz=fs,
        unit=unit,
    )


def test_scale_unit_uv_to_mv_divides_by_1000():
    sig = make_named(["I"], unit="uV")
    out = ScaleUnit("uV", "mV").apply(sig)
    assert out.unit == "mV"
    np.testing.assert_allclose(out.data, sig.data / 1000.0, rtol=1e-6)


def test_scale_unit_same_unit_is_identity():
    sig = make_named(["I"], unit="mV")
    out = ScaleUnit("mV", "mV").apply(sig)
    np.testing.assert_array_equal(out.data, sig.data)


def test_scale_unit_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unsupported unit"):
        ScaleUnit("V", "mV").apply(make_named(["I"], unit="V"))


def test_select_leads_reorders_and_subsets():
    sig = make_named(LEADS_12)
    order = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
    out = SelectLeads(order).apply(sig)
    assert out.leads == order
    np.testing.assert_array_equal(out.data[2], sig.data[LEADS_12.index("V1")])


def test_select_leads_rejects_missing_lead():
    with pytest.raises(KeyError, match="V6"):
        SelectLeads(("I", "V6")).apply(make_named(["I", "II"]))


def test_reconstruct_derives_lead_iii_from_i_and_ii():
    sig = make_named(["I", "II"])
    out = ReconstructLeads(("I", "II", "III")).apply(sig)
    assert out.leads == ("I", "II", "III")
    np.testing.assert_allclose(out.data[2], sig.data[1] - sig.data[0], rtol=1e-6)


def test_reconstruct_derives_avr():
    sig = make_named(["I", "II"])
    out = ReconstructLeads(("aVR",)).apply(sig)
    np.testing.assert_allclose(
        out.data[0], -0.5 * sig.data[0] - 0.5 * sig.data[1], rtol=1e-6
    )


def test_reconstruct_rejects_underivable_lead():
    with pytest.raises(KeyError, match="V3"):
        ReconstructLeads(("V3",)).apply(make_named(["I", "II"]))


def test_normalize_global_zscore_uses_whole_array():
    sig = make_named(["I", "II"])
    out = Normalize("global_zscore").apply(sig)
    assert out.data.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.data.std() == pytest.approx(1.0, abs=1e-5)


def test_normalize_per_lead_zscore_normalizes_each_row():
    sig = make_named(["I", "II"])
    out = Normalize("per_lead_zscore").apply(sig)
    for row in out.data:
        assert row.mean() == pytest.approx(0.0, abs=1e-5)
        assert row.std() == pytest.approx(1.0, abs=1e-5)


def test_normalize_none_is_identity():
    sig = make_named(["I"])
    np.testing.assert_array_equal(Normalize("none").apply(sig).data, sig.data)


def test_normalize_handles_constant_lead_without_nan():
    sig = Signal(
        data=np.ones((1, 100), dtype=np.float32),
        leads=("I",),
        sampling_rate_hz=500.0,
        unit="mV",
    )
    out = Normalize("per_lead_zscore").apply(sig)
    assert np.isfinite(out.data).all()


def test_highpass_filter_removes_dc_offset():
    sig = make_named(["I"], n_samples=5000)
    offset = Signal(
        data=sig.data + 5.0,
        leads=sig.leads,
        sampling_rate_hz=sig.sampling_rate_hz,
        unit=sig.unit,
    )
    out = BandFilter("highpass", 0.5).apply(offset)
    assert abs(float(out.data.mean())) < 0.05
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/test_ops.py -v
```

Expected: FAIL — `ImportError: cannot import name 'ScaleUnit' from 'mival.ops'`

- [ ] **Step 3: `src/mival/ops.py`에 구현 추가**

파일 상단 import에 `from mival.signal import DERIVABLE_LEADS, Signal`로 변경하고, 아래를 `OpChain` 정의 **앞**에 추가한다.

```python
_UNIT_TO_MV = {"mV": 1.0, "uV": 1e-3, "µV": 1e-3}


@dataclass(frozen=True)
class ScaleUnit(Op):
    source_unit: str
    target_unit: str
    name: str = field(default="scale_unit", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"source_unit": self.source_unit, "target_unit": self.target_unit}

    def apply(self, sig: Signal) -> Signal:
        for unit in (self.source_unit, self.target_unit):
            if unit not in _UNIT_TO_MV:
                raise ValueError(f"unsupported unit: {unit}")
        factor = _UNIT_TO_MV[self.source_unit] / _UNIT_TO_MV[self.target_unit]
        data = sig.data if factor == 1.0 else sig.data * np.float32(factor)
        return Signal(
            data=np.ascontiguousarray(data, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=self.target_unit,
        )


@dataclass(frozen=True)
class SelectLeads(Op):
    order: Tuple[str, ...]
    name: str = field(default="select_leads", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"order": list(self.order)}

    def apply(self, sig: Signal) -> Signal:
        index = {lead: i for i, lead in enumerate(sig.leads)}
        missing = [lead for lead in self.order if lead not in index]
        if missing:
            raise KeyError(f"leads not present in source: {', '.join(missing)}")
        rows = [sig.data[index[lead]] for lead in self.order]
        return Signal(
            data=np.ascontiguousarray(np.stack(rows), dtype=np.float32),
            leads=self.order,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class ReconstructLeads(Op):
    order: Tuple[str, ...]
    name: str = field(default="reconstruct_leads", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"order": list(self.order)}

    def apply(self, sig: Signal) -> Signal:
        index = {lead: i for i, lead in enumerate(sig.leads)}
        rows = []
        for lead in self.order:
            if lead in index:
                rows.append(sig.data[index[lead]])
                continue
            recipe = DERIVABLE_LEADS.get(lead)
            if recipe is None or any(src not in index for src in recipe):
                raise KeyError(f"lead cannot be derived from source: {lead}")
            acc = np.zeros(sig.n_samples, dtype=np.float32)
            for src, coeff in recipe.items():
                acc = acc + np.float32(coeff) * sig.data[index[src]]
            rows.append(acc)
        return Signal(
            data=np.ascontiguousarray(np.stack(rows), dtype=np.float32),
            leads=self.order,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class Normalize(Op):
    method: str
    name: str = field(default="normalize", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"method": self.method}

    def apply(self, sig: Signal) -> Signal:
        if self.method == "none":
            return sig
        if self.method == "global_zscore":
            mean = sig.data.mean()
            std = sig.data.std()
            std = std if std > 0 else 1.0
            data = (sig.data - mean) / std
        elif self.method == "per_lead_zscore":
            mean = sig.data.mean(axis=1, keepdims=True)
            std = sig.data.std(axis=1, keepdims=True)
            std = np.where(std > 0, std, 1.0)
            data = (sig.data - mean) / std
        else:
            raise ValueError(f"unknown normalization method: {self.method}")
        return Signal(
            data=np.ascontiguousarray(data, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class BandFilter(Op):
    kind: str
    cutoff_hz: Any
    order: int = 4
    name: str = field(default="filter", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"kind": self.kind, "cutoff_hz": self.cutoff_hz, "order": self.order}

    def apply(self, sig: Signal) -> Signal:
        nyquist = sig.sampling_rate_hz / 2.0
        if self.kind == "notch":
            quality = 30.0
            b, a = sps.iirnotch(self.cutoff_hz / nyquist, quality)
            filtered = sps.filtfilt(b, a, sig.data, axis=1)
        elif self.kind in ("highpass", "lowpass"):
            sos = sps.butter(
                self.order, self.cutoff_hz / nyquist, btype=self.kind, output="sos"
            )
            filtered = sps.sosfiltfilt(sos, sig.data, axis=1)
        else:
            raise ValueError(f"unknown filter kind: {self.kind}")
        return Signal(
            data=np.ascontiguousarray(filtered, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_ops.py -v
```

Expected: PASS (22 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/mival/ops.py tests/test_ops.py
git commit -m "feat: add unit scaling, lead selection/derivation, normalization, filtering ops"
```

---

### Task 4: InputContract와 ModelCard

**Files:**
- Create: `src/mival/contract.py`
- Create: `src/mival/modelcard.py`
- Create: `registry/models/prophecg-stemi.json`
- Create: `registry/models/ecgfounder.json`
- Create: `tests/test_modelcard.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `mival.contract.InputContract(leads, sampling_rate_hz, duration_s, unit, filters, scaling, layout, dtype)` — frozen dataclass. `.n_samples -> int`, `.from_dict(d) -> InputContract`
  - `mival.contract.CompileError(reason_code: str, detail: str)` — `Exception` 하위. `REASON_CODES: FrozenSet[str]`
  - `mival.modelcard.ModelCard` — frozen dataclass. 필드: `model_id`, `name`, `adapter`, `weights`, `ensemble`, `input_contract`, `output`, `feature_layer`, `threshold`, `runtime`, `training_modes_supported`, `pretraining_corpora`, `raw`
  - `mival.modelcard.load_card(path) -> ModelCard`
  - `mival.modelcard.load_registry(dir) -> Dict[str, ModelCard]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_modelcard.py`:

```python
import json
from pathlib import Path

import pytest

from mival.contract import REASON_CODES, CompileError, InputContract
from mival.modelcard import load_card, load_registry

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "models"


def test_input_contract_computes_sample_count():
    contract = InputContract.from_dict(
        {
            "leads": ["I", "II"],
            "sampling_rate_hz": 500,
            "duration_s": 10,
            "unit": "mV",
            "scaling": "none",
            "layout": "lead_time",
            "dtype": "float32",
        }
    )
    assert contract.n_samples == 5000
    assert contract.filters == ()


def test_reason_codes_are_the_agreed_vocabulary():
    assert REASON_CODES == frozenset(
        {"lead_unavailable", "upsample_required", "duration_short", "unit_missing"}
    )


def test_compile_error_rejects_unknown_reason_code():
    with pytest.raises(ValueError, match="unknown reason_code"):
        CompileError("something_else", "detail")


def test_compile_error_carries_reason_and_detail():
    err = CompileError("unit_missing", "no sensitivity in DICOM")
    assert err.reason_code == "unit_missing"
    assert "sensitivity" in str(err)


def test_prophecg_card_matches_verified_contract():
    card = load_card(REGISTRY / "prophecg-stemi.json")
    assert card.model_id == "prophecg-stemi"
    assert card.adapter == "keras"
    assert card.input_contract.leads == ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
    assert card.input_contract.n_samples == 5000
    assert card.input_contract.layout == "time_lead"
    assert len(card.weights) == 5
    assert card.ensemble["method"] == "mean_probability"
    assert card.output["positive_index"] == 1
    assert card.threshold["status"] == "legacy"


def test_ecgfounder_card_matches_verified_contract():
    card = load_card(REGISTRY / "ecgfounder.json")
    assert card.adapter == "torch"
    assert len(card.input_contract.leads) == 12
    assert card.input_contract.layout == "lead_time"
    assert card.input_contract.scaling == "global_zscore"
    assert "linear_probe" in card.training_modes_supported


def test_every_card_declares_pretraining_corpora():
    cards = load_registry(REGISTRY)
    assert cards
    for card in cards.values():
        assert isinstance(card.pretraining_corpora, tuple)


def test_card_without_pretraining_corpora_is_rejected(tmp_path):
    body = json.loads((REGISTRY / "ecgfounder.json").read_text())
    del body["x-mival"]["pretraining_corpora"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="pretraining_corpora"):
        load_card(bad)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/test_modelcard.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.contract'`

- [ ] **Step 3: `src/mival/contract.py` 구현**

```python
"""Model input contracts and the failure vocabulary of the contract gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple

REASON_CODES: FrozenSet[str] = frozenset(
    {"lead_unavailable", "upsample_required", "duration_short", "unit_missing"}
)


class CompileError(Exception):
    """Raised when a recipe cannot be compiled for a record.

    Every instance carries a reason_code from the fixed vocabulary so the
    exclusion ledger stays queryable.
    """

    def __init__(self, reason_code: str, detail: str) -> None:
        if reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason_code: {reason_code}")
        super().__init__(f"[{reason_code}] {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class InputContract:
    leads: Tuple[str, ...]
    sampling_rate_hz: float
    duration_s: float
    unit: str
    scaling: str
    layout: str
    dtype: str
    filters: Tuple[Dict[str, Any], ...] = field(default=())

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.sampling_rate_hz))

    @classmethod
    def from_dict(cls, body: Dict[str, Any]) -> "InputContract":
        return cls(
            leads=tuple(body["leads"]),
            sampling_rate_hz=float(body["sampling_rate_hz"]),
            duration_s=float(body["duration_s"]),
            unit=str(body["unit"]),
            scaling=str(body["scaling"]),
            layout=str(body["layout"]),
            dtype=str(body["dtype"]),
            filters=tuple(body.get("filters", ())),
        )
```

- [ ] **Step 4: `src/mival/modelcard.py` 구현**

```python
"""ModelCard loading.

A card is ROADMAP-compatible descriptive metadata plus an `x-mival` execution
extension. Cards are data, not code: adding a model must not require a source
change anywhere in this package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mival.contract import InputContract

_REQUIRED_MIVAL_FIELDS = (
    "adapter",
    "weights",
    "input_contract",
    "output",
    "training_modes_supported",
    "pretraining_corpora",
)


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    name: str
    adapter: str
    weights: Tuple[Dict[str, Any], ...]
    ensemble: Dict[str, Any]
    input_contract: InputContract
    output: Dict[str, Any]
    feature_layer: Optional[str]
    threshold: Dict[str, Any]
    runtime: Dict[str, Any]
    training_modes_supported: Tuple[str, ...]
    pretraining_corpora: Tuple[str, ...]
    raw: Dict[str, Any]


def load_card(path: Path) -> ModelCard:
    body = json.loads(Path(path).read_text())
    ext = body.get("x-mival")
    if ext is None:
        raise ValueError(f"{path}: missing x-mival extension block")
    for key in _REQUIRED_MIVAL_FIELDS:
        if key not in ext:
            raise ValueError(f"{path}: x-mival.{key} is required")
    return ModelCard(
        model_id=body["model_id"],
        name=body["Name"],
        adapter=ext["adapter"],
        weights=tuple(ext["weights"]),
        ensemble=ext.get("ensemble", {"method": "none", "members": 1}),
        input_contract=InputContract.from_dict(ext["input_contract"]),
        output=ext["output"],
        feature_layer=ext.get("feature_layer"),
        threshold=ext.get("threshold", {}),
        runtime=ext.get("runtime", {}),
        training_modes_supported=tuple(ext["training_modes_supported"]),
        pretraining_corpora=tuple(ext["pretraining_corpora"]),
        raw=body,
    )


def load_registry(directory: Path) -> Dict[str, ModelCard]:
    cards = {}
    for path in sorted(Path(directory).glob("*.json")):
        card = load_card(path)
        cards[card.model_id] = card
    return cards
```

- [ ] **Step 5: ModelCard 인스턴스 작성**

`registry/models/prophecg-stemi.json`:

```json
{
  "model_id": "prophecg-stemi",
  "Name": "PROPHECG-STEMI",
  "Summary": "Single-task 8-lead ECG classifier for ST-segment elevation myocardial infarction, deployed as a five-member mean ensemble.",
  "Link": "https://doi.org/10.1016/j.annemergmed.2024.06.004",
  "Descriptors": {
    "Version": "2022-12-04 BestModelSaved",
    "References": [
      {
        "DOI": "10.1016/j.annemergmed.2024.06.004",
        "PMID": "39066765",
        "Title": "Development of Clinically Validated Artificial Intelligence Model for Detecting ST-segment Elevation Myocardial Infarction"
      }
    ]
  },
  "Model properties": {
    "Input": "8-lead 12-lead-derived surface ECG, 10 seconds at 500 Hz"
  },
  "Imaging": {
    "Modality": "ECG"
  },
  "x-mival": {
    "adapter": "keras",
    "weights": [
      {"uri": "/data/mi-val/models/prophecg-stemi/221204_16431_bestmodel.h5", "sha256": "316e4fac1ccf994625dfc918f5bb5c6cecc3ff689dcf38b698b4322427754894", "role": "ensemble_member"},
      {"uri": "/data/mi-val/models/prophecg-stemi/221204_16432_bestmodel.h5", "sha256": "5754744b3ef464266d7891ab62a403c82ce6314f17578b400157c2e2684d30f3", "role": "ensemble_member"},
      {"uri": "/data/mi-val/models/prophecg-stemi/221204_16433_bestmodel.h5", "sha256": "8691d778cb439189a5d19173daed58052d2570b9fe4b3f4ccae10894349f3f39", "role": "ensemble_member"},
      {"uri": "/data/mi-val/models/prophecg-stemi/221204_16434_bestmodel.h5", "sha256": "6cb467412d62d4ac542695fdb674f6e760ab0dbd02c4a9f9ad8a665f2fa4690b", "role": "ensemble_member"},
      {"uri": "/data/mi-val/models/prophecg-stemi/221204_16435_bestmodel.h5", "sha256": "1fea2e24a4a0acc68141ab4a4cc5b699eab58b0b485e777a226e2e6fd187939d", "role": "ensemble_member"}
    ],
    "ensemble": {"method": "mean_probability", "members": 5},
    "input_contract": {
      "leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 500,
      "duration_s": 10,
      "unit": "mV",
      "filters": [],
      "scaling": "none",
      "layout": "time_lead",
      "dtype": "float32"
    },
    "output": {"type": "softmax", "positive_index": 1},
    "feature_layer": null,
    "threshold": {
      "value": 0.0768,
      "provenance": "archived PTB-XL revision notebook",
      "status": "legacy"
    },
    "runtime": {"framework": "tensorflow==2.7.4", "keras": "2.7.0", "python": "3.9.25", "device": "cpu"},
    "training_modes_supported": ["inference_only", "linear_probe", "full_finetune"],
    "pretraining_corpora": ["institutional-ed-ecg-2022"],
    "notes": "Amplitude unit/scaling is not documented in the archived dataset notebook; scaling is declared as none pending confirmation. Training used 5120 samples with 120 leading zeros, but the shared H5 signature is 5000."
  }
}
```

`registry/models/ecgfounder.json`:

```json
{
  "model_id": "ecgfounder",
  "Name": "ECGFounder (12-lead)",
  "Summary": "Large-scale supervised ECG foundation model producing 1024-dimensional representations and 150 pretraining logits.",
  "Link": "https://huggingface.co/PKUDigitalHealth/ECGFounder",
  "Descriptors": {
    "Version": "d9b1793951b2342f5f7e84f1ac03cd37f8a08724",
    "References": []
  },
  "Model properties": {
    "Input": "12-lead surface ECG, 10 seconds at 500 Hz"
  },
  "Imaging": {
    "Modality": "ECG"
  },
  "x-mival": {
    "adapter": "torch",
    "weights": [
      {"uri": "/data/mi-val/models/ecgfounder/12_lead_ECGFounder.pth", "sha256": "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997", "role": "backbone"}
    ],
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 500,
      "duration_s": 10,
      "unit": "mV",
      "filters": [],
      "scaling": "global_zscore",
      "layout": "lead_time",
      "dtype": "float32"
    },
    "output": {"type": "logits", "n_outputs": 150, "positive_index": null},
    "feature_layer": "backbone",
    "threshold": {},
    "runtime": {"framework": "torch==2.13.0+cu130", "python": "3.10.20", "device": "cuda"},
    "training_modes_supported": ["linear_probe", "partial_unfreeze", "full_finetune"],
    "pretraining_corpora": ["harvard-emory-ecg"],
    "code_path": "/opt/ecgfounder",
    "code_commit": "68d25f25e323a4a423b9d9e8ea2e0af3f234bf22",
    "feature_width": 1024,
    "notes": "This checkpoint emits no STEMI probability. A STEMI head must be fitted; inference_only is therefore not a supported training mode."
  }
}
```

- [ ] **Step 6: 테스트 통과 확인**

```bash
pytest tests/test_modelcard.py -v
```

Expected: PASS (8 passed)

- [ ] **Step 7: 커밋**

```bash
git add src/mival/contract.py src/mival/modelcard.py registry/models tests/test_modelcard.py
git commit -m "feat: add InputContract, CompileError vocabulary, and ModelCard registry"
```

> **주의:** 두 카드의 `pretraining_corpora` 값은 잠정값이다. contamination gate(Plan 4)를 켜기 전에 원 논문·모델 카드와 대조해 확정해야 한다. 이 사실을 `docs/decisions/README.md`에 open question으로 남긴다.

---

### Task 5: Recipe 컴파일러 (input contract gate)

**Files:**
- Create: `src/mival/compiler.py`
- Create: `tests/test_compiler.py`

**Interfaces:**
- Consumes: `mival.contract.InputContract`, `mival.contract.CompileError`, `mival.signal.SourceMetadata`, `mival.ops.*`
- Produces:
  - `mival.compiler.compile_recipe(contract: InputContract, source: SourceMetadata, allow_upsample: bool = False, pad_policy: str = "reject") -> OpChain`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_compiler.py`:

```python
import numpy as np
import pytest

from mival.compiler import compile_recipe
from mival.contract import CompileError, InputContract
from mival.signal import LEADS_12, Signal, SourceMetadata

PROPHECG = InputContract.from_dict(
    {
        "leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        "sampling_rate_hz": 500,
        "duration_s": 10,
        "unit": "mV",
        "scaling": "none",
        "layout": "time_lead",
        "dtype": "float32",
    }
)

ECGFOUNDER = InputContract.from_dict(
    {
        "leads": list(LEADS_12),
        "sampling_rate_hz": 500,
        "duration_s": 10,
        "unit": "mV",
        "scaling": "global_zscore",
        "layout": "lead_time",
        "dtype": "float32",
    }
)


def mimic_source(unit="mV", fs=500.0, n=5000, leads=LEADS_12):
    return SourceMetadata(
        leads=tuple(leads), sampling_rate_hz=fs, n_samples=n, unit=unit
    )


def make_signal(meta):
    t = np.arange(meta.n_samples, dtype=np.float32) / meta.sampling_rate_hz
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(len(meta.leads))]
    return Signal(
        data=np.stack(rows).astype(np.float32),
        leads=meta.leads,
        sampling_rate_hz=meta.sampling_rate_hz,
        unit=meta.unit,
    )


def test_prophecg_chain_has_canonical_order():
    chain = compile_recipe(PROPHECG, mimic_source())
    assert [op["name"] for op in chain.describe()] == [
        "scale_unit",
        "select_leads",
        "normalize",
    ]


def test_crop_is_inserted_only_when_source_is_longer():
    chain = compile_recipe(PROPHECG, mimic_source(n=6000))
    assert [op["name"] for op in chain.describe()] == [
        "scale_unit",
        "select_leads",
        "crop",
        "normalize",
    ]


def test_prophecg_chain_produces_contract_shape():
    meta = mimic_source()
    out = compile_recipe(PROPHECG, meta).apply(make_signal(meta))
    assert out.data.shape == (8, 5000)
    assert out.leads == PROPHECG.leads
    assert out.sampling_rate_hz == 500.0


def test_ecgfounder_chain_produces_contract_shape():
    meta = mimic_source()
    out = compile_recipe(ECGFOUNDER, meta).apply(make_signal(meta))
    assert out.data.shape == (12, 5000)
    assert out.data.mean() == pytest.approx(0.0, abs=1e-5)


def test_resample_is_inserted_when_rates_differ():
    chain = compile_recipe(PROPHECG, mimic_source(fs=1000.0, n=10000))
    assert "resample" in [op["name"] for op in chain.describe()]


def test_reconstruct_is_used_when_limb_leads_are_derivable():
    source = mimic_source(leads=("I", "II", "V1", "V2", "V3", "V4", "V5", "V6"))
    chain = compile_recipe(ECGFOUNDER, source)
    assert "reconstruct_leads" in [op["name"] for op in chain.describe()]


def test_missing_unit_raises_unit_missing():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(unit=None))
    assert excinfo.value.reason_code == "unit_missing"


def test_underivable_precordial_lead_raises_lead_unavailable():
    source = mimic_source(leads=("I", "II", "V1"))
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, source)
    assert excinfo.value.reason_code == "lead_unavailable"
    assert "V2" in excinfo.value.detail


def test_upsampling_is_rejected_by_default():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(fs=250.0, n=2500))
    assert excinfo.value.reason_code == "upsample_required"


def test_upsampling_is_allowed_when_explicitly_enabled():
    chain = compile_recipe(
        PROPHECG, mimic_source(fs=250.0, n=2500), allow_upsample=True
    )
    assert "resample" in [op["name"] for op in chain.describe()]


def test_short_record_raises_duration_short():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(n=2500))
    assert excinfo.value.reason_code == "duration_short"


def test_short_record_is_padded_when_policy_allows():
    chain = compile_recipe(PROPHECG, mimic_source(n=2500), pad_policy="zero")
    assert "pad" in [op["name"] for op in chain.describe()]
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/test_compiler.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.compiler'`

- [ ] **Step 3: 최소 구현**

`src/mival/compiler.py`:

```python
"""Compile a preprocessing recipe from a model's declared input contract.

Nobody hand-writes a per-model recipe. Adding a model means adding a card.
A compile failure IS the input contract gate: the CompileError's reason_code
is what the exclusion ledger records.
"""

from __future__ import annotations

from typing import List

from mival.contract import CompileError, InputContract
from mival.ops import (
    BandFilter,
    Crop,
    Normalize,
    Op,
    OpChain,
    Pad,
    ReconstructLeads,
    Resample,
    ScaleUnit,
    SelectLeads,
)
from mival.signal import DERIVABLE_LEADS, SourceMetadata


def compile_recipe(
    contract: InputContract,
    source: SourceMetadata,
    allow_upsample: bool = False,
    pad_policy: str = "reject",
) -> OpChain:
    ops: List[Op] = []

    # 1. unit. An unknown unit is never guessed: a silently wrong amplitude
    #    cannot be detected downstream.
    if source.unit is None:
        raise CompileError(
            "unit_missing", "source declares no amplitude unit or sensitivity"
        )
    ops.append(ScaleUnit(source.unit, contract.unit))

    # 2. model-declared filters
    for spec in contract.filters:
        ops.append(
            BandFilter(
                kind=spec["kind"],
                cutoff_hz=spec["cutoff_hz"],
                order=int(spec.get("order", 4)),
            )
        )

    # 3. sampling rate
    if source.sampling_rate_hz != contract.sampling_rate_hz:
        if source.sampling_rate_hz < contract.sampling_rate_hz and not allow_upsample:
            raise CompileError(
                "upsample_required",
                f"source is {source.sampling_rate_hz} Hz but the contract needs "
                f"{contract.sampling_rate_hz} Hz",
            )
        ops.append(Resample(contract.sampling_rate_hz))
        n_after_resample = int(
            round(source.n_samples * contract.sampling_rate_hz / source.sampling_rate_hz)
        )
    else:
        n_after_resample = source.n_samples

    # 4. leads
    available = set(source.leads)
    missing = [lead for lead in contract.leads if lead not in available]
    underivable = [
        lead
        for lead in missing
        if lead not in DERIVABLE_LEADS
        or any(src not in available for src in DERIVABLE_LEADS[lead])
    ]
    if underivable:
        raise CompileError(
            "lead_unavailable",
            f"source cannot supply or derive: {', '.join(sorted(underivable))}",
        )
    ops.append(
        ReconstructLeads(contract.leads) if missing else SelectLeads(contract.leads)
    )

    # 5. length
    if n_after_resample > contract.n_samples:
        ops.append(Crop(contract.n_samples))
    elif n_after_resample < contract.n_samples:
        if pad_policy == "reject":
            raise CompileError(
                "duration_short",
                f"source yields {n_after_resample} samples but the contract needs "
                f"{contract.n_samples}",
            )
        ops.append(Pad(contract.n_samples, mode=pad_policy))

    # 6. normalization
    ops.append(Normalize(contract.scaling))

    return OpChain(tuple(ops))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_compiler.py -v
```

Expected: PASS (12 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/mival/compiler.py tests/test_compiler.py
git commit -m "feat: compile preprocessing recipes from input contracts (contract gate)"
```

---

### Task 6: Adapter protocol과 lookup

**Files:**
- Create: `src/mival/adapters/base.py`
- Create: `src/mival/adapters/__init__.py`
- Create: `tests/test_adapter_registry.py`

**Interfaces:**
- Consumes: `mival.modelcard.ModelCard`, `mival.signal.Signal`
- Produces:
  - `mival.adapters.base.Adapter` — 추상 기반. `load(card) -> Any`, `forward(handle, batch) -> np.ndarray`, `features(handle, batch) -> np.ndarray`, `trainable_groups(handle) -> List[str]`
  - `mival.adapters.base.to_model_layout(batch: np.ndarray, layout: str) -> np.ndarray`
  - `mival.adapters.get_adapter(name: str) -> Adapter`
  - `mival.adapters.available_adapters() -> List[str]`

`forward`/`features`의 `batch` 인자는 **항상 정규 layout** `(B, n_leads, n_samples)`이다. 모델별 layout 변환은 adapter 내부에서 `to_model_layout`으로 처리한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_adapter_registry.py`:

```python
import numpy as np
import pytest

from mival.adapters import available_adapters, get_adapter
from mival.adapters.base import to_model_layout


def test_lead_time_layout_is_identity():
    batch = np.zeros((2, 12, 5000), dtype=np.float32)
    assert to_model_layout(batch, "lead_time").shape == (2, 12, 5000)


def test_time_lead_layout_transposes_last_two_axes():
    batch = np.zeros((2, 8, 5000), dtype=np.float32)
    assert to_model_layout(batch, "time_lead").shape == (2, 5000, 8)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="unknown layout"):
        to_model_layout(np.zeros((1, 2, 3), dtype=np.float32), "channels_middle")


def test_registry_lists_both_backends():
    assert set(available_adapters()) == {"torch", "keras"}


def test_unknown_adapter_name_is_rejected():
    with pytest.raises(KeyError, match="no adapter named"):
        get_adapter("jax")
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/test_adapter_registry.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.adapters'`

- [ ] **Step 3: 최소 구현**

`src/mival/adapters/base.py`:

```python
"""Adapter interface.

Backends are imported lazily inside adapter methods so that `mival` core
imports cleanly in both the torch and the keras27 environment.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from mival.modelcard import ModelCard


def to_model_layout(batch: np.ndarray, layout: str) -> np.ndarray:
    """Convert canonical (B, n_leads, n_samples) into the model's layout."""
    if layout == "lead_time":
        return batch
    if layout == "time_lead":
        return np.ascontiguousarray(np.transpose(batch, (0, 2, 1)))
    raise ValueError(f"unknown layout: {layout}")


class Adapter:
    name: str = "adapter"

    def load(self, card: ModelCard) -> Any:
        raise NotImplementedError

    def forward(self, handle: Any, batch: np.ndarray) -> np.ndarray:
        """Return positive-class probability, shape (B,)."""
        raise NotImplementedError

    def features(self, handle: Any, batch: np.ndarray) -> np.ndarray:
        """Return representations, shape (B, D)."""
        raise NotImplementedError

    def trainable_groups(self, handle: Any) -> List[str]:
        raise NotImplementedError

    # fit() and attribute() are added in Plan 4 (Models stage).
```

`src/mival/adapters/__init__.py`:

```python
"""Adapter lookup. Adding a backend means adding a module and one entry here."""

from __future__ import annotations

from typing import Dict, List

from mival.adapters.base import Adapter, to_model_layout

_FACTORIES = {
    "torch": "mival.adapters.torch_adapter:TorchAdapter",
    "keras": "mival.adapters.keras_adapter:KerasAdapter",
}

_CACHE: Dict[str, Adapter] = {}


def available_adapters() -> List[str]:
    return sorted(_FACTORIES)


def get_adapter(name: str) -> Adapter:
    if name not in _FACTORIES:
        raise KeyError(f"no adapter named {name!r}; have {available_adapters()}")
    if name not in _CACHE:
        import importlib

        module_path, class_name = _FACTORIES[name].split(":")
        module = importlib.import_module(module_path)
        _CACHE[name] = getattr(module, class_name)()
    return _CACHE[name]


__all__ = ["Adapter", "available_adapters", "get_adapter", "to_model_layout"]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_adapter_registry.py -v
```

Expected: PASS (5 passed). `available_adapters`는 모듈을 import하지 않으므로 backend 미설치 환경에서도 통과한다.

- [ ] **Step 5: 커밋**

```bash
git add src/mival/adapters tests/test_adapter_registry.py
git commit -m "feat: add adapter protocol, layout conversion, and lazy backend lookup"
```

---

### Task 7: Torch adapter (ECGFounder)

**Files:**
- Create: `src/mival/adapters/torch_adapter.py`
- Create: `tests/conftest.py`
- Create: `tests/test_torch_adapter.py`

**Interfaces:**
- Consumes: `mival.adapters.base.Adapter`, `to_model_layout`, `mival.modelcard.ModelCard`
- Produces:
  - `mival.adapters.torch_adapter.TorchAdapter` — `load`, `features`, `trainable_groups` 구현. `forward`는 `NotImplementedError`를 던진다(이 체크포인트는 확률을 출력하지 않는다)
  - `TorchHandle` — `.module`, `.device`, `.card`

- [ ] **Step 1: 공용 fixture 작성**

`tests/conftest.py`:

```python
from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "models"


def _weights_present(model_id: str) -> bool:
    from mival.modelcard import load_card

    card = load_card(REGISTRY / f"{model_id}.json")
    return all(Path(w["uri"]).exists() for w in card.weights)


@pytest.fixture(scope="session")
def registry_dir() -> Path:
    return REGISTRY


def pytest_runtest_setup(item):
    if item.get_closest_marker("torch"):
        pytest.importorskip("torch")
    if item.get_closest_marker("keras"):
        pytest.importorskip("tensorflow")
    marker = item.get_closest_marker("weights")
    if marker and marker.args:
        model_id = marker.args[0]
        if not _weights_present(model_id):
            pytest.skip(f"weights for {model_id} not present on this machine")
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_torch_adapter.py`:

```python
import numpy as np
import pytest

from mival.adapters import get_adapter
from mival.modelcard import load_card

pytestmark = [pytest.mark.torch, pytest.mark.weights("ecgfounder")]


def synthetic_batch(batch_size=2, n_leads=12, n_samples=5000, fs=500.0):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(n_leads)]
    one = np.stack(rows).astype(np.float32)
    return np.repeat(one[None, ...], batch_size, axis=0)


def test_load_returns_handle_with_module_and_device(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    handle = get_adapter("torch").load(card)
    assert handle.module is not None
    assert handle.card.model_id == "ecgfounder"


def test_features_have_declared_width(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    adapter = get_adapter("torch")
    feats = adapter.features(adapter.load(card), synthetic_batch())
    assert feats.shape == (2, 1024)
    assert np.isfinite(feats).all()


def test_trainable_groups_are_named(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    adapter = get_adapter("torch")
    groups = adapter.trainable_groups(adapter.load(card))
    assert groups
    assert all(isinstance(name, str) for name in groups)


def test_forward_is_unavailable_without_a_head(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    adapter = get_adapter("torch")
    with pytest.raises(NotImplementedError, match="no STEMI head"):
        adapter.forward(adapter.load(card), synthetic_batch())
```

- [ ] **Step 3: 테스트가 실패하는지 확인**

DICOM-MIVA 인스턴스의 torch 환경에서:

```bash
pytest tests/test_torch_adapter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.adapters.torch_adapter'`

- [ ] **Step 4: 최소 구현**

`src/mival/adapters/torch_adapter.py`:

```python
"""PyTorch backend.

torch is imported inside methods so that importing `mival.adapters` stays
cheap and works in the keras27 environment where torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class TorchHandle:
    module: Any
    device: str
    card: ModelCard


class TorchAdapter(Adapter):
    name = "torch"

    def load(self, card: ModelCard) -> TorchHandle:
        import gc
        import sys

        import torch

        code_path = card.raw["x-mival"]["code_path"]
        if code_path not in sys.path:
            sys.path.insert(0, code_path)
        from net1d import Net1D  # official ECGFounder checkout

        # The checkpoint stores two NumPy scalar metadata objects. Only those
        # known globals are allow-listed; arbitrary pickle execution stays off.
        safe_globals = [
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            (np.dtype, "numpy.dtype"),
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(safe_globals):
            checkpoint = torch.load(
                card.weights[0]["uri"], map_location="cpu", weights_only=True
            )

        state_dict = checkpoint["state_dict"]
        n_classes = int(state_dict["dense.weight"].shape[0])
        declared = card.output.get("n_outputs")
        if declared is not None and int(declared) != n_classes:
            raise ValueError(
                f"card declares {declared} outputs but the checkpoint has {n_classes}"
            )
        module = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=n_classes,
            use_bn=False,
            use_do=False,
            return_features=True,
            verbose=False,
        )
        module.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        gc.collect()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = module.to(device).eval()
        return TorchHandle(module=module, device=device, card=card)

    def features(self, handle: TorchHandle, batch: np.ndarray) -> np.ndarray:
        import torch

        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        tensor = torch.from_numpy(np.ascontiguousarray(arranged)).to(handle.device)
        with torch.inference_mode():
            _logits, features = handle.module(tensor)
        return features.detach().cpu().numpy()

    def forward(self, handle: TorchHandle, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ECGFounder checkpoint has no STEMI head; fit one with linear_probe "
            "or full_finetune (Plan 4) before calling forward"
        )

    def trainable_groups(self, handle: TorchHandle) -> List[str]:
        return [name for name, _ in handle.module.named_children()]
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
pytest tests/test_torch_adapter.py -v
```

Expected: PASS (4 passed)

이 `load`/`features` 구현은 2026-08-10에 실제로 성공한 `infra/aws/verify-ecgfounder.py`(checkpoint dict 구조, `n_classes`를 `dense.weight`에서 유도, `return_features=True`, `(logits, features)` 반환, `safe_globals` allow-list)와 동일한 호출 방식이다. 불일치가 나면 그 스크립트를 정본으로 삼는다.

- [ ] **Step 6: 커밋**

```bash
git add src/mival/adapters/torch_adapter.py tests/conftest.py tests/test_torch_adapter.py
git commit -m "feat: add torch adapter with ECGFounder feature extraction"
```

---

### Task 8: Keras adapter (PROPHECG 평균 앙상블)

**Files:**
- Create: `src/mival/adapters/keras_adapter.py`
- Create: `tests/test_keras_adapter.py`

**Interfaces:**
- Consumes: `mival.adapters.base.Adapter`, `to_model_layout`, `mival.modelcard.ModelCard`
- Produces:
  - `mival.adapters.keras_adapter.KerasAdapter` — `load`, `forward`(5-member 평균 후 positive index 추출), `trainable_groups`. `features`는 `feature_layer`가 `null`이면 `NotImplementedError`
  - `KerasHandle` — `.members: List[Any]`, `.card`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_keras_adapter.py`:

```python
import numpy as np
import pytest

from mival.adapters import get_adapter
from mival.modelcard import load_card

pytestmark = [pytest.mark.keras, pytest.mark.weights("prophecg-stemi")]


def synthetic_batch(batch_size=3, n_leads=8, n_samples=5000, fs=500.0):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(n_leads)]
    one = np.stack(rows).astype(np.float32)
    return np.repeat(one[None, ...], batch_size, axis=0)


def test_load_returns_five_members(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    handle = get_adapter("keras").load(card)
    assert len(handle.members) == 5


def test_forward_returns_one_probability_per_record(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    probs = adapter.forward(adapter.load(card), synthetic_batch())
    assert probs.shape == (3,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_forward_is_deterministic(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    handle = adapter.load(card)
    batch = synthetic_batch()
    np.testing.assert_allclose(
        adapter.forward(handle, batch), adapter.forward(handle, batch), rtol=0, atol=0
    )


def test_identical_rows_produce_identical_probabilities(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    probs = adapter.forward(adapter.load(card), synthetic_batch(batch_size=3))
    assert probs[0] == pytest.approx(probs[1]) == pytest.approx(probs[2])


def test_features_unavailable_when_no_feature_layer(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    with pytest.raises(NotImplementedError, match="feature_layer"):
        adapter.features(adapter.load(card), synthetic_batch())
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

DICOM-MIVA 인스턴스의 keras27 환경에서:

```bash
pytest tests/test_keras_adapter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.adapters.keras_adapter'`

- [ ] **Step 3: 최소 구현**

`src/mival/adapters/keras_adapter.py`:

```python
"""Keras 2.7 backend.

The archived PROPHECG runtime is intentionally CPU-only: it predates the CUDA
toolkit on this instance. TensorFlow is imported inside load() so that the
torch environment can import mival.adapters without TensorFlow installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class KerasHandle:
    members: List[Any]
    card: ModelCard


class KerasAdapter(Adapter):
    name = "keras"

    def load(self, card: ModelCard) -> KerasHandle:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        members = [
            tf.keras.models.load_model(spec["uri"], compile=False)
            for spec in card.weights
        ]
        return KerasHandle(members=members, card=card)

    def forward(self, handle: KerasHandle, batch: np.ndarray) -> np.ndarray:
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        outputs = [
            np.asarray(member(arranged, training=False)) for member in handle.members
        ]
        method = handle.card.ensemble.get("method", "none")
        if method == "mean_probability":
            pooled = np.mean(np.stack(outputs, axis=0), axis=0)
        elif method == "none":
            pooled = outputs[0]
        else:
            raise ValueError(f"unsupported ensemble method: {method}")
        return pooled[:, handle.card.output["positive_index"]].astype(np.float64)

    def features(self, handle: KerasHandle, batch: np.ndarray) -> np.ndarray:
        if handle.card.feature_layer is None:
            raise NotImplementedError(
                f"{handle.card.model_id} declares no feature_layer; "
                "representation extraction is unavailable"
            )
        import tensorflow as tf

        member = handle.members[0]
        extractor = tf.keras.Model(
            inputs=member.input, outputs=member.get_layer(handle.card.feature_layer).output
        )
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        return np.asarray(extractor(arranged, training=False))

    def trainable_groups(self, handle: KerasHandle) -> List[str]:
        return [layer.name for layer in handle.members[0].layers]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_keras_adapter.py -v
```

Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/mival/adapters/keras_adapter.py tests/test_keras_adapter.py
git commit -m "feat: add keras adapter with PROPHECG mean-probability ensemble"
```

---

### Task 9: 합성 ECG 생성기와 golden test

**Files:**
- Create: `src/mival/synthetic.py`
- Create: `scripts/make_golden.py`
- Create: `tests/golden/expected.json`
- Create: `tests/test_golden.py`
- Create: `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `mival.signal.Signal`, `mival.compiler.compile_recipe`, `mival.adapters.get_adapter`
- Produces:
  - `mival.synthetic.synthetic_ecg(record_index: int, leads=LEADS_12, sampling_rate_hz=500.0, n_samples=5000) -> Signal` — RNG 없이 닫힌 수식으로 생성, `record_index`가 파형을 결정
  - `mival.synthetic.GOLDEN_SOURCE: SourceMetadata`

- [ ] **Step 1: 합성 생성기 테스트 작성**

`tests/test_synthetic.py`:

```python
import numpy as np

from mival.signal import LEADS_12
from mival.synthetic import synthetic_ecg


def test_same_index_produces_identical_signal():
    a = synthetic_ecg(3)
    b = synthetic_ecg(3)
    np.testing.assert_array_equal(a.data, b.data)


def test_different_index_produces_different_signal():
    assert not np.allclose(synthetic_ecg(0).data, synthetic_ecg(1).data)


def test_shape_and_metadata_match_mimic_style_source():
    sig = synthetic_ecg(0)
    assert sig.data.shape == (12, 5000)
    assert sig.leads == LEADS_12
    assert sig.sampling_rate_hz == 500.0
    assert sig.unit == "mV"


def test_values_are_finite_and_physiologically_scaled():
    data = synthetic_ecg(7).data
    assert np.isfinite(data).all()
    assert np.abs(data).max() < 10.0
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/test_synthetic.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mival.synthetic'`

- [ ] **Step 3: 합성 생성기 구현**

`src/mival/synthetic.py`:

```python
"""Deterministic synthetic ECG.

No RNG: the waveform is a closed-form function of record_index and lead index,
so golden hashes stay stable across numpy versions and machines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from mival.signal import LEADS_12, Signal, SourceMetadata

GOLDEN_SOURCE = SourceMetadata(
    leads=LEADS_12, sampling_rate_hz=500.0, n_samples=5000, unit="mV"
)


def synthetic_ecg(
    record_index: int,
    leads: Tuple[str, ...] = LEADS_12,
    sampling_rate_hz: float = 500.0,
    n_samples: int = 5000,
) -> Signal:
    t = np.arange(n_samples, dtype=np.float64) / sampling_rate_hz
    heart_rate_hz = 1.0 + 0.05 * (record_index % 10)
    rows = []
    for lead_index in range(len(leads)):
        phase = 0.13 * lead_index + 0.07 * record_index
        amplitude = 0.5 + 0.1 * ((lead_index + record_index) % 5)
        wave = (
            amplitude * np.sin(2 * np.pi * heart_rate_hz * t + phase)
            + 0.30 * amplitude * np.sin(2 * np.pi * 3 * heart_rate_hz * t + 2 * phase)
            + 0.10 * amplitude * np.sin(2 * np.pi * 7 * heart_rate_hz * t + 3 * phase)
        )
        rows.append(wave)
    return Signal(
        data=np.ascontiguousarray(np.stack(rows), dtype=np.float32),
        leads=tuple(leads),
        sampling_rate_hz=sampling_rate_hz,
        unit="mV",
    )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_synthetic.py -v
```

Expected: PASS (4 passed)

- [ ] **Step 5: golden 생성 스크립트 작성**

`scripts/make_golden.py`:

```python
#!/usr/bin/env python3
"""Regenerate the golden expectation file.

Run this ONLY when a model's weights or runtime intentionally change, and
record why in the commit message. A golden diff you did not intend is a bug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mival.adapters import get_adapter
from mival.compiler import compile_recipe
from mival.modelcard import load_card
from mival.synthetic import GOLDEN_SOURCE, synthetic_ecg

N_RECORDS = 10


def probabilities(model_id: str, registry: Path) -> np.ndarray:
    card = load_card(registry / f"{model_id}.json")
    chain = compile_recipe(card.input_contract, GOLDEN_SOURCE)
    batch = np.stack(
        [chain.apply(synthetic_ecg(i)).data for i in range(N_RECORDS)]
    ).astype(np.float32)
    adapter = get_adapter(card.adapter)
    return adapter.forward(adapter.load(card), batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--registry", type=Path, default=Path("registry/models"))
    parser.add_argument("--out", type=Path, default=Path("tests/golden/expected.json"))
    args = parser.parse_args()

    probs = np.round(probabilities(args.model_id, args.registry), 6)
    body = json.loads(args.out.read_text()) if args.out.exists() else {}
    body[args.model_id] = {
        "n_records": N_RECORDS,
        "probabilities": [float(p) for p in probs],
        "sha256": hashlib.sha256(probs.tobytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps(body[args.model_id], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: golden test 작성**

`tests/test_golden.py`:

```python
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mival.adapters import get_adapter
from mival.compiler import compile_recipe
from mival.modelcard import load_card
from mival.synthetic import GOLDEN_SOURCE, synthetic_ecg

GOLDEN = Path(__file__).parent / "golden" / "expected.json"


def _probabilities(model_id, registry_dir):
    card = load_card(registry_dir / f"{model_id}.json")
    chain = compile_recipe(card.input_contract, GOLDEN_SOURCE)
    batch = np.stack(
        [chain.apply(synthetic_ecg(i)).data for i in range(10)]
    ).astype(np.float32)
    adapter = get_adapter(card.adapter)
    return adapter.forward(adapter.load(card), batch)


@pytest.mark.keras
@pytest.mark.weights("prophecg-stemi")
def test_prophecg_output_matches_golden(registry_dir):
    expected = json.loads(GOLDEN.read_text())["prophecg-stemi"]
    probs = np.round(_probabilities("prophecg-stemi", registry_dir), 6)
    assert hashlib.sha256(probs.tobytes()).hexdigest() == expected["sha256"]


def test_golden_file_records_every_model_with_a_forward_pass():
    body = json.loads(GOLDEN.read_text())
    assert "prophecg-stemi" in body
    assert len(body["prophecg-stemi"]["probabilities"]) == 10
```

- [ ] **Step 7: golden 값 생성 후 테스트 통과 확인**

keras27 환경에서:

```bash
python scripts/make_golden.py prophecg-stemi
pytest tests/test_golden.py -v
```

Expected: `make_golden.py`가 10개 확률과 sha256을 출력하고 `tests/golden/expected.json`에 기록. 이어서 PASS (2 passed).

ECGFounder는 STEMI head가 없어 `forward`가 없으므로 golden 대상이 아니다. Plan 4에서 head를 학습한 뒤 golden에 추가한다.

- [ ] **Step 8: 전체 테스트 실행**

```bash
pytest -v
```

Expected: core 테스트 전부 PASS. backend/weights 테스트는 해당 환경에서 PASS, 다른 환경에서는 SKIP.

- [ ] **Step 9: 커밋**

```bash
git add src/mival/synthetic.py scripts/make_golden.py tests/golden tests/test_golden.py tests/test_synthetic.py
git commit -m "feat: add deterministic synthetic ECG and golden output regression test"
```

---

## 완료 기준

- [ ] `pytest`가 core 환경에서 전부 통과한다 (backend 테스트는 skip)
- [ ] torch 환경에서 `pytest -m torch`가 통과한다
- [ ] keras27 환경에서 `pytest -m keras`가 통과한다
- [ ] `registry/models/`에 카드를 하나 더 넣는 것만으로 새 모델이 `load_registry`에 나타나고, 해당 카드의 입력 계약으로 recipe가 컴파일된다 — **C1 주장의 실행 가능한 증거**
- [ ] `docs/decisions/README.md`에 다음 open question을 추가한다: 두 카드의 `pretraining_corpora` 잠정값 확정, PROPHECG amplitude scaling 확정(현재 `scaling: none`으로 선언)

## 다음 plan으로 넘기는 것

| 항목 | 이유 | 이관 |
|---|---|---|
| `fit()`, `attribute()` | 학습은 split 동결과 leakage gate를 전제로 한다 | Plan 4 |
| perturbation op와 grid | preprocess stage의 sweep 실행과 함께 설계해야 한다 | Plan 3 |
| contamination gate | `pretraining_corpora` 확정과 cohort 출처 메타데이터가 필요하다 | Plan 4 |
| ECGFounder golden | STEMI head 학습 후에야 `forward`가 존재한다 | Plan 4 |
