# Task별 카드·study 초안 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** task 네 개(LVEF 분류, 회귀, 생존, 분할)의 모델 17개를 카드 초안으로, study 6개를 yaml 초안으로 적고, 지금 프레임워크 코드에 넣어 본 결과로 "표현 못 하는 것 목록"을 원장과 요약 문서에 남긴다.

**Architecture:** 코드는 바꾸지 않는다. 초안은 `registry/drafts/`와 `studies/drafts/`에 두어 기존 LVEF 실행(`registry/models/`, `studies/lvef/`)에 영향이 없게 한다. 판정은 scratchpad의 일회용 검사 스크립트가 `load_card`, `load_study`, stage별 spec 파서를 실제로 불러 얻은 결과와, 카드 내용을 기존 adapter·stage가 어떻게 읽는지에 대한 코드 근거로 내린다.

**Tech Stack:** JSON 카드, YAML study, Python 3.9 (`PYTHONPATH=src`), PyYAML. weight·torch·서버 불필요.

**Spec:** `docs/superpowers/specs/2026-10-06-task-draft-cards-design.md`. 후보 근거: `docs/model-search/task-candidates-2026-10-06.md`.

## Global Constraints

- `registry/models/`, `studies/lvef/study.yaml`, `src/`, `tests/`는 **수정하지 않는다**.
- weight 다운로드, 서버 접속, 추론을 하지 않는다.
- 확인하지 않은 값(sha256, 서버 `uri`, `code_commit`, runtime 버전)은 `null`로 두고 `notes`에 "unverified"를 쓴다. 추측한 값을 확인한 값처럼 적지 않는다.
- 지금 스키마로 표현 못 하는 것은 카드의 `x-mival-proposed`, study의 `x-proposed` 블록에 **이 모델에서 본 사실만** 적는다. 일반화된 스키마를 설계하지 않는다.
- 문서끼리 모순되는 값은 모순 그대로 `notes`에 적는다.
- 원장 형식은 `docs/decisions/adaptations.md`의 L2 항목 형식을 따른다 (제목, 설명, 발견, 상태 "보고됨. 만들지 않음", 착수 조건). 장황한 설명, 횟수 규칙을 넣지 않는다.
- 커밋 메시지 끝:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7
  ```

## Review Focus

1. **로드는 되는데 뜻이 틀린 카드를 "통과"로 판정하는 것.** loader는 모르는 키와 모르는 `adapter`/`layout` 값을 거부하지 않는다. 검사 스크립트가 adapter·layout·output.type을 지금 코드가 아는 값과 대조해 경고를 내야 하고, 판정은 그 경고와 코드 근거로 한다 (Task 1).
2. **초안 디렉터리가 기존 실행에 섞이는 것.** `registry/drafts/`가 `registry/models/`의 glob에 잡히거나 `studies/lvef`가 바뀌면 LVEF `config_hash`가 바뀐다. 각 task 끝에 `git diff --stat registry/models studies/lvef src tests`가 비어 있는지 확인한다 (모든 task의 마지막 검증 step).
3. **study 초안이 다른 디렉터리의 카드를 가리키는 것.** `lvef` 초안의 arm은 `registry/models`(기존 4개)와 `registry/drafts`(새 2개)를 동시에 쓴다. 검사 스크립트가 arm의 `model_id`를 study의 `models.registry` 하나에서만 찾아 없는 것을 보고해야 한다 (Task 1, Task 2).
4. **unverified 값이 확인값처럼 남는 것.** sha256이 `null`이 아닌데 출처가 없는 카드를 검사 스크립트가 셀 수 있어야 한다 (Task 1의 `sha256_known` 열).
5. **JSON·YAML 문법 오류로 판정이 아예 안 되는 것.** 검사 스크립트는 파싱 실패도 한 행으로 보고하고 계속 진행한다 (Task 1).

---

## File Structure

```
registry/drafts/                                  새 카드 17개 (Task 2~5)
  heartwise-lvef-under50.json  echonext-mini.json
  lima-ecg-age.json  singstad-ecg-age.json  kardionet-k-12lead.json
  vonbachmann-k.json  ai-ntprobnp.json
  ml4h-ecg2af.json  ml4h-ecg2hf.json  ml4h-ecg2stroke.json
  cavalab-deepsurv-code15.json  cavalab-mtlr-code15.json
  openecg-codec-v6.json  hrnetv2-delineation.json  semisegecg-resnet18.json
  semisegecg-vit-tiny.json  heartkit-seg-tcn.json
studies/drafts/<study>/study.yaml                 study 6개
  lvef  ecg-age  potassium  ntprobnp  mortality  delineation
docs/model-search/schema-gaps.md                  판정 요약 (Task 6)
docs/decisions/adaptations.md                     L2 항목 추가 (Task 6)
$SCRATCH/check_drafts.py                          검사 스크립트, repo 밖 (Task 1)
```

`$SCRATCH` = `/private/tmp/claude-503/-Users-minseongkim-Desktop-youlab-mi-va/2e33b0ed-5b03-4472-a899-6ac81cb71ea8/scratchpad`

---

### Task 1: 검사 스크립트

**Files:**
- Create: `$SCRATCH/check_drafts.py` (repo 밖)
- Create: `registry/drafts/.gitkeep`, `studies/drafts/.gitkeep`

**Interfaces:**
- Produces: `PYTHONPATH=src python3 $SCRATCH/check_drafts.py` → 카드 표와 study 표를 stdout에 출력. 이후 모든 task가 이 출력으로 검증한다.
  - 카드 행: `model_id | load | adapter_known | layout_known | output_type_known | proposed_keys | sha256_known | error`
  - study 행: `study | load | retrieve | profile | models | evaluate | arms_missing | proposed_keys | error`

- [ ] **Step 1: 빈 디렉터리 만들기**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
mkdir -p registry/drafts studies/drafts
touch registry/drafts/.gitkeep studies/drafts/.gitkeep
```

- [ ] **Step 2: 검사 스크립트 작성**

`$SCRATCH/check_drafts.py`:

```python
"""Throwaway checker for registry/drafts and studies/drafts (spec 2026-10-06).

Runs the framework's own parsers on each draft and flags values the parsers
accept but the runtime does not know. Prints two tables; never raises.
"""
import json
import sys
from pathlib import Path

from mival.adapters import available_adapters
from mival.modelcard import load_card
from mival.pipeline.study import load_study
from mival.stages.evaluate import DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS, _Settings
from mival.stages.models import ModelsSpec
from mival.stages.profile import ProfileSpec
from mival.stages.retrieve import RetrieveSpec

KNOWN_LAYOUTS = {"lead_time", "time_lead"}  # src/mival/adapters/base.py:35
KNOWN_OUTPUT_TYPES = {"softmax", "logits", "sigmoid", "regression"}  # torch_adapter.py:334


def card_row(path):
    row = {"model_id": path.stem, "load": "-", "adapter_known": "-", "layout_known": "-",
           "output_type_known": "-", "proposed_keys": "-", "sha256_known": "-", "error": ""}
    try:
        body = json.loads(path.read_text())
    except Exception as exc:  # parse failure is a row, not a crash
        row.update(load="parse_fail", error=str(exc)[:80])
        return row
    proposed = body.get("x-mival-proposed") or {}
    row["proposed_keys"] = ",".join(sorted(proposed)) or "none"
    ext = body.get("x-mival") or {}
    weights = ext.get("weights") or []
    row["sha256_known"] = f"{sum(1 for w in weights if w.get('sha256'))}/{len(weights)}"
    try:
        card = load_card(path)
    except Exception as exc:
        row.update(load="fail", error=str(exc)[:80])
        return row
    row["load"] = "ok"
    row["adapter_known"] = "yes" if card.adapter in available_adapters() else f"NO({card.adapter})"
    layout = card.input_contract.layout
    row["layout_known"] = "yes" if layout in KNOWN_LAYOUTS else f"NO({layout})"
    kind = card.output.get("type", "logits")
    row["output_type_known"] = "yes" if kind in KNOWN_OUTPUT_TYPES else f"NO({kind})"
    return row


def study_row(path):
    row = {"study": path.parent.name, "load": "-", "retrieve": "-", "profile": "-",
           "models": "-", "evaluate": "-", "arms_missing": "-", "proposed_keys": "-", "error": ""}
    try:
        study = load_study(path)
    except Exception as exc:
        row.update(load="fail", error=str(exc)[:80])
        return row
    row["load"] = "ok"
    proposed = set()
    for stage, spec in study.stages.items():
        if isinstance(spec, dict) and spec.get("x-proposed"):
            proposed.add(stage)
    row["proposed_keys"] = ",".join(sorted(proposed)) or "none"
    checks = {
        "retrieve": lambda s: RetrieveSpec.from_mapping(s),
        "profile": lambda s: ProfileSpec.from_mapping(s),
        "models": lambda s: ModelsSpec.from_mapping(s),
        "evaluate": lambda s: _Settings.from_spec(s, DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS),
    }
    errors = []
    for stage, check in checks.items():
        try:
            check(study.stage_spec(stage))
            row[stage] = "ok"
        except Exception as exc:
            row[stage] = "fail"
            errors.append(f"{stage}: {str(exc)[:60]}")
    models = study.stage_spec("models")
    registry = Path(str(models.get("registry", "registry/models")))
    have = set()
    for card_path in (registry.glob("*.json") if registry.is_dir() else []):
        try:
            have.add(json.loads(card_path.read_text()).get("model_id"))
        except Exception:
            have.add(card_path.stem)
    missing = sorted({str(a.get("model_id")) for a in models.get("arms") or [] if a.get("model_id") not in have})
    row["arms_missing"] = ",".join(missing) or "none"
    row["error"] = " | ".join(errors)
    return row


def table(rows):
    if not rows:
        print("(none)\n")
        return
    keys = list(rows[0])
    print(" | ".join(keys))
    for row in rows:
        print(" | ".join(str(row[k]) for k in keys))
    print()


if __name__ == "__main__":
    cards = sorted(Path("registry/drafts").glob("*.json"))
    studies = sorted(Path("studies/drafts").glob("*/study.yaml"))
    print(f"## cards ({len(cards)})")
    table([card_row(p) for p in cards])
    print(f"## studies ({len(studies)})")
    table([study_row(p) for p in studies])
    sys.exit(0)
```

- [ ] **Step 3: 빈 상태로 실행해 동작 확인**

Run: `cd /Users/minseongkim/Desktop/youlab/mi-va && PYTHONPATH=src python3 $SCRATCH/check_drafts.py`
Expected:
```
## cards (0)
(none)

## studies (0)
(none)
```

- [ ] **Step 4: 기존 카드로 스크립트가 맞게 판정하는지 확인 (positive control)**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
PYTHONPATH=src python3 - <<'EOF'
import sys; sys.argv=["x"]
sys.path.insert(0, "/private/tmp/claude-503/-Users-minseongkim-Desktop-youlab-mi-va/2e33b0ed-5b03-4472-a899-6ac81cb71ea8/scratchpad")
from pathlib import Path
from check_drafts import card_row, study_row, table
table([card_row(p) for p in sorted(Path("registry/models").glob("*.json"))])
table([study_row(Path("studies/lvef/study.yaml"))])
EOF
```
Expected: 기존 카드 5개 모두 `load=ok`, `adapter_known=yes`, `layout_known=yes`, `output_type_known=yes`. `lvef` study는 `retrieve/profile/models/evaluate` 모두 `ok`, `arms_missing=none`. 다르게 나오면 스크립트를 고친다 (기존 카드는 실행이 확인된 것이므로 스크립트 쪽 문제다).

- [ ] **Step 5: 기존 실행 무변경 확인 후 커밋**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
git diff --stat registry/models studies/lvef src tests   # 출력이 비어 있어야 한다
git add registry/drafts/.gitkeep studies/drafts/.gitkeep
git commit -m "chore: draft directories for cards and studies outside the live registry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7"
```

---

### Task 2: LVEF 분류 카드 2개와 `lvef` study 초안

**Files:**
- Create: `registry/drafts/heartwise-lvef-under50.json`, `registry/drafts/echonext-mini.json`
- Create: `studies/drafts/lvef/study.yaml`

**Interfaces:**
- Consumes: Task 1의 `check_drafts.py`
- Produces: 카드 2개, study 1개. Task 6이 판정한다.

근거 문서: `docs/models/heartwise-lvef.md`, `docs/models/echonext-mini.md`, 기존 카드 `registry/models/heartwise-lvef-binary.json`.

- [ ] **Step 1: `heartwise-lvef-under50.json`**

```json
{
  "model_id": "heartwise-lvef-under50",
  "Name": "HeartWise DeepECG-SL EfficientNetV2, LVEF < 50",
  "Summary": "Supervised single-task binary classifier for LVEF < 50%.",
  "Link": "https://huggingface.co/heartwise/EfficientNetV2_LVEF_under_50",
  "Descriptors": {"Version": "f3d8d94e703c", "References": ["doi:10.1093/eurheartj/ehaf1119"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 250 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": "/data/mi-val/models/heartwise-lvef/lvef_under_50.pt", "sha256": null, "role": "full"}],
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
    "notes": "Server-verified 2026-09-21 (torch.jit forward (1,12,2500)->(1,1)). sha256 truncated in docs/models/heartwise-lvef.md as 01cf98fe…129d7d95; full value is in /data/mi-val/manifests/models/heartwise-lvef.json, to fill when the server is up. Same input contract and preprocessing caveats as heartwise-lvef-binary (gain 1/0.0048; cohort-level spectral scaling L2-2 not applied). Pairs with label_def sens1 (value < 50)."
  }
}
```

- [ ] **Step 2: `echonext-mini.json`**

```json
{
  "model_id": "echonext-mini",
  "Name": "EchoNext-Mini (12-label structural heart disease)",
  "Summary": "Columbia ResNet1d with tabular features, 12 sigmoid labels; index 0 is lvef_lte_45.",
  "Link": "https://github.com/PierreElias/IntroECG",
  "Descriptors": {"Version": "15233e9392e9dc10c136a5624abf10199207bce9",
                  "References": ["doi:10.1056/AIdbp2500516", "doi:10.1038/s41586-025-09227-0"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 250 Hz, plus 7 tabular features"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": "/data/mi-val/models/echonext-mini/weights.pt", "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "state_dict_key": "model",
    "builder": {"module": "cradlenet.models.resnet1d_tabular:ResNet1dWithTabular",
                "kwargs": {"len_tabular_feature_vector": 7, "num_classes": 12, "filter_size": 16}},
    "code_path": "/opt/introecg",
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 250, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "per_lead_zscore", "layout": "time_lead", "dtype": "float32"
    },
    "output": {"type": "sigmoid", "n_outputs": 12, "positive_index": 0},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": "torch==2.13.0+cu130", "python": "3.10.20", "device": "cuda",
                "env": "/data/mi-val/envs/ecgfounder", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["columbia"],
    "code_commit": "15233e9392e9dc10c136a5624abf10199207bce9",
    "notes": "Server-verified 2026-09-21: strict load, (1,1,2500,12)+tabular(1,7) forward -> (1,12). Repo has no LICENSE file. sha256 truncated in docs (aac57aad…e21cd6). layout here is the closest existing value; the real tensor is (B,1,2500,12) and (B,12,2500) is rejected. per_lead_zscore is approximate: the authors clip per lead and z-score with fixed per-lead mean/std from waveform_normalization_params.json, after baseline-wander removal. The tabular input cannot be expressed (existing L2-3)."
  },
  "x-mival-proposed": {
    "input_layout": {"shape": ["batch", 1, "time", "lead"], "note": "4-D tensor with a singleton channel axis"},
    "waveform_preprocessing": [
      {"op": "baseline_wander_removal", "source": "parse_xml.py"},
      {"op": "per_lead_clip_then_zscore", "params_file": "waveform_normalization_params.json", "params_sha256": null}
    ],
    "tabular_inputs": {
      "order": ["sex_male", "age_at_ecg", "ventricular_rate", "atrial_rate", "pr_interval", "qrs_duration", "qt_corrected"],
      "missing_as_zero": ["atrial_rate", "pr_interval"],
      "transformer": {"file": "tabular_transformer.joblib", "kind": "sklearn StandardScaler + median impute", "sklearn": "1.1.3"},
      "mimic_source": "machine_measurements.csv (RR, PR, QRS, QT, QTc) + person age/sex; atrial_rate mapping unconfirmed"
    },
    "label_cutoff": {"lvef_lte_45": "LVEF <= 45%", "note": "retrieve has only primary (<=40) and sens1 (<50)"},
    "exclusions": ["age < 18", "poor quality flag", "ventricular pacing", "missing age or sex", "all measurements missing"]
  }
}
```

- [ ] **Step 3: `studies/drafts/lvef/study.yaml`**

기존 `studies/lvef/study.yaml`을 복사한 뒤 아래만 바꾼다.

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
mkdir -p studies/drafts/lvef
cp studies/lvef/study.yaml studies/drafts/lvef/study.yaml
```

그다음 `studies/drafts/lvef/study.yaml`에서:
1. 맨 위 `study_id: lvef`를 `study_id: lvef-draft`로 바꾼다.
2. `stages.retrieve` 블록 끝에 추가:
```yaml
    x-proposed:
      extra_cutoffs:
        - {name: lte45, rule: "value <= 45", used_by: echonext-mini}
      note: >-
        Cutoffs are fixed to primary (<= primary_cutoff) and sens1 (< sens1_cutoff)
        in RetrieveSpec (src/mival/stages/retrieve.py:43). A model trained on a
        third cutoff has no label column to score against.
```
3. `stages.models.registry: registry/models`를 `registry: registry/drafts`로 바꾸고, `arms` 끝에 추가:
```yaml
      - {model_id: heartwise-lvef-under50, training_mode: inference_only, label_def: sens1}
      - {model_id: echonext-mini, training_mode: inference_only, label_def: primary}  # true cutoff is <=45, see x-proposed
```
4. `stages.models` 블록 끝에 추가:
```yaml
    x-proposed:
      registries: [registry/models, registry/drafts]
      note: >-
        The six existing arms live in registry/models and the two new ones in
        registry/drafts. models.registry and preprocess.registry each take one
        directory, and preprocess compiles every card in it
        (src/mival/stages/preprocess.py:250).
```

- [ ] **Step 4: 검사 실행**

Run: `cd /Users/minseongkim/Desktop/youlab/mi-va && PYTHONPATH=src python3 $SCRATCH/check_drafts.py`
Expected:
- 카드 `heartwise-lvef-under50`: `load=ok`, 모든 `_known=yes`, `proposed_keys=none`, `sha256_known=0/1`
- 카드 `echonext-mini`: `load=ok`, `adapter_known=yes`, `layout_known=yes`, `output_type_known=yes`, `proposed_keys=exclusions,input_layout,label_cutoff,tabular_inputs,waveform_preprocessing`
- study `lvef`: `retrieve/profile/models/evaluate=ok`, `arms_missing=ecgfounder,heartwise-lvef-binary,heartwise-lvef-regression,xecg`, `proposed_keys=models,retrieve`

`arms_missing`이 위와 같으면 Review Focus 3번 gap이 재현된 것이다. 출력 전체를 `$SCRATCH/check_task2.txt`에 저장한다:
`PYTHONPATH=src python3 $SCRATCH/check_drafts.py > $SCRATCH/check_task2.txt`

- [ ] **Step 5: 무변경 확인 후 커밋**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
git diff --stat registry/models studies/lvef src tests   # 비어 있어야 한다
git add registry/drafts/heartwise-lvef-under50.json registry/drafts/echonext-mini.json studies/drafts/lvef/study.yaml
git commit -m "docs: draft cards for HeartWise LVEF<50 and EchoNext-Mini, and an lvef study draft

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7"
```

---

### Task 3: 회귀 카드 5개와 study 3개 (`ecg-age`, `potassium`, `ntprobnp`)

**Files:**
- Create: `registry/drafts/{lima-ecg-age,singstad-ecg-age,kardionet-k-12lead,vonbachmann-k,ai-ntprobnp}.json`
- Create: `studies/drafts/{ecg-age,potassium,ntprobnp}/study.yaml`

**Interfaces:**
- Consumes: Task 1의 `check_drafts.py`
- Produces: 카드 5개, study 3개. Task 6이 판정한다.

모든 값의 출처: `docs/model-search/task-candidates-2026-10-06.md` Regression 절.

- [ ] **Step 1: `lima-ecg-age.json`**

```json
{
  "model_id": "lima-ecg-age",
  "Name": "Lima et al. ECG-age (CODE ResNet)",
  "Summary": "12-lead ResNet regressing chronological age in years.",
  "Link": "https://github.com/antonior92/ecg-age-prediction",
  "Descriptors": {"Version": "zenodo:4892365", "References": ["doi:10.1038/s41467-021-25351-7"]},
  "Model properties": {"Input": "12-lead surface ECG, 4096 samples at 400 Hz (10 s centred, zero-padded)"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "state_dict_key": "model",
    "builder": {"module": "resnet:ResNet1d", "kwargs": {}},
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 400, "duration_s": 10.24, "unit": "mV", "filters": [],
      "scaling": "none", "gain": 10.0, "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "regression", "n_outputs": 1, "unit": "years"},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["code"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha, builder kwargs (read config.json in model.zip), code_commit. Weights CC-BY-4.0, code MIT. Input unit contradictory in README ('scale 1e-4 V' vs 'multiply V by 1000'); gain 10 (0.1 mV units) is inferred from CODE-15 per-lead SDs and a one-ECG test (mV 79.1 y, mV×10 47.7 y, mV×1000 -390 y). 10 s are centred and zero-padded to 4096 samples; duration_s 10.24 encodes 4096/400 but the model sees 10 s of signal."
  },
  "x-mival-proposed": {
    "padding": {"to_samples": 4096, "signal_samples": 4000, "mode": "zero, centred"},
    "unit_ambiguity": {"candidates": ["mV", "mV*10", "mV*1000"], "chosen": "mV*10", "evidence": "inferred, see notes"}
  }
}
```

- [ ] **Step 2: `singstad-ecg-age.json`**

```json
{
  "model_id": "singstad-ecg-age",
  "Name": "Singstad ECG-age (Inception, both genders)",
  "Summary": "12-lead Inception network regressing age in years.",
  "Link": "https://github.com/Bsingstad/ECG-age",
  "Descriptors": {"Version": null, "References": ["doi:10.1101/2022.10.03.22280640"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 100 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "keras",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 100, "duration_s": 10, "unit": "uV", "filters": [],
      "scaling": "none", "layout": "time_lead", "dtype": "int32"
    },
    "output": {"type": "regression", "n_outputs": 1, "unit": "years"},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cpu", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["physionet-challenge-2021"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (model_weights/model_weights_inception_both genders.h5), keras version, code_commit. No license file on the 12-lead repo. Reference code feeds raw digital units (~uV) and pad_sequences casts to int32; dtype int32 records that cast. Workshop paper (NLDL 2022), medRxiv version cited."
  }
}
```

- [ ] **Step 3: `kardionet-k-12lead.json`**

```json
{
  "model_id": "kardionet-k-12lead",
  "Name": "Kardio-Net 12-lead serum potassium",
  "Summary": "EfficientNet regressing serum potassium (mEq/L) from 12-lead ECG.",
  "Link": "https://github.com/ecg-net/hyperkalemia",
  "Descriptors": {"Version": null, "References": ["doi:10.1016/j.jacep.2024.07.023"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 500 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "state_dict_key": "state_dict",
    "builder": {"module": null, "kwargs": {}},
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 500, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "per_lead_zscore", "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "regression", "n_outputs": 1, "unit": "mEq/L"},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["cedars-sinai"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (model_12_lead.pt), PyTorch Lightning builder, code_commit, whether the 12-lead checkpoint is the general or the dialysis fine-tune. 12-lead preprocessing undocumented; the single-lead pipeline does baseline removal, wavelet denoising, then lead-wise z-score with mean/SD computed on the target dataset. ecg_denoising() returns inside its loop (processes only the first file). Output assumed raw mEq/L (MSE loss)."
  },
  "x-mival-proposed": {
    "dataset_level_normalization": {"op": "per_lead_zscore", "statistics_from": "target cohort", "see": "L2-2"},
    "denoising": {"op": "wavelet", "source": "single-lead pipeline, undocumented for 12-lead"}
  }
}
```

- [ ] **Step 4: `vonbachmann-k.json`**

```json
{
  "model_id": "vonbachmann-k",
  "Name": "von Bachmann et al. electrolyte regression, potassium",
  "Summary": "8-lead ResNet regressing z-scored serum potassium; 5 trained models.",
  "Link": "https://github.com/philippvb/ecg-electrolyte-regression",
  "Descriptors": {"Version": "zenodo:7456316", "References": ["doi:10.1038/s41598-024-65223-w"]},
  "Model properties": {"Input": "8-lead surface ECG (I, II, V1-V6), 4096 samples at 400 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"}
    ],
    "weights_format": "checkpoint",
    "builder": {"module": null, "kwargs": {}},
    "ensemble": {"method": "mean_probability", "members": 5},
    "input_contract": {
      "leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 400, "duration_s": 10.24, "unit": "mV",
      "filters": [{"kind": "highpass", "cutoff_hz": 0.5, "order": 1}, {"kind": "notch", "cutoff_hz": 50, "order": 1}],
      "scaling": "none", "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "regression", "n_outputs": 1, "unit": "z-score of mmol/L"},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["swedish-ed"],
    "code_commit": null,
    "notes": "unverified: all 5 weight uris/shas (regression_models/potassium/model_{0-4}/model.pth inside a 7.5 GB zip), builder, filter cutoffs and orders (high-pass and notch are documented, parameters are not), input unit. The paper's median model is #3; ensemble 'mean_probability' is the nearest existing method but this is a mean of regression outputs. Targets were z-scored (normalize: true) and the training mean/SD are not shipped."
  },
  "x-mival-proposed": {
    "output_inverse_transform": {"kind": "zscore", "mean": 3.99, "sd": 0.50, "unit": "mmol/L",
                                 "source": "paper Table 1 cohort values, not the training statistics"},
    "ensemble_method": {"name": "mean_regression", "note": "existing method name says probability"}
  }
}
```

- [ ] **Step 5: `ai-ntprobnp.json`**

```json
{
  "model_id": "ai-ntprobnp",
  "Name": "AI-NT-proBNP (Lima-style ResNet, 5-fold)",
  "Summary": "12-lead ResNet regressing log NT-proBNP; 5-fold ensemble.",
  "Link": "https://github.com/JanBrem/AI-NT-proBNP",
  "Descriptors": {"Version": null, "References": ["doi:10.1515/cclm-2023-0743"]},
  "Model properties": {"Input": "12-lead surface ECG, 2048 samples at 250 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"},
      {"uri": null, "sha256": null, "role": "ensemble_member"}
    ],
    "weights_format": "checkpoint",
    "builder": {"module": null, "kwargs": {}},
    "ensemble": {"method": "mean_probability", "members": 5},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 250, "duration_s": 8.192, "unit": "mV",
      "filters": [{"kind": "highpass", "cutoff_hz": 0.5, "order": 1}, {"kind": "notch", "cutoff_hz": 50, "order": 1}],
      "scaling": "per_lead_zscore", "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "regression", "n_outputs": 1, "unit": "log NT-proBNP"},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["hamburg-city-health-study"],
    "code_commit": null,
    "notes": "unverified: weights (Google Drive weights.zip, 380 MB, id 1dT4N-2PQzp2UxWpuGLZgzkPnys0FKCKI), builder, filter orders, log base. No license in the repo. Preprocessing: 500 Hz downsampled 2x to 250 Hz, 0.5 Hz high-pass, 50 Hz notch, keep the first 2048 samples, per-lead z-score then subtract the median. Sample outputs 1.6-5.5 suggest ln(pg/mL)."
  },
  "x-mival-proposed": {
    "crop": {"keep": "first", "samples": 2048},
    "post_zscore": {"op": "subtract_median_per_lead"},
    "output_inverse_transform": {"kind": "exp", "base": "e (unverified)", "unit": "pg/mL"},
    "ensemble_method": {"name": "mean_regression", "note": "existing method name says probability"}
  }
}
```

- [ ] **Step 6: study 3개**

`studies/drafts/ecg-age/study.yaml`:

```yaml
study_id: ecg-age-draft
site: dicom-miva
seed: 20261006

stages:
  retrieve:
    dsn_env: /data/mi-val/secrets/micdm.env
    schema: cdm
    modality_concept_id: 4145308
    local_path_root: /scratch/mi-val/dicom
    require_local_file: true
    window_days: 0
    window_days_sens2: 0
    implausible_below: 18
    x-proposed:
      label_source: {kind: derived, from: "person.year_of_birth / birth_datetime", value: "age at index_datetime in years"}
      note: >-
        RetrieveSpec reads one measurement concept near each ECG
        (label_concept_id, window_days). Age has no measurement row; it is
        computed from the person table at the ECG's own time. label_concept_id
        is left at its LVEF default here, which would silently pull LVEF.

  profile:
    test_fraction: 0.2
    n_folds: 5
    stratify_on: label_value
    min_test_positives: 0

  preprocess:
    loader: dicom
    registry: registry/drafts
    allow_upsample: false
    pad_policy: reject

  models:
    registry: registry/drafts
    cohort_sources: [mimic-iv-ecg]
    batch_size: 64
    arms:
      - {model_id: lima-ecg-age, training_mode: inference_only, label_def: value}
      - {model_id: singstad-ecg-age, training_mode: inference_only, label_def: value}

  evaluate:
    outcomes: {primary: null}
    regression_cuts: []
    bootstrap_replicates: 2000
    figures: true
```

`studies/drafts/potassium/study.yaml`:

```yaml
study_id: potassium-draft
site: dicom-miva
seed: 20261006

stages:
  retrieve:
    dsn_env: /data/mi-val/secrets/micdm.env
    schema: cdm
    label_concept_id: null
    modality_concept_id: 4145308
    local_path_root: /scratch/mi-val/dicom
    require_local_file: true
    window_days: 0
    window_days_sens2: 1
    implausible_below: 1.5
    primary_cutoff: 5.5
    sens1_cutoff: 6.0
    x-proposed:
      label_concept_id: "serum potassium (LOINC 2823-3) concept id, not looked up"
      window: {before_hours: 1, after_hours: 1, note: "window_days is whole days; a lab drawn hours from the ECG needs a sub-day window"}
      cutoff_direction: {rule: "value >= cutoff", note: "primary_cutoff means value <= cutoff (low LVEF); hyperkalemia is high values"}

  profile:
    test_fraction: 0.2
    n_folds: 5
    stratify_on: label_primary
    min_test_positives: 100

  preprocess:
    loader: dicom
    registry: registry/drafts
    allow_upsample: false
    pad_policy: reject

  models:
    registry: registry/drafts
    cohort_sources: [mimic-iv-ecg]
    batch_size: 64
    arms:
      - {model_id: kardionet-k-12lead, training_mode: inference_only, label_def: value}
      - {model_id: vonbachmann-k, training_mode: inference_only, label_def: value}

  evaluate:
    outcomes: {primary: null}
    regression_cuts: [5.5]
    bootstrap_replicates: 2000
    figures: true
```

`studies/drafts/ntprobnp/study.yaml`:

```yaml
study_id: ntprobnp-draft
site: dicom-miva
seed: 20261006

stages:
  retrieve:
    dsn_env: /data/mi-val/secrets/micdm.env
    schema: cdm
    label_concept_id: null
    modality_concept_id: 4145308
    local_path_root: /scratch/mi-val/dicom
    require_local_file: true
    window_days: 1
    window_days_sens2: 7
    implausible_below: 0
    primary_cutoff: 125
    sens1_cutoff: 300
    x-proposed:
      label_concept_id: "NT-proBNP (LOINC 33762-6) concept id, not looked up"
      label_transform: {kind: log, base: e, note: "the model predicts log NT-proBNP; the label must be on the same scale or the output inverted"}
      cutoff_direction: {rule: "value >= cutoff"}

  profile:
    test_fraction: 0.2
    n_folds: 5
    stratify_on: label_primary
    min_test_positives: 100

  preprocess:
    loader: dicom
    registry: registry/drafts
    allow_upsample: false
    pad_policy: reject

  models:
    registry: registry/drafts
    cohort_sources: [mimic-iv-ecg]
    batch_size: 64
    arms:
      - {model_id: ai-ntprobnp, training_mode: inference_only, label_def: value}

  evaluate:
    outcomes: {primary: null}
    regression_cuts: [125]
    bootstrap_replicates: 2000
    figures: true
```

- [ ] **Step 7: 검사 실행**

Run: `cd /Users/minseongkim/Desktop/youlab/mi-va && PYTHONPATH=src python3 $SCRATCH/check_drafts.py > $SCRATCH/check_task3.txt; cat $SCRATCH/check_task3.txt`
Expected:
- 카드 5개 모두 행이 있다 (`parse_fail`이 없다). `sha256_known`은 `0/1` 또는 `0/5`.
- `potassium`, `ntprobnp`: `label_concept_id: null` 때문에 `retrieve=fail`일 가능성이 높다. 실패 메시지를 그대로 기록한다 (기대된 gap).
- `ecg-age`: `retrieve`가 `ok`로 나오면 그것이 Review Focus 1번의 사례다 (LVEF concept 기본값을 조용히 씀). 기록한다.
- 결과가 위와 다르면 카드·yaml을 고치지 말고 그대로 기록한다. JSON/YAML 문법 오류만 고친다.

- [ ] **Step 8: 무변경 확인 후 커밋**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
git diff --stat registry/models studies/lvef src tests   # 비어 있어야 한다
git add registry/drafts/lima-ecg-age.json registry/drafts/singstad-ecg-age.json registry/drafts/kardionet-k-12lead.json registry/drafts/vonbachmann-k.json registry/drafts/ai-ntprobnp.json studies/drafts/ecg-age studies/drafts/potassium studies/drafts/ntprobnp
git commit -m "docs: draft cards for five regression models and age, potassium, NT-proBNP studies

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7"
```

---

### Task 4: 생존 카드 5개와 `mortality` study

**Files:**
- Create: `registry/drafts/{ml4h-ecg2af,ml4h-ecg2hf,ml4h-ecg2stroke,cavalab-deepsurv-code15,cavalab-mtlr-code15}.json`
- Create: `studies/drafts/mortality/study.yaml`

**Interfaces:**
- Consumes: Task 1의 `check_drafts.py`
- Produces: 카드 5개, study 1개. Task 6이 판정한다.

모든 값의 출처: `docs/model-search/task-candidates-2026-10-06.md` Survival 절.

각 카드의 `x-mival.output`에는 지금 스키마 안에서 가장 가까운 값(`logits`, 사망 head 하나)을 적어 "로드되지만 뜻이 틀림"을 재현하고, 실제 출력은 `x-mival-proposed`에 적는다.

- [ ] **Step 1: `ml4h-ecg2af.json`**

```json
{
  "model_id": "ml4h-ecg2af",
  "Name": "ml4h ECG2AF (2024 five-head)",
  "Summary": "12-lead network with incident-AF and death survival heads plus sex, age and prevalent-AF heads.",
  "Link": "https://github.com/broadinstitute/ml4h/tree/master/model_zoo/ECG2AF",
  "Descriptors": {"Version": "ecg2af_quintuplet_v2024_01_13", "References": ["doi:10.1161/CIRCULATIONAHA.121.057480"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 500 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "keras",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 500, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "global_zscore", "layout": "time_lead", "dtype": "float32"
    },
    "output": {"type": "sigmoid", "n_outputs": 50, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": "keras==3.9.0", "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["mgh"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (ecg2af_quintuplet_v2024_01_13.keras, 43.8 MB, Git LFS), python version. Keras 3 .keras zip format per metadata.json; the project keras env is 2.7. GPL-3.0 (repo-level). The x-mival.output above is the nearest existing value for the death head alone and is wrong in meaning: a sigmoid over 50 units read as class probabilities. The death head was added in a 2025 update and has no paper of its own."
  },
  "x-mival-proposed": {
    "heads": [
      {"name": "output_survival_curve_af_survival_curve", "shape": [50], "activation": "sigmoid", "kind": "survival_curve", "endpoint": "incident AF", "intervals": 25, "window_days": 1825},
      {"name": "output_sex_categorical", "shape": [2], "activation": "softmax", "kind": "categorical", "classes": ["female", "male"]},
      {"name": "output_death_event_survival_curve", "shape": [50], "activation": "sigmoid", "kind": "survival_curve", "endpoint": "all-cause death", "intervals": 25, "window_days": 3650},
      {"name": "output_age_in_days_continuous", "shape": [1], "activation": "linear", "kind": "regression", "unit": "unverified"},
      {"name": "output_af_in_read_categorical", "shape": [2], "activation": "softmax", "kind": "categorical", "classes": ["no_af", "af_in_read"]}
    ],
    "use_head": "output_death_event_survival_curve",
    "survival_curve_decoding": {
      "meaningful_units": "first 25 of 50",
      "unit_meaning": "conditional probability of surviving interval i given survival to its start",
      "survival": "S(t_k) = prod_{i<=k} p_i",
      "bin_days": "window_days / intervals (code: 73 for AF, 146 for death); README writes 1 + window_days // intervals",
      "risk_at_horizon": "1 - S(horizon)"
    }
  }
}
```

- [ ] **Step 2: `ml4h-ecg2hf.json`**

```json
{
  "model_id": "ml4h-ecg2hf",
  "Name": "ml4h ECG2HF (five-head)",
  "Summary": "12-lead network with two incident-HF survival heads, a death survival head, and sex and age heads.",
  "Link": "https://github.com/broadinstitute/ml4h/tree/master/model_zoo/ECG2HF",
  "Descriptors": {"Version": "ecg_5000_hf_quintuplet_dropout_v2023_04_17", "References": ["doi:10.1161/CIRCHEARTFAILURE.125.013927"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 500 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "keras",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 500, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "global_zscore", "layout": "time_lead", "dtype": "float32"
    },
    "output": {"type": "sigmoid", "n_outputs": 50, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": "keras==3", "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["mgh"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (33.2 MB .keras), exact keras version. Broad Academic Software License: academic/non-profit use only, acknowledgement required, covers model outputs. Tested at MGH, BWH and BIDMC (MIMIC's source hospital) in the paper, but not trained on MIMIC. x-mival.output is the nearest existing value and is wrong in meaning (see ml4h-ecg2af)."
  },
  "x-mival-proposed": {
    "heads": [
      {"name": "output_is_male_categorical", "shape": [2], "activation": "softmax", "kind": "categorical"},
      {"name": "output_hf_primary_event_survival_curve", "shape": [50], "activation": "sigmoid", "kind": "survival_curve", "endpoint": "HF by primary ICD", "intervals": 25, "window_days": 3650},
      {"name": "output_hf_nlp_event_survival_curve", "shape": [50], "activation": "sigmoid", "kind": "survival_curve", "endpoint": "HF by NLP (primary task)", "intervals": 25, "window_days": 3650},
      {"name": "output_death_event_survival_curve", "shape": [50], "activation": "sigmoid", "kind": "survival_curve", "endpoint": "all-cause death", "intervals": 25, "window_days": 3650},
      {"name": "output_age_in_days_continuous", "shape": [1], "activation": "linear", "kind": "regression", "unit": "unverified"}
    ],
    "use_head": "output_death_event_survival_curve",
    "survival_curve_decoding": {"same_as": "ml4h-ecg2af", "bin_days": 146},
    "license_scope": "academic only; applies to outputs"
  }
}
```

- [ ] **Step 3: `ml4h-ecg2stroke.json`**

```json
{
  "model_id": "ml4h-ecg2stroke",
  "Name": "ml4h ECG2Stroke",
  "Summary": "12-lead network with incident-stroke and death survival heads plus sex, age and AF heads.",
  "Link": "https://github.com/broadinstitute/ml4h/tree/master/model_zoo/ECG2Stroke",
  "Descriptors": {"Version": "ecg2stroke_dropout_2024_10_04_10_49_43", "References": ["pmid:42126358"]},
  "Model properties": {"Input": "12-lead surface ECG, 10 seconds at 500 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "keras",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 500, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "global_zscore", "layout": "time_lead", "dtype": "float32"
    },
    "output": {"type": "sigmoid", "n_outputs": 50, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["mgh"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (225 MB .h5, not opened), keras version, head order. README says per-lead z-score, code uses whole-ECG z-score (global_zscore chosen, conflict unresolved). README bin width uses 2*days_window unlike ECG2AF/ECG2HF (unresolved). The published score is a Cox model on the network output plus age and sex whose coefficients are not shipped. BIDMC was an external test site in the paper. x-mival.output is the nearest existing value and is wrong in meaning."
  },
  "x-mival-proposed": {
    "heads": [
      {"name": "stroke survival curve", "shape": [50], "kind": "survival_curve", "endpoint": "incident ischaemic stroke", "window_days": 3650},
      {"name": "death survival curve", "shape": [50], "kind": "survival_curve", "endpoint": "all-cause death", "window_days": 3650},
      {"name": "sex", "kind": "categorical"},
      {"name": "age", "kind": "regression"},
      {"name": "af", "kind": "categorical"}
    ],
    "use_head": "death survival curve",
    "head_names": "unverified (file not opened)",
    "survival_curve_decoding": {"same_as": "ml4h-ecg2af", "bin_days": "conflict: window/intervals vs 2*window in README"},
    "published_score": {"kind": "cox_on_output", "covariates": ["age", "sex"], "coefficients": "not shipped"},
    "normalization_conflict": ["per_lead_zscore (README)", "global_zscore (code)"]
  }
}
```

- [ ] **Step 4: `cavalab-deepsurv-code15.json`**

```json
{
  "model_id": "cavalab-deepsurv-code15",
  "Name": "cavalab Code-15 ResNet DeepSurv (no demographics)",
  "Summary": "Ribeiro ResNet1d with a fusion MLP emitting one Cox log-risk for all-cause mortality.",
  "Link": "https://github.com/cavalab/ecg-survival-benchmark",
  "Descriptors": {"Version": "zenodo:16877773", "References": ["arXiv:2406.17002", "pmid:41408648"]},
  "Model properties": {"Input": "12-lead surface ECG, middle 7 s at 400 Hz zero-padded to 4096 samples"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "state_dict_key": "model_state_dict",
    "builder": {"module": null, "kwargs": {}},
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 400, "duration_s": 7, "unit": "mV", "filters": [],
      "scaling": "per_lead_zscore", "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "regression", "n_outputs": 1},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["code-15"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha ('Code15 ResNet DeepSurv no Dem.pt', 30.5 MB), builder, input unit (CODE-15 native scale; check mV vs 1e-4 V). Code GPL-3.0, weights CC-BY-4.0. Per-lead z-score uses the checkpoint's stored NM/NS, not the cohort's own statistics. x-mival.output 'regression' is the nearest existing value: a log-risk read as a continuous target, wrong in meaning. No baseline hazard is shipped."
  },
  "x-mival-proposed": {
    "output": {"kind": "cox_log_risk", "baseline_hazard": "not shipped; fit (Breslow) on a cohort = recalibration, or report discrimination only"},
    "normalization": {"op": "per_lead_zscore", "statistics_from": "checkpoint keys NM, NS"},
    "crop_and_pad": {"keep": "middle 2800 samples (7 s)", "pad": "648 zeros each side", "to_samples": 4096}
  }
}
```

- [ ] **Step 5: `cavalab-mtlr-code15.json`**

```json
{
  "model_id": "cavalab-mtlr-code15",
  "Name": "cavalab Code-15 ResNet MTLR (with demographics)",
  "Summary": "Ribeiro ResNet1d with a fusion MLP emitting 100 MTLR logits for all-cause mortality; takes age and sex.",
  "Link": "https://github.com/cavalab/ecg-survival-benchmark",
  "Descriptors": {"Version": "zenodo:16877773", "References": ["arXiv:2406.17002", "pmid:41408648"]},
  "Model properties": {"Input": "12-lead surface ECG, middle 7 s at 400 Hz zero-padded to 4096 samples, plus age and sex"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "state_dict_key": "model_state_dict",
    "builder": {"module": null, "kwargs": {}},
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
      "sampling_rate_hz": 400, "duration_s": 7, "unit": "mV", "filters": [],
      "scaling": "per_lead_zscore", "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "softmax", "n_outputs": 100, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["code-15"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha ('Code15 ResNet MTLR w Dem.pt', 30.6 MB), builder, covariate scaling (Age assumed raw years). Same input caveats as cavalab-deepsurv-code15. x-mival.output 'softmax' over 100 is the nearest existing value and is wrong in meaning: MTLR logits need the pycox MTLR transform, not a softmax over classes."
  },
  "x-mival-proposed": {
    "output": {"kind": "mtlr_logits", "n_bins": 100, "cuts": "pycox label_transform(100), equidistant on [0, max_duration]", "max_duration_years": 7.677, "decode": "pycox MTLR predict_surv"},
    "covariate_inputs": {"order": ["Age", "Is_Male"], "scaling": "unverified"},
    "normalization": {"op": "per_lead_zscore", "statistics_from": "checkpoint keys NM, NS"},
    "crop_and_pad": {"keep": "middle 2800 samples (7 s)", "pad": "648 zeros each side", "to_samples": 4096}
  }
}
```

- [ ] **Step 6: `studies/drafts/mortality/study.yaml`**

```yaml
study_id: mortality-draft
site: dicom-miva
seed: 20261006

stages:
  retrieve:
    dsn_env: /data/mi-val/secrets/micdm.env
    schema: cdm
    label_concept_id: null
    modality_concept_id: 4145308
    local_path_root: /scratch/mi-val/dicom
    require_local_file: true
    window_days: 0
    window_days_sens2: 0
    implausible_below: 0
    x-proposed:
      label_source: {kind: time_to_event, event_table: death, event_date: death_date,
                     censor_at: "last observation_period_end_date", time_origin: index_datetime, unit: days}
      columns: [event_observed, time_to_event_days]
      note: >-
        RetrieveSpec produces label_value and binary label_* columns from one
        measurement near the ECG. Survival needs an event flag and a follow-up
        time with censoring, neither of which the cohort_index has.

  profile:
    test_fraction: 0.2
    n_folds: 5
    stratify_on: label_primary
    min_test_positives: 100
    x-proposed:
      stratify_on: event_observed

  preprocess:
    loader: dicom
    registry: registry/drafts
    allow_upsample: false
    pad_policy: reject

  models:
    registry: registry/drafts
    cohort_sources: [mimic-iv-ecg]
    batch_size: 64
    arms:
      - {model_id: ml4h-ecg2af, training_mode: inference_only, label_def: primary}
      - {model_id: ml4h-ecg2hf, training_mode: inference_only, label_def: primary}
      - {model_id: ml4h-ecg2stroke, training_mode: inference_only, label_def: primary}
      - {model_id: cavalab-deepsurv-code15, training_mode: inference_only, label_def: value}
      - {model_id: cavalab-mtlr-code15, training_mode: inference_only, label_def: primary}
    x-proposed:
      label_def: survival
      note: >-
        label_def must name an existing label column (models.py:216), so every
        arm here borrows primary or value. inference_only also requires the
        card's output.type to match regression vs classification
        (models.py:771); a survival arm is neither.

  evaluate:
    outcomes: {primary: null}
    bootstrap_replicates: 2000
    figures: true
    x-proposed:
      metrics: [harrell_c_index, uno_c_index, time_dependent_auc_at_horizon, brier_at_horizon]
      horizons_years: [1, 5, 10]
      note: "evaluate has binary and regression categories only"
```

- [ ] **Step 7: 검사 실행**

Run: `cd /Users/minseongkim/Desktop/youlab/mi-va && PYTHONPATH=src python3 $SCRATCH/check_drafts.py > $SCRATCH/check_task4.txt; cat $SCRATCH/check_task4.txt`
Expected:
- 생존 카드 5개 모두 `load=ok`, `output_type_known=yes` (지금 스키마 값으로 적었으므로). 이것이 판정 2(로드되지만 뜻이 틀림)의 재현이다.
- `mortality`: `label_concept_id: null`이면 `retrieve=fail`. 기록한다.
- JSON/YAML 문법 오류만 고친다.

- [ ] **Step 8: 무변경 확인 후 커밋**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
git diff --stat registry/models studies/lvef src tests   # 비어 있어야 한다
git add registry/drafts/ml4h-ecg2af.json registry/drafts/ml4h-ecg2hf.json registry/drafts/ml4h-ecg2stroke.json registry/drafts/cavalab-deepsurv-code15.json registry/drafts/cavalab-mtlr-code15.json studies/drafts/mortality
git commit -m "docs: draft cards for five survival models and a mortality study

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7"
```

---

### Task 5: 분할 카드 5개와 `delineation` study

**Files:**
- Create: `registry/drafts/{openecg-codec-v6,hrnetv2-delineation,semisegecg-resnet18,semisegecg-vit-tiny,heartkit-seg-tcn}.json`
- Create: `studies/drafts/delineation/study.yaml`

**Interfaces:**
- Consumes: Task 1의 `check_drafts.py`
- Produces: 카드 5개, study 1개. Task 6이 판정한다.

모든 값의 출처: `docs/model-search/task-candidates-2026-10-06.md` Segmentation 절. 다섯 모델 모두 lead 하나를 받으므로 `input_contract.leads`에는 lead 하나(`II`)만 적고 실제 처리 방식은 `x-mival-proposed`에 적는다.

- [ ] **Step 1: `openecg-codec-v6.json`**

```json
{
  "model_id": "openecg-codec-v6",
  "Name": "OpenECG codec_v6 frame segmenter",
  "Summary": "Single-lead network emitting per-sample none/P/QRS/T logits plus beat and rhythm heads.",
  "Link": "https://github.com/vitaldb/openecg",
  "Descriptors": {"Version": "openecg==0.11.0", "References": []},
  "Model properties": {"Input": "single-lead ECG, 10 seconds at 500 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "onnx",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["II"], "sampling_rate_hz": 500, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "none", "layout": "time_lead", "dtype": "float32"
    },
    "output": {"type": "softmax", "n_outputs": 4, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": "onnxruntime", "python": null, "device": "cpu", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["ludb", "qtdb", "isp", "mit-bih-ds1", "vitaldb", "snuh-private"],
    "code_commit": null,
    "notes": "unverified: sha256 of the packaged files (codec_v6.pt 4.9 MB; codec_v6_int8.onnx sha256 starts 640d2d13), python version. No paper; README and openecg/models/codec_v6_MODEL_CARD.md only. Apache-2.0. Weights ship inside the PyPI package. ONNX output batch fixed at 1. Rank normalisation to [-1, 1] per window is not an existing scaling. Ran on LUDB record 1 lead II with sensible P/QRS/T runs (search 2026-10-06). x-mival.output is the nearest existing value and is wrong in meaning: 4 classes per sample, not per record."
  },
  "x-mival-proposed": {
    "per_lead": {"leads_accepted": 1, "apply_to": "each lead independently (benchmarks use lead II)"},
    "scaling": {"op": "rank_normalize", "range": [-1, 1], "per": "window"},
    "heads": [
      {"name": "frame_logits", "shape": [5000, 4], "kind": "per_sample_mask", "classes": ["other", "P", "QRS", "T"]},
      {"name": "beat_logits", "shape": [5000, 6], "kind": "per_sample_mask"},
      {"name": "rhythm_logits", "shape": [5000, 6], "kind": "per_sample_mask"}
    ],
    "use_head": "frame_logits",
    "decode": {"mask": "argmax per sample", "events": "codec.events('frame') -> (start, end, class)", "intervals": "openecg.report() -> pr, qrs_duration, qt in ms"},
    "scoring_window_s": [2, 8],
    "batch_size_fixed": 1
  }
}
```

- [ ] **Step 2: `hrnetv2-delineation.json`**

```json
{
  "model_id": "hrnetv2-delineation",
  "Name": "MedicalAI-DP HRNetV2 ECG delineation",
  "Summary": "Single-lead HRNetV2 emitting per-sample multi-label P/QRS/T logits.",
  "Link": "https://huggingface.co/spaces/MedicalAI-DP/ECG_Delineation",
  "Descriptors": {"Version": null, "References": []},
  "Model properties": {"Input": "single-lead ECG, 10 seconds at 500 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "builder": {"module": "res.impl.HRNetV2:HRNetV2", "kwargs": {}},
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["II"], "sampling_rate_hz": 500, "duration_s": 10, "unit": "mV", "filters": [],
      "scaling": "per_lead_zscore", "gain": 0.1, "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "sigmoid", "n_outputs": 3, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["ludb"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (res/models/hrnetv2/weights.pth, 22.4 MB, sha256 starts 4d4d7721), builder kwargs, Space revision. No paper. Apache-2.0 per the Space README. Normalisation inferred from demo inputs, not documented: per-lead (x-mean)/std then x0.1; gain 0.1 is applied after z-score here only if the compiler orders it that way (unverified). Ran on LUDB record 1 (QRS onsets 643, 1325, 1979; search 2026-10-06). x-mival.output is the nearest existing value and is wrong in meaning."
  },
  "x-mival-proposed": {
    "per_lead": {"leads_accepted": 1, "apply_to": "each lead independently; 12 leads go in as batch 12"},
    "output": {"kind": "per_sample_multilabel", "shape": [3, 5000], "channels": ["P", "QRS", "T"], "activation": "none (raw logits)"},
    "decode": {"thresholds_on_logits": [0.00116, 0.1509, -0.5879], "postprocess_min_run_samples": [5, 25, 25, 25, 15, 25]},
    "scaling_order": "zscore then multiply by 0.1"
  }
}
```

- [ ] **Step 3: `semisegecg-resnet18.json`**

```json
{
  "model_id": "semisegecg-resnet18",
  "Name": "SemiSegECG ResNet-18 + FCN (supervised, cross-domain)",
  "Summary": "Single-lead ResNet-18 segmenter emitting per-sample background/P/QRS/T.",
  "Link": "https://github.com/vuno/semi-seg-ecg",
  "Descriptors": {"Version": null, "References": ["doi:10.1145/3746252.3760790", "arXiv:2507.18323"]},
  "Model properties": {"Input": "single-lead ECG, 10 seconds at 250 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "state_dict_key": "model",
    "builder": {"module": null, "kwargs": {"num_leads": 1}},
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["II"], "sampling_rate_hz": 250, "duration_s": 10, "unit": "mV",
      "filters": [{"kind": "highpass", "cutoff_hz": 0.67, "order": 1}, {"kind": "lowpass", "cutoff_hz": 40, "order": 1}],
      "scaling": "global_zscore", "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "softmax", "n_outputs": 4, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": "torch==1.11.0", "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["ludb", "qtdb", "isp", "zhejiang", "ptb-xl"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (Google Drive id 1uGO-YHayjO4bWRnK4MzT0CMgaoUOgHB0, 48.6 MB, MeanIoU 0.726), filter orders, builder (checkpoint config names resnet18_seg, repo config resnet18). License conflict: LICENSE is Apache-2.0, README says all rights reserved. 250 Hz inferred from the filter config. Per-record z-score over (lead, time) = global_zscore for one lead. x-mival.output is the nearest existing value and is wrong in meaning."
  },
  "x-mival-proposed": {
    "per_lead": {"leads_accepted": 1, "apply_to": "each lead independently"},
    "output": {"kind": "per_sample_mask", "shape": [4, 2500], "classes": ["none", "P", "QRS", "T"], "activation": "softmax over dim 1", "resolution": "input rate (decoder interpolated)"},
    "decode": {"mask": "argmax", "intervals": "notebooks/perf_eval.ipynb compute_numerics -> median PR, QRS, QT in ms"},
    "license_conflict": ["Apache-2.0 (LICENSE)", "all rights reserved (README)"]
  }
}
```

- [ ] **Step 4: `semisegecg-vit-tiny.json`**

```json
{
  "model_id": "semisegecg-vit-tiny",
  "Name": "SemiSegECG ViT-Tiny (FixMatch, cross-domain)",
  "Summary": "Single-lead ViT-Tiny segmenter emitting per-sample background/P/QRS/T.",
  "Link": "https://github.com/vuno/semi-seg-ecg",
  "Descriptors": {"Version": null, "References": ["doi:10.1145/3746252.3760790", "arXiv:2507.18323"]},
  "Model properties": {"Input": "single-lead ECG, 10 seconds at 250 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "torch",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "weights_format": "checkpoint",
    "state_dict_key": "model",
    "builder": {"module": null, "kwargs": {"num_leads": 1}},
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["II"], "sampling_rate_hz": 250, "duration_s": 10, "unit": "mV",
      "filters": [{"kind": "highpass", "cutoff_hz": 0.67, "order": 1}, {"kind": "lowpass", "cutoff_hz": 40, "order": 1}],
      "scaling": "global_zscore", "layout": "lead_time", "dtype": "float32"
    },
    "output": {"type": "softmax", "n_outputs": 4, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": "torch==1.11.0", "python": null, "device": "cuda", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["ludb", "qtdb", "isp", "zhejiang", "ptb-xl"],
    "code_commit": null,
    "notes": "unverified: weights uri/sha (Google Drive vit_tiny-fixmatch.pth), filter orders, builder, whether the ViT patching changes the output length handling. Same family, data and license conflict as semisegecg-resnet18; different architecture. x-mival.output is the nearest existing value and is wrong in meaning."
  },
  "x-mival-proposed": {
    "per_lead": {"leads_accepted": 1, "apply_to": "each lead independently"},
    "output": {"kind": "per_sample_mask", "shape": [4, 2500], "classes": ["none", "P", "QRS", "T"], "activation": "softmax over dim 1"},
    "decode": {"same_as": "semisegecg-resnet18"},
    "license_conflict": ["Apache-2.0 (LICENSE)", "all rights reserved (README)"]
  }
}
```

- [ ] **Step 5: `heartkit-seg-tcn.json`**

```json
{
  "model_id": "heartkit-seg-tcn",
  "Name": "HeartKit seg-4-tcn-lg",
  "Summary": "Single-lead TCN emitting per-sample none/P/QRS/T over 2.56 s frames.",
  "Link": "https://github.com/AmbiqAI/heartkit",
  "Descriptors": {"Version": "seg-4-tcn-lg/latest", "References": []},
  "Model properties": {"Input": "single-lead ECG, 256 samples (2.56 s) at 100 Hz"},
  "Imaging": {"Modality": "ECG"},
  "x-mival": {
    "adapter": "keras",
    "weights": [{"uri": null, "sha256": null, "role": "full"}],
    "ensemble": {"method": "none", "members": 1},
    "input_contract": {
      "leads": ["II"], "sampling_rate_hz": 100, "duration_s": 2.56, "unit": "mV", "filters": [],
      "scaling": "global_zscore", "layout": "time_lead", "dtype": "float32"
    },
    "output": {"type": "softmax", "n_outputs": 4, "positive_index": null},
    "feature_layer": null,
    "threshold": {},
    "runtime": {"framework": null, "python": null, "device": "cpu", "returns": ["logits"]},
    "training_modes_supported": ["inference_only"],
    "pretraining_corpora": ["ludb", "synthetic"],
    "code_commit": null,
    "notes": "unverified: sha256 of model.keras (Ambiq S3 model zoo), keras version. No paper; docs only. BSD-3-Clause. Intended for wearables. A 10 s ECG needs sliding 256-sample windows; duration_s 2.56 makes the compiler crop to one window. 10 ms resolution is coarse for interval measurement. x-mival.output is the nearest existing value and is wrong in meaning."
  },
  "x-mival-proposed": {
    "per_lead": {"leads_accepted": 1, "apply_to": "each lead independently"},
    "windowing": {"frame_samples": 256, "stride_samples": "unspecified", "note": "one record -> several frames -> stitched mask"},
    "output": {"kind": "per_sample_mask", "shape": [256, 4], "classes": ["none", "P", "QRS", "T"]}
  }
}
```

- [ ] **Step 6: `studies/drafts/delineation/study.yaml`**

```yaml
study_id: delineation-draft
site: dicom-miva
seed: 20261006

stages:
  retrieve:
    dsn_env: /data/mi-val/secrets/micdm.env
    schema: cdm
    label_concept_id: null
    modality_concept_id: 4145308
    local_path_root: /scratch/mi-val/dicom
    require_local_file: true
    window_days: 0
    window_days_sens2: 0
    implausible_below: 0
    x-proposed:
      label_source: {kind: per_ecg_measurements, from: "MIMIC-IV-ECG machine_measurements.csv",
                     fields: [pr_interval, qrs_duration, qt_interval], unit: ms, join: "study_id of the same ECG"}
      note: >-
        The reference is several machine measurements of the same ECG, not one
        measurement concept near it. Human P/QRS/T annotations do not exist in
        MIMIC-IV-ECG.

  profile:
    test_fraction: 0.2
    n_folds: 5
    stratify_on: label_primary
    min_test_positives: 0

  preprocess:
    loader: dicom
    registry: registry/drafts
    allow_upsample: false
    pad_policy: reject
    x-proposed:
      per_lead_fanout: "one 12-lead record becomes 12 single-lead inputs for these models"

  models:
    registry: registry/drafts
    cohort_sources: [mimic-iv-ecg]
    batch_size: 64
    arms:
      - {model_id: openecg-codec-v6, training_mode: inference_only, label_def: primary}
      - {model_id: hrnetv2-delineation, training_mode: inference_only, label_def: primary}
      - {model_id: semisegecg-resnet18, training_mode: inference_only, label_def: primary}
      - {model_id: semisegecg-vit-tiny, training_mode: inference_only, label_def: primary}
      - {model_id: heartkit-seg-tcn, training_mode: inference_only, label_def: primary}
    x-proposed:
      prediction_shape: "per sample per lead, not one score per record"
      derived_outputs: [pr_ms, qrs_ms, qt_ms]
      lead_aggregation: "unspecified (median across leads? lead II only?)"

  evaluate:
    outcomes: {primary: null}
    bootstrap_replicates: 2000
    figures: true
    x-proposed:
      metrics: [mean_absolute_error_ms, bland_altman_bias_and_limits]
      per: [pr_ms, qrs_ms, qt_ms]
```

- [ ] **Step 7: 검사 실행**

Run: `cd /Users/minseongkim/Desktop/youlab/mi-va && PYTHONPATH=src python3 $SCRATCH/check_drafts.py > $SCRATCH/check_task5.txt; cat $SCRATCH/check_task5.txt`
Expected:
- `openecg-codec-v6`: `adapter_known=NO(onnx)`.
- 나머지 분할 카드: `load=ok`, `_known=yes`. 판정 2의 재현이다.
- `heartkit-seg-tcn`: `load=ok` (duration 2.56이 통과하는 것 자체를 기록).
- `delineation`: `label_concept_id: null`이면 `retrieve=fail`. 기록한다.
- JSON/YAML 문법 오류만 고친다.

- [ ] **Step 8: 무변경 확인 후 커밋**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
git diff --stat registry/models studies/lvef src tests   # 비어 있어야 한다
git add registry/drafts/openecg-codec-v6.json registry/drafts/hrnetv2-delineation.json registry/drafts/semisegecg-resnet18.json registry/drafts/semisegecg-vit-tiny.json registry/drafts/heartkit-seg-tcn.json studies/drafts/delineation
git commit -m "docs: draft cards for five delineation models and a delineation study

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7"
```

---

### Task 6: 판정, 요약 문서, 원장 L2

**Files:**
- Create: `docs/model-search/schema-gaps.md`
- Modify: `docs/decisions/adaptations.md` (끝의 L2 목록에 항목 추가, 기존 L2-3에 근거 모델 추가)

**Interfaces:**
- Consumes: Task 2~5의 카드·study 17+6개, `$SCRATCH/check_task{2,3,4,5}.txt`
- Produces: 최종 판정표와 L2 항목. 이 작업의 산출물이다.

- [ ] **Step 1: 전체 검사 출력 저장**

Run: `cd /Users/minseongkim/Desktop/youlab/mi-va && PYTHONPATH=src python3 $SCRATCH/check_drafts.py > $SCRATCH/check_final.txt; cat $SCRATCH/check_final.txt`
Expected: `## cards (17)`, `## studies (6)`, `parse_fail` 없음.

- [ ] **Step 2: 카드·study마다 판정 내리기**

판정 규칙 (스펙 "판정 방법"):
- **1 통과**: `load=ok`, 모든 `_known=yes`, `proposed_keys=none`, 그리고 카드 내용을 기존 코드가 뜻대로 읽는다.
- **2 로드되지만 뜻이 틀림**: `load=ok`이고 `_known=yes`인데, `notes`에 "wrong in meaning"이 있거나 `x-mival.output`이 실제 출력의 대체값이다.
- **3 표현 불가**: `load=fail`, 또는 어떤 `_known=NO`, 또는 `proposed_keys`가 있고 그 내용 없이는 실행 의미가 성립하지 않는다.

각 판정에 코드 근거 한 줄을 붙인다. 근거로 쓸 위치:
- `src/mival/adapters/__init__.py:9` (adapter는 torch, keras 둘)
- `src/mival/adapters/base.py:35` (layout은 lead_time, time_lead 둘)
- `src/mival/adapters/torch_adapter.py:325-334` (output.type 넷)
- `src/mival/ops.py:160` (unit은 mV, uV), `src/mival/ops.py:257` (scaling 셋)
- `src/mival/stages/retrieve.py:43`, `:150` (cutoff 둘, 측정값 하나)
- `src/mival/stages/models.py:216`, `:771` (label_def는 기존 label 열, inference_only의 regression/분류 분기)
- `src/mival/stages/preprocess.py:250` (registry 디렉터리의 모든 카드 컴파일)

- [ ] **Step 3: `docs/model-search/schema-gaps.md` 작성**

구조 (값은 Step 2 판정으로 채운다):

```markdown
# 스키마 gap 판정 (2026-10-06)

[스펙](../superpowers/specs/2026-10-06-task-draft-cards-design.md)대로 카드 17개와 study 6개를
지금 코드에 넣어 본 결과다. 카드는 `registry/drafts/`, study는 `studies/drafts/`에 있다.

## 카드

| model_id | task | 판정 | 근거 (코드 위치) | 원장 |
|---|---|---|---|---|
| heartwise-lvef-under50 | LVEF 분류 | 1/2/3 | … | — 또는 L2-n |
| … 17행 … |

## study

| study | 판정 | 막히는 stage | 근거 | 원장 |
|---|---|---|---|---|
| lvef | … |
| … 6행 … |

## gap별 묶음

| L2 | gap | 근거 모델·study |
|---|---|---|
| L2-n | … | … |
```

- 판정 1인 카드가 있으면 "LVEF 실행에 바로 넣을 수 있음"이라고 표시한다 (예상: `heartwise-lvef-under50`).
- 스펙의 "이미 보이는 gap" 표 18개 각각이 위 묶음 중 어디에 들어갔는지, 또는 판정 결과 gap이 아니었는지를 빠짐없이 적는다.

- [ ] **Step 4: `docs/decisions/adaptations.md`에 L2 추가**

기존 마지막 L2 번호(현재 L2-6) 다음부터 매긴다. 원인이 같은 gap은 한 항목으로 묶는다. 예상 묶음 (판정 결과에 따라 합치거나 나눈다):
- 출력 형태: 샘플별 마스크, 생존(구간 조건부·Cox·MTLR), 여러 head와 head 선택
- 출력 후처리: 단위 역변환(z-score, log), 생존곡선 → 기준 시점 위험, baseline hazard 부재
- 입력: lead별 독립 실행(per-lead fan-out), 4-D layout, 창 분할(windowing), rank 정규화, crop 위치(first/middle)
- 공변량·tabular: **기존 L2-3에 근거 모델(EchoNext, cavalab MTLR)만 추가**, 새 항목 만들지 않음
- 라벨: 측정값 하나가 아닌 라벨(생년 계산, 사건까지 시간·censoring, 같은 ECG의 기계 측정), cutoff 개수·방향, 하루 미만 시간 창
- 평가: 생존 지표, 간격 오차 지표
- 실행 형식: keras 3, ONNX
- 모델 선택 단위: study가 registry 디렉터리 하나를 통째로 씀
- 앙상블: 회귀 출력 평균 (`mean_probability` 이름)

항목 형식:

```markdown
### L2-7 · <한 줄 제목>

<무엇이 표현되지 않는지 2~4문장. 코드 위치 포함.>

- 근거: <model_id 나열>, `docs/model-search/schema-gaps.md`
- 발견: 2026-10-06, task별 카드 초안 작성 중
- 상태: **보고됨. 만들지 않음**
- 착수 조건: <이 gap이 막는 task를 실행하기로 할 때 등 한 줄>
```

기존 L2-3 항목 끝에 한 줄 추가: `- 추가 근거 (2026-10-06): echonext-mini (7개), cavalab-mtlr-code15 (나이·성별)`

- [ ] **Step 5: 완결성 확인**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
ls registry/drafts/*.json | wc -l                         # 17
ls studies/drafts/*/study.yaml | wc -l                    # 6
grep -c "^| " docs/model-search/schema-gaps.md            # 카드 17 + study 6 + 묶음 행 + 헤더
grep -n "TBD\|TODO\|…" docs/model-search/schema-gaps.md   # 출력 없음
git diff --stat registry/models studies/lvef src tests    # 비어 있어야 한다
```

- [ ] **Step 6: 커밋**

```bash
cd /Users/minseongkim/Desktop/youlab/mi-va
git add docs/model-search/schema-gaps.md docs/decisions/adaptations.md
git commit -m "docs: schema-gap verdicts for the draft cards and studies, L2 entries

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7"
```
