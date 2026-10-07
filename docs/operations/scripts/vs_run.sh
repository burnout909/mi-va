#!/bin/bash
# Sample (or full) end-to-end for one study: retrieve -> profile -> preprocess -> models per env -> evaluate.
# usage: vs_sample.sh <study> [N|all]   (N rows of the cohort, 40% from test persons; "all" = whole cohort)
set -o pipefail
S=$1; N=${2:-1000}; R=/data/mi-val/runs; W=/data/mi-val/mi-va-vs
if [ "$N" = all ]; then
  # Full cohorts: tensors (~160 GB over all studies) go to /scratch; retrieve and
  # profile are reused from /data so their hashes match and they are skipped.
  D=$R; R=/scratch/mi-val/runs; mkdir -p $R/$S
  for st in retrieve profile; do [ -d $D/$S/$st ] && rsync -a $D/$S/$st $R/$S/; done
  O=$R/$S/_full
else O=$R/$S/_sample; fi; mkdir -p $O
cd $W
mv_py=/data/mi-val/envs/mival/bin/python
cli() { local py=$1; shift; PYTHONPATH=src $py -c "import sys; from mival.pipeline.cli import main; sys.exit(main())" "$@"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/sample.log; }
latest() { ls -td $R/$S/$1/*/ 2>/dev/null | while read d; do [ -f $d/manifest.json ] && echo $d && break; done; }
log "retrieve"; cli $mv_py run retrieve --study studies/$S --runs-root $R > $O/retrieve.log 2>&1 || { log "retrieve FAILED"; tail -20 $O/retrieve.log; exit 1; }
CI=$(latest retrieve)artifacts/cohort_index.parquet; log "cohort_index $CI"
log "profile"; cli $mv_py run profile --study studies/$S --runs-root $R --input cohort_index=$CI > $O/profile.log 2>&1 || { log "profile FAILED"; tail -20 $O/profile.log; exit 1; }
SPL=$(latest profile)artifacts/cohort_split.parquet
if [ "$N" = all ]; then COH=$CI; else
$mv_py - <<PY
import pandas as pd
c = pd.read_parquet("$CI"); s = pd.read_parquet("$SPL")
t = set(s[s.split=="test"].person_id); c = c.assign(_t=c.person_id.isin(t))
pick = pd.concat([c[c._t].head(int($N*0.4)), c[~c._t].head($N - int($N*0.4))]).drop(columns="_t")
pick.to_parquet("$O/cohort_index_sample.parquet", index=False); print("sample", len(pick))
PY
COH=$O/cohort_index_sample.parquet; fi
log "preprocess"; cli $mv_py run preprocess --study studies/$S --runs-root $R --input cohort_index=$COH > $O/preprocess.log 2>&1 || { log "preprocess FAILED"; tail -20 $O/preprocess.log; exit 1; }
PRE=$(latest preprocess)artifacts/preprocess_index.parquet
# one models run per environment named by the arms' cards
ENVS=$($mv_py - <<PY
import json, yaml, pathlib
st = yaml.safe_load(open("studies/$S/study.yaml")); m = st["stages"]["models"]; reg = pathlib.Path(m["registry"])
cards = {json.loads(p.read_text())["model_id"]: json.loads(p.read_text()) for p in reg.glob("*.json")}
envs = {}
for arm in m["arms"]:
    envs.setdefault(cards[arm["model_id"]]["x-mival"]["runtime"]["env"], []).append(arm)
for env, arms in envs.items():
    name = pathlib.Path(env).name; d = pathlib.Path("$O") / f"study-{name}"; d.mkdir(exist_ok=True)
    s2 = json.loads(json.dumps(st)); s2["stages"]["models"]["arms"] = arms
    (d / "study.yaml").write_text(yaml.safe_dump(s2, sort_keys=False)); print(env)
PY
)
rm -rf $O/predictions_all; mkdir -p $O/predictions_all
for E in $ENVS; do
  name=$(basename $E); log "models [$name]"
  if [ "$name" = xecg ]; then export CUDA_HOME=/usr/local/cuda-12.8 CC=gcc-14 CXX=g++-14; fi
  cli $E/bin/python run models --study $O/study-$name --runs-root $R --input preprocess_index=$PRE --input cohort_split=$SPL --input cohort_index=$COH > $O/models-$name.log 2>&1 || { log "models [$name] FAILED"; tail -30 $O/models-$name.log; continue; }
  M=$(grep -o "$R/$S/models/[0-9a-f]*" $O/models-$name.log | tail -1); [ -z "$M" ] && M=$(latest models)
  cp ${M%/}/artifacts/predictions/*.parquet $O/predictions_all/
done
log "evaluate"; cli $mv_py run evaluate --study studies/$S --runs-root $R --input predictions=$O/predictions_all --input cohort_split=$SPL > $O/evaluate.log 2>&1 || { log "evaluate FAILED"; tail -30 $O/evaluate.log; exit 1; }
E=$(latest evaluate); log "done $E"
if [ "$N" = all ]; then rsync -a $R/$S/models $R/$S/evaluate $O $D/$S/ && log "copied models, evaluate, _full to $D/$S"; fi
$mv_py -c "
import pandas as pd; t=pd.read_parquet('${E}artifacts/metrics_long.parquet')
t=t[(t.split=='test')&(t.subgroup=='all')&(t.metric.isin(['mae','bias','r2','c_index','auroc_horizon@365']))]
print(t[['model_id','label_def','metric','value','ci_lo','ci_hi','n','n_events']].to_string(index=False))" | tee -a $O/sample.log
