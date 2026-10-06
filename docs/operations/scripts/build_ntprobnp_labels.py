"""MIMIC-IV 3.1 labevents itemid 50963 (NT-proBNP, pg/mL) -> person_id, measurement_datetime, value."""
import pandas as pd, psycopg
from mival.stages.retrieve import read_connection_settings
raw = pd.read_csv("/data/mi-val/datasets/mimic-iv/ntprobnp_labevents_50963.csv")
s = read_connection_settings("/data/mi-val/secrets/micdm.env")
with psycopg.connect(**s) as c, c.cursor() as cur:
    cur.execute("select person_id, person_source_value from cdm.person")
    person = pd.DataFrame(cur.fetchall(), columns=["person_id", "subject_id"])
person["subject_id"] = person["subject_id"].astype("int64")
out = raw.merge(person, on="subject_id", how="inner")
out = out[out["valuenum"].notna()][["person_id", "charttime", "valuenum"]].rename(columns={"charttime": "measurement_datetime", "valuenum": "value"})
out.to_csv("/data/mi-val/datasets/mimic-iv/ntprobnp_labels.csv", index=False)
print(len(raw), "raw;", len(out), "with person and value;", out["value"].describe().round(1).to_dict())
