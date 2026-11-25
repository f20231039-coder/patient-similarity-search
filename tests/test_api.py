import pytest
from fastapi.testclient import TestClient

from simsearch import api
from simsearch.index import PatientIndex


@pytest.fixture(scope="module")
def client(docs1, risk_model, monkeypatch_module):
    monkeypatch_module.setenv("WARMUP_MODELS", "0")
    idx = PatientIndex(collection="test_api")
    idx.reset()
    idx.upsert(docs1)
    api.init_state(index=idx)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client, docs1):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["patients_indexed"] == len(docs1) == h["qdrant_points"]


def test_search_endpoint(client, docs1):
    r = client.post("/search", json={"patient_id": docs1[0]["short_id"], "k": 3, "filters": {"gender": "male"}})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) <= 3 and all(x["gender"] == "male" for x in body["results"])
    assert "fusion_ms" in body["timings"] and body["mode"] in ("ltr", "hybrid")  # hybrid when no ranker is trained
    assert client.post("/search", json={"k": 3}).status_code == 422
    assert client.post("/search", json={"patient_id": "P-00000000"}).status_code == 404



def test_reindex_endpoint(client, docs1, docs2):
    r = client.post("/reindex", json={"source": "period2"}).json()
    assert r["total_after"] == r["qdrant_points"] == len({d["patient_id"] for d in docs1 + docs2})
    h = client.get("/health").json()
    assert h["records_by_period"]["2"] == len(docs2)


def test_report(client, docs1):
    pid = docs1[3]["short_id"]
    assert client.get(f"/patients/browse?q={pid[:6]}").json()[0]["short_id"].startswith(pid[:6])  # ID prefix
    r = client.post("/report", json={"patient_id": pid, "k": 5, "filters": {"age_min": 1}})
    assert r.status_code == 200
    body = r.json()
    assert body["patient"]["short_id"] == pid and len(body["matches"]) <= 5
    assert pid not in [m["short_id"] for m in body["matches"]]
    assert all(m["age"] >= 1 for m in body["matches"])
    assert "probability" in body["risk"] and "top_features" in body["explanation"]
    assert client.post("/report", json={"patient_id": "P-00000000"}).status_code == 404


def test_app_pages_served(client):
    home = client.get("/")
    assert home.status_code == 200 and "Patient Similarity" in home.text
    for path in ("/static/common.js",):
        assert client.get(path).status_code == 200, path
    for gone in ("/patients/examples", "/patients/find?q=P-0", "/options/dashboard"):
        assert client.get(gone).status_code == 404, gone
    assert client.post("/chat", json={"message": "hi"}).status_code in (404, 405)


def test_find_patient_without_id(client):
    cases = client.get("/patients/showcase").json()
    assert cases and all(c["title"] and c["short_id"].startswith("P-") for c in cases)
    conds = client.get("/patients/conditions?limit=5").json()
    assert conds and conds[0]["patients"] >= conds[-1]["patients"]
    name = conds[0]["name"]
    listed = client.get(f"/patients/browse?condition={name}&limit=5").json()
    assert listed and all(name in p["chronic_conditions"] for p in listed)
    described = client.get("/patients/browse?q=older woman high blood pressure&gender=female&limit=5").json()
    assert described and all(p["gender"] == "female" for p in described)


def test_compare_two_patients(client, docs1):
    a, b = docs1[0]["short_id"], docs1[5]["short_id"]
    r = client.get(f"/compare?a={a}&b={b}")
    assert r.status_code == 200
    c = r.json()
    assert c["a"]["short_id"] == a and c["b"]["short_id"] == b
    assert "deceased" in c["a"] and "probability" in c["a"]["risk"]
    ca, cb = set(c["a"]["chronic_conditions"]), set(c["b"]["chronic_conditions"])
    assert set(c["shared_conditions"]) == ca & cb and set(c["only_a"]) == ca - cb and set(c["only_b"]) == cb - ca
    assert client.get(f"/compare?a={a}&b=P-00000000").status_code == 404


def test_plain_language_query(client):
    r = client.post("/query", json={"q": "women with high BMI", "k": 5}).json()
    assert r["parser"].startswith("rules") and {c["field"] for c in r["chips"]} == {"gender", "labs.bmi"}
    assert all(x["gender"] == "female" for x in r["results"])
    edited = client.post("/query", json={"q": "women with high BMI", "parsed": {"labs": {"bmi": "high"}}, "k": 5}).json()
    assert edited["parser"] == "edited by user" and [c["field"] for c in edited["chips"]] == ["labs.bmi"]
    assert client.post("/query", json={"q": ""}).status_code == 422
