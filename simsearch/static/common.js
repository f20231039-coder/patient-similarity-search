/* Shared data layer for the app: API calls, plain-language lab names and ranges, and the "why similar" logic. No UI here. */
(function () {
  "use strict";

  async function api(path, body, method) {
    const opts = body !== undefined
      ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : { method: method || "GET" };
    let r;
    try {
      r = await fetch(path, opts);
    } catch (e) {
      throw new Error("Can't reach the app server. Check that it is running, then try again.");
    }
    let j = null;
    try { j = await r.json(); } catch (e) { /* non-JSON error */ }
    if (!r.ok) {
      const d = j && j.detail;
      if (r.status === 404) throw new Error("That patient ID isn't in the database. Check the ID or pick an example.");
      if (r.status === 504) throw new Error("This took too long. Try again, or narrow the filters.");
      throw new Error(typeof d === "string" ? d : "Something went wrong on the server. Try again.");
    }
    return j;
  }

  /* Friendly lab names, the normal range shown to the reader, and the group they belong to. */
  const LABS = {
    bmi: { name: "BMI", plain: "Body mass index", range: "18.5–24.9", group: "Body" },
    systolic: { name: "Blood pressure (upper)", plain: "Systolic blood pressure", range: "under 130 mmHg", unit: "mmHg", group: "Heart" },
    diastolic: { name: "Blood pressure (lower)", plain: "Diastolic blood pressure", range: "under 80 mmHg", unit: "mmHg", group: "Heart" },
    heart_rate: { name: "Heart rate", plain: "Resting heart rate", range: "50–100 bpm", unit: "bpm", group: "Heart" },
    resp_rate: { name: "Breathing rate", plain: "Respiratory rate", range: "12–20 /min", unit: "/min", group: "Body" },
    glucose: { name: "Blood sugar", plain: "Glucose", range: "70–99 mg/dL", unit: "mg/dL", group: "Sugar" },
    hba1c: { name: "HbA1c", plain: "3-month average blood sugar", range: "under 5.7%", unit: "%", group: "Sugar" },
    cholesterol: { name: "Total cholesterol", plain: "Total cholesterol", range: "under 200 mg/dL", unit: "mg/dL", group: "Fats" },
    ldl: { name: "LDL", plain: "LDL (bad) cholesterol", range: "under 130 mg/dL", unit: "mg/dL", group: "Fats" },
    hdl: { name: "HDL", plain: "HDL (good) cholesterol", range: "40 mg/dL or more", unit: "mg/dL", group: "Fats" },
    triglycerides: { name: "Triglycerides", plain: "Triglycerides", range: "under 150 mg/dL", unit: "mg/dL", group: "Fats" },
    creatinine: { name: "Creatinine", plain: "Creatinine (kidney)", range: "up to 1.3 mg/dL", unit: "mg/dL", group: "Kidney" },
    egfr: { name: "eGFR", plain: "Kidney filtering rate", range: "60 or more", unit: "mL/min", group: "Kidney" },
    microalbumin: { name: "Urine albumin", plain: "Urine albumin/creatinine", range: "under 30 mg/g", unit: "mg/g", group: "Kidney" },
    hemoglobin: { name: "Hemoglobin", plain: "Hemoglobin (blood count)", range: "12–17.5 g/dL", unit: "g/dL", group: "Blood" },
    leukocytes: { name: "White cells", plain: "White blood cell count", range: "4–11 ×10³/µL", unit: "×10³/µL", group: "Blood" },
    fev1_fvc: { name: "FEV1/FVC", plain: "Lung function ratio", range: "70% or more", unit: "%", group: "Lungs" },
    pain: { name: "Pain score", plain: "Reported pain (0–10)", range: "under 7", group: "Wellbeing" },
    phq2: { name: "PHQ-2", plain: "Depression screen", range: "under 3", group: "Wellbeing" },
    gad7: { name: "GAD-7", plain: "Anxiety screen", range: "under 10", group: "Wellbeing" },
    audit_c: { name: "AUDIT-C", plain: "Alcohol screen", range: "under 4", group: "Wellbeing" },
  };
  /* Normal range as numbers (for the compare dot plots) and the body area each test belongs to. */
  const NORMAL = { systolic: [90, 130], diastolic: [60, 80], heart_rate: [50, 100], resp_rate: [12, 20], bmi: [18.5, 25],
    glucose: [70, 100], hba1c: [4, 5.7], cholesterol: [120, 200], ldl: [50, 130], hdl: [40, 80], triglycerides: [40, 150],
    creatinine: [0.6, 1.3], egfr: [60, 120], microalbumin: [0, 30], hemoglobin: [12, 17.5], leukocytes: [4, 11],
    fev1_fvc: [70, 90], pain: [0, 7], phq2: [0, 3], gad7: [0, 10], audit_c: [0, 4] };
  const AREAS = [
    ["Heart and blood pressure", ["systolic", "diastolic", "heart_rate"]],
    ["Blood sugar", ["glucose", "hba1c"]],
    ["Cholesterol and fats", ["cholesterol", "ldl", "hdl", "triglycerides"]],
    ["Kidneys", ["creatinine", "egfr", "microalbumin"]],
    ["Blood count", ["hemoglobin", "leukocytes"]],
    ["Body and breathing", ["bmi", "resp_rate", "fev1_fvc"]],
    ["Wellbeing scores", ["pain", "phq2", "gad7", "audit_c"]],
  ];

  const LAB_ORDER = ["systolic", "diastolic", "heart_rate", "bmi", "glucose", "hba1c", "cholesterol", "ldl", "hdl",
    "triglycerides", "creatinine", "egfr", "microalbumin", "hemoglobin", "leukocytes", "fev1_fvc", "resp_rate",
    "pain", "phq2", "gad7", "audit_c"];
  /* Typical spread, used to decide when two values count as "similar". */
  const SCALE = { bmi: 6, systolic: 15, diastolic: 10, heart_rate: 12, resp_rate: 3, glucose: 20, hba1c: 0.8,
    cholesterol: 35, ldl: 30, hdl: 12, triglycerides: 50, creatinine: 0.4, egfr: 20, hemoglobin: 1.5,
    leukocytes: 2, microalbumin: 30, fev1_fvc: 10, pain: 2, phq2: 1.5, gad7: 4, audit_c: 2 };
  /* Main labs compared value-by-value in "why similar". */
  const CORE_LABS = ["systolic", "diastolic", "bmi", "glucose", "hba1c", "cholesterol", "ldl", "hdl",
    "triglycerides", "creatinine", "egfr", "hemoglobin"];

  const FLAG_WORDS = { high: "high", low: "low", borderline: "borderline", obese: "obese", overweight: "overweight",
    underweight: "underweight" };

  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a >= 100) return String(Math.round(v));
    if (a >= 10) return String(Math.round(v * 10) / 10);
    return String(Math.round(v * 100) / 100);
  }

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function labRows(structured) {
    const latest = (structured && structured.latest) || {};
    return LAB_ORDER.filter((k) => latest[k]).map((k) => {
      const v = latest[k];
      const t = structured.trends && structured.trends[k];
      return { key: k, ...LABS[k], value: v.value, flag: v.flag, flagWord: v.flag ? FLAG_WORDS[v.flag] || v.flag : null,
        trend: t ? t.direction : null, trendFrom: t ? t.first : null };
    });
  }

  /* Plain-language reasons a match resembles the query patient, plus key differences. */
  function whySimilar(q, m) {
    const qs = q.structured, ms = m.structured, same = [], diff = [];
    if (qs.age != null && ms.age != null) {
      const d = Math.abs(qs.age - ms.age);
      if (d <= 6) same.push({ text: `Similar age (${Math.round(qs.age)} and ${Math.round(ms.age)})`, kind: "age" });
      else if (d >= 15) diff.push({ text: `Age differs (${Math.round(qs.age)} vs ${Math.round(ms.age)})`, kind: "age" });
    }
    const ql = qs.latest || {}, ml = ms.latest || {};
    for (const k of LAB_ORDER) {
      const a = ql[k], b = ml[k];
      if (!a || !b) continue;
      if (a.flag && a.flag === b.flag) {
        same.push({ text: `Both have ${FLAG_WORDS[a.flag] || a.flag} ${LABS[k].plain.toLowerCase()} (${fmt(a.value)} and ${fmt(b.value)})`, kind: "flag", key: k });
      } else if ((a.flag && a.flag !== "borderline") !== (b.flag && b.flag !== "borderline") && (a.flag || b.flag)) {
        const who = a.flag ? "this patient" : "the match";
        diff.push({ text: `${LABS[k].plain} ${FLAG_WORDS[a.flag || b.flag] || (a.flag || b.flag)} only in ${who}`, kind: "flag", key: k });
      } else if (!a.flag && !b.flag && CORE_LABS.includes(k) && Math.abs(a.value - b.value) <= SCALE[k] * 0.35) {
        same.push({ text: `Similar ${LABS[k].plain.toLowerCase()} (${fmt(a.value)} and ${fmt(b.value)})`, kind: "close", key: k });
      }
    }
    const qt = qs.trends || {}, mt = ms.trends || {};
    for (const k of Object.keys(qt)) {
      if (mt[k] && qt[k].direction !== "stable" && qt[k].direction === mt[k].direction) {
        same.push({ text: `${LABS[k] ? LABS[k].plain : k} ${qt[k].direction} in both`, kind: "trend", key: k });
      }
    }
    if (qs.smoking && qs.smoking === ms.smoking && !/never/i.test(qs.smoking)) {
      same.push({ text: "Same smoking history", kind: "life" });
    }
    const order = { flag: 0, trend: 1, age: 2, close: 3, life: 4 };
    same.sort((x, y) => order[x.kind] - order[y.kind]);
    return { same: same.slice(0, 5), diff: diff.slice(0, 3), sameCount: same.length };
  }

  /* How strongly a match resembles the query, in words, from its rank and shared signals. */
  function matchWords(why, rank) {
    if (rank != null) return rank <= 3 ? "Closest match" : rank <= 7 ? "Close match" : "Similar";
    return why.sameCount >= 4 ? "Close match" : "Similar";
  }

  function riskWords(r) {
    if (!r || r.error) return null;
    const pct = Math.round(r.probability * 100);
    const level = r.risk_level;
    const sentence = level === "high"
      ? `High chance (${pct}%) of a chronic metabolic or heart-related condition.`
      : level === "moderate"
        ? `Moderate chance (${pct}%) of a chronic metabolic or heart-related condition.`
        : `Low chance (${pct}%) of a chronic metabolic or heart-related condition.`;
    return { pct, level, sentence, conditions: "diabetes, high blood pressure, prediabetes or metabolic syndrome", model: r.model };
  }

  function reasonWords(ex) {
    if (!ex || ex.error || !ex.top_features) return [];
    return ex.top_features.map((f) => {
      const raises = f.effect === "raises risk";
      if (f.feature === "male sex") {
        const sex = (v) => (v >= 0.5 ? "male" : "female");
        return { feature: "sex", value: sex(f.value), typical: sex(f.typical_value), raises,
          text: `Being ${sex(f.value)} ${raises ? "raises" : "lowers"} the risk`, weight: Math.abs(f.contribution) };
      }
      return { feature: f.feature, value: fmt(f.value), typical: fmt(f.typical_value), raises,
        text: `${capital(f.feature)} ${fmt(f.value)} (typical ${fmt(f.typical_value)}) ${raises ? "raises" : "lowers"} the risk`,
        weight: Math.abs(f.contribution) };
    });
  }

  function capital(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

  function conditionShare(report) {
    const n = report.matches.length || 1;
    return report.condition_counts.map(([name, count]) => ({ name, count, of: n, pct: Math.round((count / n) * 100) }));
  }

  function personLine(p) {
    const age = p.age != null ? `${Math.round(p.age)}-year-old` : "Age unknown,";
    const ins = p.insurance ? `, ${p.insurance === "uninsured" ? "no insurance" : p.insurance + " insurance"}` : "";
    return `${age} ${p.gender}${ins}`;
  }

  function filtersFrom(form) {
    const f = {};
    const amin = form.querySelector("[name=age_min]"), amax = form.querySelector("[name=age_max]");
    const g = form.querySelector("[name=gender]"), ins = form.querySelector("[name=insurance]");
    if (amin && amin.value !== "") f.age_min = Number(amin.value);
    if (amax && amax.value !== "") f.age_max = Number(amax.value);
    if (g && g.value) f.gender = g.value;
    if (ins && ins.value) f.insurance = ins.value;
    return f;
  }

  const HOW_IT_WORKS = [
    { title: "You pick a patient", text: "Choose a demo case, describe who you're looking for in plain English, or browse by condition. Their latest vitals, labs, trends and visit history become the starting point." },
    { title: "The app finds look-alikes", text: "It compares that record with all 2,275 patients in two ways at once (matching words and matching overall meaning), merges the two lists, then re-orders the top 50 by how close ages, labs and vitals really are." },
    { title: "It shows what they have in common", text: "For the 10 closest patients you see their recorded long-term conditions. If many share one, this patient may be worth checking for it too." },
    { title: "It estimates risk and explains it", text: "A trained model estimates the chance of diabetes, high blood pressure, prediabetes or metabolic syndrome, and lists the measurements that pushed the estimate up or down." },
    { title: "You can compare any two patients", text: "Side by side: every result with both values and flags, the conditions they share or don't, and both risk estimates, with differences highlighted." },
  ];

  async function report(id, filters) { return api("/report", { patient_id: id, filters, k: 10 }); }

  window.SIM = { api, report, LABS, LAB_ORDER, NORMAL, AREAS, labRows, whySimilar, matchWords, riskWords, reasonWords,
    conditionShare, personLine, filtersFrom, fmt, esc, capital, HOW_IT_WORKS };
})();
