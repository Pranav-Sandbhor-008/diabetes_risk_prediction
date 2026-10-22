import os
import warnings

import numpy as np
import joblib
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --------------------------------------------------------------------------
# 1. LOAD MODEL  (svc.pkl must sit next to app.py in the project root)
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svc.pkl")
model = joblib.load(MODEL_PATH)  # joblib.load also reads plain pickle files

# --------------------------------------------------------------------------
# 2. CONFIG  -- EDIT THESE IF YOUR TRAINING ENCODING WAS DIFFERENT
#    Each list position = the number the model was trained with.
#    (Assumed: LabelEncoder, i.e. alphabetical order.)
# --------------------------------------------------------------------------
CLASS_MAP = {0: "Diabetes", 1: "No Diabetes", 2: "Pre-Diabetes"}

CATEGORIES = {
    "gender": ["Female", "Male", "Other"],
    "city": [f"City {i + 1}" for i in range(18)],  # replace with real city names
    "family_history_diabetes": ["No", "Yes"],
    "physical_activity_level": ["High", "Low", "Moderate"],
    "diet_type": ["Balanced", "High Carb", "Low Carb", "Vegetarian"],
    "smoking_status": ["Current", "Former", "Never"],
    "alcohol_consumption": ["Heavy", "Moderate", "None"],
    "income_bracket": ["High", "Low", "Middle"],
}

# Numeric fields: (min, max, step, default, unit)
NUMERIC = {
    "age": (18, 90, 1, 40, "years"),
    "bmi": (12, 60, 0.1, 25.0, "kg/m²"),
    "hours_sleep_per_night": (2, 14, 0.5, 7, "hours"),
    "stress_level": (1, 10, 1, 5, "/ 10"),
    "fasting_blood_sugar": (60, 400, 1, 100, "mg/dL"),
    "hba1c_level": (3.5, 15, 0.1, 5.5, "%"),
    "blood_pressure_systolic": (80, 220, 1, 120, "mmHg"),
    "blood_pressure_diastolic": (40, 140, 1, 80, "mmHg"),
    "waist_circumference_cm": (50, 170, 0.5, 90, "cm"),
}

# Exact column order the model was trained with
FEATURES = [
    "age", "gender", "city", "bmi", "family_history_diabetes",
    "physical_activity_level", "diet_type", "smoking_status",
    "alcohol_consumption", "hours_sleep_per_night", "stress_level",
    "fasting_blood_sugar", "hba1c_level", "blood_pressure_systolic",
    "blood_pressure_diastolic", "waist_circumference_cm", "income_bracket",
]

UI_CONFIG = {
    "categories": CATEGORIES,
    "numeric": {k: dict(min=v[0], max=v[1], step=v[2], default=v[3], unit=v[4])
                for k, v in NUMERIC.items()},
}


# --------------------------------------------------------------------------
# 3. PREDICTION + PLAIN-LANGUAGE EXPLANATION
# --------------------------------------------------------------------------
def parse_input(data):
    values, errors = [], []
    for name in FEATURES:
        raw = data.get(name)
        if raw in (None, ""):
            errors.append(f"'{name}' is missing.")
            continue
        try:
            if name in CATEGORIES:
                idx = int(raw)
                if not 0 <= idx < len(CATEGORIES[name]):
                    raise ValueError
                values.append(idx)
            else:
                lo, hi = NUMERIC[name][0], NUMERIC[name][1]
                num = float(raw)
                if not lo <= num <= hi:
                    errors.append(f"'{name}' must be between {lo} and {hi}.")
                    continue
                values.append(num)
        except (ValueError, TypeError):
            errors.append(f"'{name}' has an invalid value.")
    return values, errors


def build_insights(d):
    """Simple rule-based health notes shown next to the ML prediction."""
    notes = []

    def add(level, title, text):
        notes.append({"level": level, "title": title, "text": text})

    a1c, fbs = d["hba1c_level"], d["fasting_blood_sugar"]
    if a1c >= 6.5:
        add("high", "HbA1c", f"{a1c}% is in the diabetic range (6.5% or above).")
    elif a1c >= 5.7:
        add("mid", "HbA1c", f"{a1c}% is in the pre-diabetic range (5.7%–6.4%).")
    else:
        add("good", "HbA1c", f"{a1c}% is in the normal range (below 5.7%).")

    if fbs >= 126:
        add("high", "Fasting sugar", f"{fbs:g} mg/dL is in the diabetic range (126 or above).")
    elif fbs >= 100:
        add("mid", "Fasting sugar", f"{fbs:g} mg/dL is slightly raised (100–125).")
    else:
        add("good", "Fasting sugar", f"{fbs:g} mg/dL is normal (below 100).")

    bmi = d["bmi"]
    if bmi >= 30:
        add("high", "BMI", f"{bmi} falls in the obesity range (30 or above).")
    elif bmi >= 25:
        add("mid", "BMI", f"{bmi} falls in the overweight range (25–29.9).")
    elif bmi >= 18.5:
        add("good", "BMI", f"{bmi} is in the healthy range (18.5–24.9).")
    else:
        add("mid", "BMI", f"{bmi} is below the healthy range (under 18.5).")

    sys_, dia = d["blood_pressure_systolic"], d["blood_pressure_diastolic"]
    if sys_ >= 140 or dia >= 90:
        add("high", "Blood pressure", f"{sys_:g}/{dia:g} mmHg is high.")
    elif sys_ >= 130 or dia >= 80:
        add("mid", "Blood pressure", f"{sys_:g}/{dia:g} mmHg is slightly elevated.")
    else:
        add("good", "Blood pressure", f"{sys_:g}/{dia:g} mmHg is in a healthy range.")

    waist = d["waist_circumference_cm"]
    limit = 94 if CATEGORIES["gender"][int(d["gender"])] == "Male" else 80
    if waist > limit:
        add("mid", "Waist size", f"{waist:g} cm is above the suggested limit of about {limit} cm.")
    else:
        add("good", "Waist size", f"{waist:g} cm is within the suggested limit.")

    if d["hours_sleep_per_night"] < 6:
        add("mid", "Sleep", "Less than 6 hours of sleep can affect blood sugar control.")
    if d["stress_level"] >= 8:
        add("mid", "Stress", "A high stress level can raise blood sugar over time.")
    if CATEGORIES["family_history_diabetes"][int(d["family_history_diabetes"])] == "Yes":
        add("mid", "Family history", "A close relative with diabetes raises your own risk.")
    if CATEGORIES["smoking_status"][int(d["smoking_status"])] == "Current":
        add("mid", "Smoking", "Smoking increases the risk of type 2 diabetes and its complications.")
    return notes


SUMMARIES = {
    "Diabetes": {
        "tone": "high",
        "headline": "High likelihood of diabetes",
        "text": "Based on the values you entered, the model's pattern matches people who "
                "have diabetes. This is not a diagnosis, but it is a strong signal to see a "
                "doctor and get a confirmatory blood test soon.",
        "tips": [
            "Book an appointment with a doctor or diabetologist for confirmatory tests.",
            "Cut back on sugary drinks, sweets and refined carbohydrates.",
            "Aim for at least 30 minutes of walking or movement on most days.",
            "Keep a record of your sugar readings to show your doctor.",
        ],
    },
    "Pre-Diabetes": {
        "tone": "mid",
        "headline": "Pre-diabetes pattern detected",
        "text": "Your blood sugar markers are higher than ideal but not yet in the diabetic "
                "range. The good news: lifestyle changes at this stage can often prevent or "
                "delay type 2 diabetes.",
        "tips": [
            "Aim to lose 5–7% of body weight if you are overweight.",
            "Eat more fibre (vegetables, whole grains, pulses) and fewer refined carbs.",
            "Exercise about 150 minutes per week, spread across the week.",
            "Re-check HbA1c and fasting sugar every 3–6 months.",
        ],
    },
    "No Diabetes": {
        "tone": "good",
        "headline": "Low likelihood of diabetes",
        "text": "Based on the values you entered, the model sees no strong sign of diabetes "
                "or pre-diabetes. Keep up your healthy habits and continue regular check-ups.",
        "tips": [
            "Keep a balanced diet and stay physically active.",
            "Maintain a healthy weight and 7–8 hours of sleep.",
            "Get a routine blood sugar test once a year.",
            "Limit smoking and alcohol.",
        ],
    },
}


def softmax(x, temp=1.0):
    z = np.asarray(x, dtype=float) / temp
    z = np.exp(z - z.max())
    return z / z.sum()


def run_prediction(values):
    X = np.array(values, dtype=float).reshape(1, -1)
    pred = int(model.predict(X)[0])

    # SVC was trained without probability=True, so we derive an approximate
    # confidence from the decision scores.
    probs = {}
    try:
        scores = np.ravel(model.decision_function(X))
        classes = [int(c) for c in model.classes_]
        conf = softmax(scores)
        probs = {CLASS_MAP.get(c, str(c)): round(float(p) * 100, 1)
                 for c, p in zip(classes, conf)}
    except Exception:
        pass

    label = CLASS_MAP.get(pred, str(pred))
    info = SUMMARIES.get(label, SUMMARIES["No Diabetes"])
    return {
        "label": label,
        "tone": info["tone"],
        "headline": info["headline"],
        "text": info["text"],
        "tips": info["tips"],
        "confidence": probs.get(label),
        "probabilities": probs,
    }


# --------------------------------------------------------------------------
# 4. ROUTES
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE, config=UI_CONFIG)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    values, errors = parse_input(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    try:
        result = run_prediction(values)
        result["insights"] = build_insights(dict(zip(FEATURES, values)))
        return jsonify({"ok": True, "result": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "errors": [f"Prediction failed: {exc}"]}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__})


# --------------------------------------------------------------------------
# 5. FRONT-END (single page, light + dark theme)
# --------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DiaCheck – Diabetes Risk Predictor</title>
<script>
  (function () {
    var t = localStorage.getItem('theme');
    if (!t) t = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
  })();
</script>
<style>
  :root {
    --bg: #f4f6fb; --card: #ffffff; --text: #1b2333; --muted: #667085;
    --border: #e4e8f0; --primary: #4f6df5; --primary-2: #8b5cf6;
    --good: #12b76a; --mid: #f79009; --high: #f04438;
    --shadow: 0 10px 30px rgba(31, 41, 80, .08); --input: #f8f9fc;
  }
  [data-theme="dark"] {
    --bg: #0d111b; --card: #161c2b; --text: #e8ecf5; --muted: #98a2b8;
    --border: #263049; --primary: #7b93ff; --primary-2: #a78bfa;
    --shadow: 0 10px 30px rgba(0, 0, 0, .45); --input: #0f1522;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); transition: background .3s, color .3s;
    background-image: radial-gradient(900px 400px at 85% -10%, rgba(139,92,246,.15), transparent),
                      radial-gradient(700px 400px at -10% 0%, rgba(79,109,245,.15), transparent);
    background-repeat: no-repeat;
  }
  header {
    position: sticky; top: 0; z-index: 20; backdrop-filter: blur(12px);
    background: color-mix(in srgb, var(--bg) 80%, transparent);
    border-bottom: 1px solid var(--border);
  }
  .nav { max-width: 1180px; margin: auto; padding: 14px 20px; display: flex; align-items: center; justify-content: space-between; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.15rem; }
  .logo { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center; color: #fff;
          background: linear-gradient(135deg, var(--primary), var(--primary-2)); font-size: 1.1rem; }
  .icon-btn { border: 1px solid var(--border); background: var(--card); color: var(--text); border-radius: 12px;
              padding: 9px 14px; cursor: pointer; font-size: .9rem; display: inline-flex; gap: 8px; align-items: center; transition: .2s; }
  .icon-btn:hover { transform: translateY(-1px); border-color: var(--primary); }
  main { max-width: 1180px; margin: auto; padding: 28px 20px 60px; }
  .hero h1 { font-size: clamp(1.7rem, 4vw, 2.5rem); margin: 6px 0 8px; line-height: 1.15; }
  .hero h1 span { background: linear-gradient(135deg, var(--primary), var(--primary-2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .hero p { color: var(--muted); max-width: 640px; margin: 0 0 18px; }
  .grid { display: grid; grid-template-columns: 1.55fr 1fr; gap: 22px; align-items: start; }
  @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 22px; box-shadow: var(--shadow); }
  .card + .card { margin-top: 18px; }
  .card h2 { font-size: 1.05rem; margin: 0 0 16px; display: flex; align-items: center; gap: 8px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; align-items: center; }
  .chip { border: 1px solid var(--border); background: var(--input); color: var(--text); padding: 7px 13px; border-radius: 999px; cursor: pointer; font-size: .82rem; transition: .2s; }
  .chip:hover { border-color: var(--primary); color: var(--primary); }
  .progress { height: 6px; background: var(--border); border-radius: 99px; overflow: hidden; margin-bottom: 20px; }
  .progress i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--primary), var(--primary-2)); transition: width .3s; }
  .fields { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px 18px; }
  @media (max-width: 600px) { .fields { grid-template-columns: 1fr; } }
  .field label { display: flex; justify-content: space-between; font-size: .82rem; font-weight: 600; margin-bottom: 6px; }
  .field label em { font-style: normal; font-weight: 500; color: var(--muted); }
  .field .row { display: flex; gap: 10px; align-items: center; }
  input[type=number], select {
    width: 100%; padding: 10px 12px; border-radius: 12px; border: 1px solid var(--border);
    background: var(--input); color: var(--text); font-size: .95rem; outline: none; transition: .2s; font-family: inherit;
  }
  input[type=number]:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 25%, transparent); }
  input[type=number] { width: 96px; flex-shrink: 0; }
  input[type=range] { flex: 1; accent-color: var(--primary); height: 4px; cursor: pointer; }
  .hint { font-size: .74rem; margin-top: 5px; min-height: 1em; color: var(--muted); }
  .hint.good { color: var(--good); } .hint.mid { color: var(--mid); } .hint.high { color: var(--high); }
  .btn { width: 100%; margin-top: 22px; padding: 14px; border: 0; border-radius: 14px; font-size: 1rem; font-weight: 600; color: #fff; cursor: pointer;
         background: linear-gradient(135deg, var(--primary), var(--primary-2)); transition: .2s; box-shadow: 0 8px 20px rgba(79,109,245,.35); }
  .btn:hover { transform: translateY(-2px); } .btn:disabled { opacity: .6; cursor: wait; transform: none; }
  .empty { text-align: center; color: var(--muted); padding: 30px 10px; }
  .empty .big { font-size: 2.6rem; }
  .result { display: none; animation: pop .45s ease; }
  @keyframes pop { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
  .gauge { position: relative; width: 150px; height: 150px; margin: 4px auto 10px; }
  .gauge svg { transform: rotate(-90deg); }
  .gauge circle { fill: none; stroke-width: 12; stroke-linecap: round; }
  .gauge .track { stroke: var(--border); }
  .gauge .bar { stroke: var(--tone); stroke-dasharray: 377; stroke-dashoffset: 377; transition: stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1); }
  .gauge .center { position: absolute; inset: 0; display: grid; place-content: center; text-align: center; }
  .gauge .center b { font-size: 1.9rem; } .gauge .center small { color: var(--muted); font-size: .72rem; }
  .badge { display: inline-block; padding: 5px 14px; border-radius: 99px; font-weight: 700; font-size: .85rem; color: var(--tone); background: color-mix(in srgb, var(--tone) 15%, transparent); }
  .tone-good { --tone: var(--good); } .tone-mid { --tone: var(--mid); } .tone-high { --tone: var(--high); }
  .result h3 { margin: 10px 0 6px; font-size: 1.25rem; }
  .result p { color: var(--muted); line-height: 1.55; margin: 0 0 14px; font-size: .93rem; }
  .probs .p { margin: 8px 0; font-size: .8rem; }
  .probs .p div { display: flex; justify-content: space-between; margin-bottom: 4px; }
  .probs .p .trk { display: block; height: 7px; border-radius: 9px; background: var(--border); overflow: hidden; }
  .probs .p i { display: block; height: 100%; width: 0; background: var(--primary); transition: width 1s; }
  .sub { font-weight: 700; font-size: .85rem; margin: 18px 0 8px; }
  .insight { display: flex; gap: 10px; padding: 9px 11px; border-radius: 12px; background: var(--input); margin-bottom: 7px; font-size: .84rem; border-left: 4px solid var(--c); }
  .insight.good { --c: var(--good); } .insight.mid { --c: var(--mid); } .insight.high { --c: var(--high); }
  .insight b { display: block; } .insight span { color: var(--muted); }
  ul.tips { padding-left: 18px; margin: 0; font-size: .87rem; line-height: 1.7; color: var(--muted); }
  .actions { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
  .history .item { display: flex; justify-content: space-between; align-items: center; font-size: .83rem; padding: 8px 0; border-bottom: 1px dashed var(--border); }
  .history .item:last-child { border: 0; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; background: var(--c); }
  .disclaimer { margin-top: 22px; font-size: .76rem; color: var(--muted); text-align: center; line-height: 1.5; }
  .error { background: color-mix(in srgb, var(--high) 12%, transparent); color: var(--high); padding: 10px 14px; border-radius: 12px; margin-top: 14px; font-size: .85rem; display: none; }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,.4); border-top-color: #fff; border-radius: 50%; display: inline-block; animation: spin .7s linear infinite; vertical-align: -3px; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media print { header, .toolbar, form, .actions, .history { display: none !important; } .grid { display: block; } }
</style>
</head>
<body>
<header>
  <div class="nav">
    <div class="brand"><div class="logo">🩺</div> DiaCheck</div>
    <button class="icon-btn" id="themeBtn" type="button" aria-label="Toggle theme"><span id="themeIcon">🌙</span><span id="themeText">Dark</span></button>
  </div>
</header>

<main>
  <section class="hero">
    <h1>Check your <span>diabetes risk</span> in under a minute</h1>
    <p>Enter your health details and our machine-learning model will estimate whether you are likely to be healthy, pre-diabetic or diabetic — explained in plain language.</p>
  </section>

  <div class="grid">
    <section class="card">
      <h2>📝 Your health details</h2>
      <div class="toolbar">
        <span style="font-size:.8rem;color:var(--muted)">Quick fill:</span>
        <button class="chip" type="button" data-sample="healthy">🟢 Healthy</button>
        <button class="chip" type="button" data-sample="risk">🟠 At risk</button>
        <button class="chip" type="button" data-sample="high">🔴 High risk</button>
        <button class="chip" type="button" id="resetBtn">↺ Reset</button>
      </div>
      <div class="progress"><i id="bar"></i></div>
      <form id="form" novalidate>
        <div class="fields" id="fields"></div>
        <button class="btn" id="submitBtn" type="submit">Predict my risk</button>
        <div class="error" id="error"></div>
      </form>
    </section>

    <aside>
      <section class="card" id="resultCard">
        <h2>📊 Your result</h2>
        <div class="empty" id="empty"><div class="big">🧬</div><p>Fill in the form and press <b>Predict my risk</b> to see your result here.</p></div>
        <div class="result" id="result">
          <div class="gauge" id="gauge">
            <svg width="150" height="150" viewBox="0 0 150 150"><circle class="track" cx="75" cy="75" r="60"/><circle class="bar" id="gaugeBar" cx="75" cy="75" r="60"/></svg>
            <div class="center"><b id="confNum">0%</b><small>model confidence</small></div>
          </div>
          <div style="text-align:center"><span class="badge" id="badge"></span><h3 id="headline"></h3></div>
          <p id="summary"></p>
          <div class="probs" id="probs"></div>
          <div class="sub">What your numbers say</div><div id="insights"></div>
          <div class="sub">What you can do next</div><ul class="tips" id="tips"></ul>
          <div class="actions">
            <button class="chip" type="button" id="printBtn">🖨 Print / Save PDF</button>
            <button class="chip" type="button" id="copyBtn">📋 Copy summary</button>
          </div>
        </div>
      </section>

      <section class="card history" id="historyCard" style="display:none">
        <h2>🕘 Recent predictions <button class="chip" id="clearHist" type="button" style="margin-left:auto">Clear</button></h2>
        <div id="history"></div>
      </section>
    </aside>
  </div>

  <p class="disclaimer">⚠️ This tool is for education and awareness only. It is <b>not</b> a medical diagnosis. Please consult a qualified doctor for any health decision.</p>
</main>

<script>
const CONFIG = {{ config|tojson }};
const LABELS = {
  age: "Age", gender: "Gender", city: "City", bmi: "BMI",
  family_history_diabetes: "Family history of diabetes", physical_activity_level: "Physical activity",
  diet_type: "Diet type", smoking_status: "Smoking", alcohol_consumption: "Alcohol consumption",
  hours_sleep_per_night: "Sleep per night", stress_level: "Stress level",
  fasting_blood_sugar: "Fasting blood sugar", hba1c_level: "HbA1c level",
  blood_pressure_systolic: "Systolic BP (upper)", blood_pressure_diastolic: "Diastolic BP (lower)",
  waist_circumference_cm: "Waist circumference", income_bracket: "Income bracket"
};
const ORDER = ["age","gender","city","bmi","family_history_diabetes","physical_activity_level","diet_type","smoking_status","alcohol_consumption","hours_sleep_per_night","stress_level","fasting_blood_sugar","hba1c_level","blood_pressure_systolic","blood_pressure_diastolic","waist_circumference_cm","income_bracket"];
const SAMPLES = {
  healthy: {age:28,gender:0,bmi:22.1,family_history_diabetes:0,physical_activity_level:0,diet_type:0,smoking_status:2,alcohol_consumption:2,hours_sleep_per_night:8,stress_level:3,fasting_blood_sugar:88,hba1c_level:5.1,blood_pressure_systolic:112,blood_pressure_diastolic:72,waist_circumference_cm:72,income_bracket:2},
  risk:{age:47,gender:1,bmi:27.8,family_history_diabetes:1,physical_activity_level:2,diet_type:1,smoking_status:1,alcohol_consumption:1,hours_sleep_per_night:6,stress_level:6,fasting_blood_sugar:185,hba1c_level:6.3,blood_pressure_systolic:132,blood_pressure_diastolic:84,waist_circumference_cm:98,income_bracket:2},
  high:{age:58,gender:1,bmi:33.5,family_history_diabetes:1,physical_activity_level:1,diet_type:1,smoking_status:0,alcohol_consumption:0,hours_sleep_per_night:5,stress_level:8,fasting_blood_sugar:210,hba1c_level:8.2,blood_pressure_systolic:150,blood_pressure_diastolic:96,waist_circumference_cm:112,income_bracket:1}
};
const $ = (id) => document.getElementById(id);
const form = $('form');

/* ---------- build form ---------- */
ORDER.forEach(name => {
  const wrap = document.createElement('div'); wrap.className = 'field';
  let html = `<label for="${name}">${LABELS[name]}<em id="u_${name}"></em></label>`;
  if (CONFIG.categories[name]) {
    html += `<select id="${name}" name="${name}">` +
      CONFIG.categories[name].map((o, i) => `<option value="${i}">${o}</option>`).join('') + `</select>`;
  } else {
    const c = CONFIG.numeric[name];
    html += `<div class="row"><input type="range" id="${name}_r" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.default}">
             <input type="number" id="${name}" name="${name}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.default}"></div>`;
  }
  html += `<div class="hint" id="h_${name}"></div>`;
  wrap.innerHTML = html; $('fields').appendChild(wrap);
  if (CONFIG.numeric[name]) $('u_' + name).textContent = CONFIG.numeric[name].unit;
});

/* range <-> number sync */
Object.keys(CONFIG.numeric).forEach(n => {
  const r = $(n + '_r'), num = $(n);
  r.addEventListener('input', () => { num.value = r.value; update(); });
  num.addEventListener('input', () => { r.value = num.value; update(); });
});
form.addEventListener('input', update);

/* ---------- live hints ---------- */
function setHint(n, text, cls) { const h = $('h_' + n); h.textContent = text; h.className = 'hint ' + (cls || ''); }
function update() {
  const v = n => parseFloat($(n).value);
  const a1c = v('hba1c_level'), fbs = v('fasting_blood_sugar'), bmi = v('bmi');
  setHint('hba1c_level', a1c >= 6.5 ? 'Diabetic range' : a1c >= 5.7 ? 'Pre-diabetic range' : 'Normal range', a1c >= 6.5 ? 'high' : a1c >= 5.7 ? 'mid' : 'good');
  setHint('fasting_blood_sugar', fbs >= 126 ? 'Diabetic range' : fbs >= 100 ? 'Slightly high' : 'Normal', fbs >= 126 ? 'high' : fbs >= 100 ? 'mid' : 'good');
  setHint('bmi', bmi >= 30 ? 'Obesity' : bmi >= 25 ? 'Overweight' : bmi >= 18.5 ? 'Healthy weight' : 'Underweight', bmi >= 30 ? 'high' : (bmi >= 25 || bmi < 18.5) ? 'mid' : 'good');
  const s = v('blood_pressure_systolic'), d = v('blood_pressure_diastolic');
  setHint('blood_pressure_systolic', (s >= 140 || d >= 90) ? 'High BP' : (s >= 130 || d >= 80) ? 'Slightly elevated' : 'Healthy', (s >= 140 || d >= 90) ? 'high' : (s >= 130 || d >= 80) ? 'mid' : 'good');
  const filled = ORDER.filter(n => $(n).value !== '').length;
  $('bar').style.width = (filled / ORDER.length * 100) + '%';
}
function fill(obj) {
  Object.entries(obj).forEach(([k, val]) => { $(k).value = val; if ($(k + '_r')) $(k + '_r').value = val; });
  update();
}
function defaults() {
  const o = {};
  ORDER.forEach(n => { o[n] = CONFIG.numeric[n] ? CONFIG.numeric[n].default : 0; });
  fill(o);
}
document.querySelectorAll('[data-sample]').forEach(b => b.addEventListener('click', () => fill(SAMPLES[b.dataset.sample])));
$('resetBtn').addEventListener('click', () => { defaults(); $('result').style.display = 'none'; $('empty').style.display = 'block'; $('error').style.display = 'none'; });

/* ---------- theme ---------- */
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t); localStorage.setItem('theme', t);
  $('themeIcon').textContent = t === 'dark' ? '☀️' : '🌙'; $('themeText').textContent = t === 'dark' ? 'Light' : 'Dark';
}
$('themeBtn').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
applyTheme(document.documentElement.dataset.theme);

/* ---------- predict ---------- */
let last = null;
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('submitBtn'), err = $('error'); err.style.display = 'none';
  const payload = {}; ORDER.forEach(n => payload[n] = $(n).value);
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>Analysing…';
  try {
    const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (!data.ok) throw new Error(data.errors.join(' '));
    last = data.result; showResult(data.result); saveHistory(data.result, payload);
    if (innerWidth < 960) $('resultCard').scrollIntoView({ behavior: 'smooth' });
  } catch (ex) { err.textContent = ex.message; err.style.display = 'block'; }
  btn.disabled = false; btn.textContent = 'Predict my risk';
});

function showResult(r) {
  $('empty').style.display = 'none';
  const box = $('result'); box.style.display = 'block'; box.className = 'result tone-' + r.tone;
  box.style.animation = 'none'; box.offsetHeight; box.style.animation = '';
  $('badge').textContent = r.label; $('headline').textContent = r.headline; $('summary').textContent = r.text;
  const conf = r.confidence || 0;
  $('gaugeBar').style.strokeDashoffset = 377;
  setTimeout(() => { $('gaugeBar').style.strokeDashoffset = 377 * (1 - conf / 100); }, 60);
  let n = 0; const step = conf / 40;
  const t = setInterval(() => { n = Math.min(conf, n + step); $('confNum').textContent = Math.round(n) + '%'; if (n >= conf) clearInterval(t); }, 25);
  $('probs').innerHTML = Object.entries(r.probabilities || {}).map(([k, v]) =>
    `<div class="p"><div><em style="font-style:normal">${k}</em><b>${v}%</b></div><span class="trk"><i data-w="${v}"></i></span></div>`).join('');
  setTimeout(() => document.querySelectorAll('.probs i').forEach(i => i.style.width = i.dataset.w + '%'), 80);
  $('insights').innerHTML = r.insights.map(i => `<div class="insight ${i.level}"><div><b>${i.title}</b><span>${i.text}</span></div></div>`).join('');
  $('tips').innerHTML = r.tips.map(t => `<li>${t}</li>`).join('');
}

/* ---------- history, print, copy ---------- */
const COLORS = { good: 'var(--good)', mid: 'var(--mid)', high: 'var(--high)' };
function getHist() { try { return JSON.parse(localStorage.getItem('hist') || '[]'); } catch (e) { return []; } }
function saveHistory(r, p) {
  const h = getHist(); h.unshift({ label: r.label, tone: r.tone, a1c: p.hba1c_level, fbs: p.fasting_blood_sugar, t: new Date().toLocaleString() });
  localStorage.setItem('hist', JSON.stringify(h.slice(0, 5))); renderHist();
}
function renderHist() {
  const h = getHist(); $('historyCard').style.display = h.length ? 'block' : 'none';
  $('history').innerHTML = h.map(i => `<div class="item"><div><span class="dot" style="--c:${COLORS[i.tone]}"></span><b>${i.label}</b><div style="color:var(--muted);font-size:.74rem;margin-left:17px">HbA1c ${i.a1c}% · Sugar ${i.fbs} · ${i.t}</div></div></div>`).join('');
}
$('clearHist').addEventListener('click', () => { localStorage.removeItem('hist'); renderHist(); });
$('printBtn').addEventListener('click', () => window.print());
$('copyBtn').addEventListener('click', async () => {
  if (!last) return;
  const txt = `DiaCheck result: ${last.label}\n${last.headline}\n${last.text}`;
  try { await navigator.clipboard.writeText(txt); $('copyBtn').textContent = '✅ Copied'; } catch (e) { $('copyBtn').textContent = 'Copy failed'; }
  setTimeout(() => $('copyBtn').textContent = '📋 Copy summary', 1800);
});

update(); renderHist();
</script>
</body>
</html>
"""

# Vercel looks for the `app` object above. This block is only for local testing.
if __name__ == "__main__":
    app.run(debug=True)
