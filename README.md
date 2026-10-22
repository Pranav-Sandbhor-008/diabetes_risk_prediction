<div align="center">

# 🩺 DiaCheck – Diabetes Risk Predictor

**A modern, interactive web app that predicts diabetes risk using a Support Vector Machine (SVC) model and explains the result in plain, human-friendly language.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikit-learn&logoColor=white)
![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?logo=vercel&logoColor=white)

[🌐 Live Demo](YOUR-VERCEL-LINK) · [🐞 Report Bug](https://github.com/Pranav-Sandbhor-008/YOUR-REPO-NAME/issues) · [💡 Request Feature](https://github.com/Pranav-Sandbhor-008/YOUR-REPO-NAME/issues)

</div>

---

## 📖 About the Project

**DiaCheck** takes 17 health and lifestyle details from the user and uses a trained **SVC (Support Vector Classifier)** model to estimate whether the person is likely to be:

| Result | Meaning |
|---|---|
| 🟢 **No Diabetes** | Low likelihood of diabetes |
| 🟠 **Pre-Diabetes** | Blood sugar is higher than ideal but not yet diabetic |
| 🔴 **Diabetes** | High likelihood of diabetes, consult a doctor |

Instead of showing a bare number, the app gives a **plain-language explanation**, a **breakdown of each health value**, and **practical next-step tips**.

---

## ✨ Features

- 🎨 **Modern, professional UI** with a smooth **Light / Dark theme** toggle (remembers your choice and follows the system theme)
- 🎚️ **Interactive form** with sliders linked to number inputs
- 💡 **Live health hints** while typing (for example HbA1c, blood sugar, BMI and blood pressure ranges)
- 📊 **Animated confidence ring** and per-class probability bars
- 🗣️ **Human-friendly results** with a clear explanation and recommendations
- 🧾 **"What your numbers say"** section with colour-coded insights
- ⚡ **Quick-fill samples** (Healthy / At risk / High risk) to try the app instantly
- 🕘 **Recent predictions history** saved in the browser
- 🖨️ **Print / Save as PDF** and 📋 **Copy summary**
- 🔌 **REST API** (`/predict`, `/health`)
- 📱 **Fully responsive** for mobile, tablet and desktop
- ✅ **Server-side input validation**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Backend | Flask |
| Machine Learning | scikit-learn (SVC, RBF kernel) |
| Numerical | NumPy, Joblib |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Deployment | Vercel |

---

## 🧠 Model Details

- **Algorithm:** Support Vector Classifier (`SVC`, RBF kernel)
- **Input features (17):**

| Category | Features |
|---|---|
| Personal | `age`, `gender`, `city`, `income_bracket` |
| Body | `bmi`, `waist_circumference_cm` |
| Lifestyle | `physical_activity_level`, `diet_type`, `smoking_status`, `alcohol_consumption`, `hours_sleep_per_night`, `stress_level` |
| Medical | `family_history_diabetes`, `fasting_blood_sugar`, `hba1c_level`, `blood_pressure_systolic`, `blood_pressure_diastolic` |

- **Output classes:** `0 → Diabetes`, `1 → No Diabetes`, `2 → Pre-Diabetes`
- **Confidence score:** calculated from the SVC decision scores, so it is an approximation and not a calibrated probability.

---

## 📂 Project Structure

```
├── app.py              # Flask app: routes, prediction logic, UI
├── svc.pkl             # Trained SVC model
├── requirements.txt    # Python dependencies
├── vercel.json         # Vercel deployment configuration
└── README.md           # Project documentation
```

---

## 🚀 Getting Started (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/Pranav-Sandbhor-008/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. 🎉

---

## ☁️ Deploy on Vercel

1. Push this project to GitHub (make sure `vercel.json` is included).
2. Go to [vercel.com](https://vercel.com) → **Add New → Project**.
3. Import your GitHub repository.
4. Click **Deploy**. Vercel detects the Flask app automatically.

Or use the CLI:
```bash
npm i -g vercel
vercel
```

---

## 🔌 API Usage

### `POST /predict`

**Request**
```json
{
  "age": 47, "gender": 1, "city": 3, "bmi": 27.8,
  "family_history_diabetes": 1, "physical_activity_level": 2,
  "diet_type": 1, "smoking_status": 1, "alcohol_consumption": 1,
  "hours_sleep_per_night": 6, "stress_level": 6,
  "fasting_blood_sugar": 185, "hba1c_level": 6.3,
  "blood_pressure_systolic": 132, "blood_pressure_diastolic": 84,
  "waist_circumference_cm": 98, "income_bracket": 2
}
```

**Response**
```json
{
  "ok": true,
  "result": {
    "label": "Pre-Diabetes",
    "headline": "Pre-diabetes pattern detected",
    "confidence": 62.4,
    "probabilities": { "Diabetes": 14.1, "No Diabetes": 23.5, "Pre-Diabetes": 62.4 },
    "insights": [ ... ],
    "tips": [ ... ]
  }
}
```

### `GET /health`
Returns the service status: `{"status": "ok", "model": "SVC"}`

---

## 📸 Screenshots

> Add your screenshots here after deployment.

| Light Mode | Dark Mode |
|---|---|
| ![Light](screenshots/light.png) | ![Dark](screenshots/dark.png) |

---

## ⚠️ Disclaimer

This project is built for **educational and awareness purposes only**. It is **not a medical diagnosis** and must not replace professional medical advice. Always consult a qualified doctor for any health decision.

---

## 🔮 Future Improvements

- [ ] Retrain the model with feature scaling (`StandardScaler` pipeline) for better accuracy
- [ ] Add calibrated probabilities (`probability=True`)
- [ ] Multi-language support (Hindi, Marathi)
- [ ] Downloadable PDF health report
- [ ] Model explainability (SHAP / feature importance)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the project
2. Create your branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m "Add AmazingFeature"`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 👨‍💻 Author

**Pranav Sandbhor**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pranavsandbhor/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/Pranav-Sandbhor-008)

---

<div align="center">

⭐ If you found this project useful, please give it a star! ⭐

</div>
