# 🚗 EV Car-Following Behavior Modeling

This repository provides classical and machine learning models to simulate and predict electric vehicle (EV) car-following behavior under varying gap settings. The models include:

- **Classical physics-based models**: IDM, OVM, OVRV, and CACC
- **Machine Learning models**: Random Forest for acceleration and spacing prediction
- **Evaluation across gap settings**: Medium, Long, and Extra Long(XLong)

---

## 📁 Project Structure

```bash
├── data/
├── notebook/
│   ├── IDM_calibration.ipynb
│   ├── OVM_calibration.ipynb
│   ├── OVRV_calibration.ipynb
│   ├── CACC_calibration.ipynb
│   ├── acc_prediction.ipynb
│   ├── space_prediction.ipynb
├── REPORTS/
├── final_plot_utils.py 
├── models/
├── environment.yml
└── README.md

```


---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/ev-car-following-models.git
cd car-following-behavior-EVACC-paper-recreation

### 2. Create Virtual Environment
. shells/install.sh
````
## 📊 Usage

Run each notebook under notebook/ to calibrate a physics-based model using RMSE minimization:

```IDM_calibration.ipynb```

```OVM_calibration.ipynb```

```OVRV_calibration.ipynb```

```CACC_calibration.ipynb```

Each notebook outputs:

Best-fit parameters

RMSE plots

Simulated vs actual spacing/speed figures

Results are saved in REPORTS/final_results/.

## 🤖 Machine Learning Model Training
### 1. Acceleration Prediction
```bash
notebook/acc_prediction.ipynb
```
### 2. Spacing Prediction
```bash
notebook/space_prediction.ipynb
```
## ML models are saved as:
```bash
notebook/rf_model_acc.pkl
notebook/rf_model_spacing.pkl
```
## 📈 Generate Final Evaluation Plots
```bash
python final_plot_utils.py
```
## 📁 Data Format
Expected columns in your dataset:
```bash
Time, Speed Leader, Speed Follower, Spacing, gap_setting
```
The pipeline automatically computes:

- delta_v – relative speed

- acc_follower – follower vehicle acceleration

- dt – time step

## ✅ Output Summary
- RMSE-calibrated parameters

- Time-series spacing and speed plots

- Residual analysis (ML)

- Machine learning predictions vs classical models

- Visuals for Medium, Long, and XLong gap scenarios
