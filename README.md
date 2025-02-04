# **Transaction Data Analysis - Quod Financial Test**

## **Overview**

This repository contains a **transaction data analysis application** built using **StreamLit** for visualization and predictive modeling. The project includes:

- **Data loading and preprocessing** to clean and structure transactions.
- **Visualizations** to explore transaction trends, seasonality, and customer behaviors.
- **Predictive modeling** using **XGBoost, Random Forest, and LSTM** to forecast transaction behaviors.
- **Feature importance analysis** to understand key drivers of transaction volume.

The analysis and results are detailed in **Jupyter notebooks** under `notebooks/`, while the StreamLit application provides an **interactive dashboard** for visualization and model evaluation.

---

## **Project Structure**

```bash
Quod_THA_Test/
│── src/
│   ├── app.py                     # Streamlit app
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── models.py                   # Model training and evaluation
│   ├── visualization.py             # Graph and visualization functions
│   ├── utils.py                     # Utility functions
│── notebooks/                       # Jupyter notebooks for analysis
│   ├── main.ipynb                   # Detailed analysis and results
│── data/                            # Data storage (not included in repo)
│── requirements.txt                 # Python dependencies
│── README.md                        # Project documentation
```

---

## **Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/yourusername/Quod_THA.git
cd Quod_THA
```

### **2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **Running the Application**

### **1. Run the Streamlit App**

```bash
cd src
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501/`, allowing you to explore:

- **Transaction visualizations**
- **Seasonality trends**
- **Prediction models (XGBoost, Random Forest, LSTM)**

---

## **Key Findings**

### **1. Data Insights**

- The transaction data shows a **clear seasonal pattern**, with peaks in October and November.
- **Certain customers contribute significantly** more transactions than others.
- **Top-selling products** remain stable over time but shift slightly every six months.

### **2. Model Performance**

| Model         | MAE   | MSE     | R²     |
| ------------- | ----- | ------- | ------ |
| XGBoost       | 4.92  | 398.99  | 0.894  |
| Random Forest | 4.18  | 280.54  | 0.925  |
| LSTM          | 34.09 | 5916.78 | -0.575 |

**Random Forest outperforms other models**, achieving the lowest MAE and highest R². However, LSTM could improve with hyperparameter tuning and additional data transformations.

---

## **Customization**

### **Adjust Hyperparameters**

Modify `train_model()` in `models.py`:

```python
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
```

### **Extend Feature Engineering**

Update `prepare_data()` in `models.py` to include new lag features:

```python
df['transactions_lag_4'] = df.groupby('customer_id')['transaction_count'].shift(4)
```

---

## **Next Steps**

- **Optimize LSTM model** with a more complex architecture.
- **Enhance feature engineering** with additional temporal and categorical variables.
- **Expand model evaluation** with cross-validation and confidence intervals.
- **Try alternative models** like ARIMA, Prophet, or CatBoost for comparison.

---

## **Author**

- **Tristan PERROT**
- Data Scientist | Machine Learning Engineer
- [LinkedIn](https://www.linkedin.com/in/tristanperrot/) | [GitHub](https://github.com/yourusername)
