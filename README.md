# LSTM Time Series Forecasting

##  Overview

This project implements an **LSTM-based time series forecasting system** for predicting monthly airline passenger demand using historical passenger data.

The system uses a **12-month historical lookback window** to learn temporal patterns and seasonal trends in airline passenger traffic. The dataset is transformed into sliding-window sequences, normalized using Min-Max Scaling, and then used to train a stacked LSTM neural network.

The forecasting performance is evaluated using **Root Mean Squared Error (RMSE)**.

---

##  Objectives

* Forecast future airline passenger demand using deep learning.
* Capture temporal and seasonal patterns from historical passenger data.
* Convert the time series into supervised learning sequences using a sliding window.
* Normalize the dataset using Min-Max Scaling.
* Train a stacked LSTM model for time series forecasting.
* Evaluate the model using RMSE.
* Visualize training/validation loss and actual vs predicted passenger values.

---

##  Dataset

The project uses the **Airline Passenger Dataset**, which contains monthly airline passenger records from **January 1949 to December 1960**.

### Dataset Details

| Property        | Value                        |
| --------------- | ---------------------------- |
| Records         | 144                          |
| Frequency       | Monthly                      |
| Time Period     | January 1949 – December 1960 |
| Target Feature  | Passengers                   |
| Unit            | Thousands of passengers      |
| Lookback Window | 12 months                    |

Example:

```text
Month,Passengers
1949-01,112
1949-02,118
1949-03,132
1949-04,129
1949-05,121
```

The `Passengers` column is used as the target variable for forecasting.

---

##  Methodology

The forecasting pipeline consists of the following stages:

```text
Airline Passenger Dataset
          ↓
      Data Loading
          ↓
    Min-Max Scaling
          ↓
  12-Month Lookback
          ↓
   Sliding Window
          ↓
   Train/Test Split
          ↓
     LSTM Model
          ↓
     Predictions
          ↓
   Inverse Scaling
          ↓
    RMSE Evaluation
          ↓
     Visualization
```

### 1. Data Loading

The dataset is loaded using Pandas, and the passenger values are extracted as the time series used by the model.

### 2. Data Scaling

The passenger values are normalized between **0 and 1** using `MinMaxScaler`.

This helps the LSTM model work with values within a consistent numerical range.

### 3. Sliding Window

A **12-month lookback window** is used to create the input sequences.

For example:

```text
Previous 12 Months → Next Month
```

The process continues across the entire dataset, producing multiple input-output sequences for the LSTM.

The resulting dataset contains:

```text
X shape: (132, 12)
y shape: (132,)
```

### 4. Train-Test Split

The generated sequences are divided into:

* **80% training data**
* **20% testing data**

The data is then reshaped into the three-dimensional format required by an LSTM:

```text
(samples, time steps, features)
```

Therefore, every sequence contains:

```text
12 time steps
1 feature
```

### 5. LSTM Model

The model consists of two stacked LSTM layers followed by a Dense output layer.

```text
Input
  ↓
LSTM (50 units)
  ↓
LSTM (50 units)
  ↓
Dense (1 unit)
  ↓
Predicted Passenger Count
```

### Model Configuration

| Parameter      | Value              |
| -------------- | ------------------ |
| LSTM Layers    | 2                  |
| Units per LSTM | 50                 |
| Activation     | ReLU               |
| Output Layer   | Dense              |
| Output Units   | 1                  |
| Optimizer      | Adam               |
| Loss Function  | Mean Squared Error |
| Epochs         | 100                |
| Batch Size     | 8                  |
| Lookback       | 12 months          |

### 6. Training

The LSTM is trained using the historical training sequences.

Training and validation loss are recorded throughout the training process to observe how the model learns over the epochs.

### 7. Forecasting

After training, the model generates passenger predictions for both the training and testing sequences.

Since the original data was normalized before training, the predictions are transformed back to the original passenger scale using the inverse transformation of the Min-Max scaler.

### 8. Evaluation

Forecasting performance is evaluated using **Root Mean Squared Error (RMSE)**.

The formula is:

```text
RMSE = √(Mean((Actual − Predicted)²))
```

A lower RMSE indicates that the predicted passenger values are closer to the actual values.

---

##  Results

The reference experiment reports the following results:

| Dataset  |  RMSE |
| -------- | ----: |
| Training | 15.42 |
| Testing  | 24.87 |

The testing RMSE indicates the model's forecasting error on previously unseen sequences.

> Note: LSTM training can be stochastic, so RMSE values may vary slightly between different runs depending on initialization and the execution environment.

---

##  Visualizations

### Training vs Validation Loss

The experiment plots training and validation MSE loss across the training epochs.

This helps analyze:

* Model learning
* Convergence
* Generalization
* Possible overfitting

### Actual vs Predicted Passengers

The testing results are visualized by comparing:

```text
Actual Passenger Values
        vs
Predicted Passenger Values
```

This provides a visual indication of how closely the LSTM follows the actual passenger demand trend.

---

##  Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **TensorFlow**
* **Keras**
* **Jupyter Notebook**

---

##  Project Structure

```text
LSTM-Time-Series-Forecasting/
│
├── airline_passengers.csv
├── experiment_6.ipynb
├── README.md
└── requirements.txt
```

---

##  Installation

Clone the repository:

```bash
git clone <repository-url>
cd LSTM-Time-Series-Forecasting
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and run the cells sequentially.


---

##  Future Improvements

Possible improvements to the current implementation include:

* Comparing LSTM with ARIMA and other forecasting methods.
* Testing GRU-based architectures.
* Using Bidirectional LSTM.
* Adding dropout and regularization.
* Performing hyperparameter optimization.
* Implementing multi-step future forecasting.
* Adding additional features that influence passenger demand.
* Deploying the forecasting model through a web application.
* Comparing RMSE with MAE and MAPE.

---

##  References

* Airline Passenger Dataset — monthly airline passenger observations from 1949–1960.
* Related time-series forecasting implementations using the Airline Passenger Dataset.
* TensorFlow/Keras documentation for LSTM-based sequence modeling.
