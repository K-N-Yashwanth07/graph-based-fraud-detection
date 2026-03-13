# Graph-Based Financial Fraud Detection System

## Overview
Financial fraud detection is a major challenge in online transactions and banking systems. Traditional fraud detection techniques analyze transactions individually, but many fraudulent activities occur through coordinated patterns between accounts.

This project models financial transactions as a **graph network** where accounts are represented as nodes and transactions as edges. Using **graph analytics and anomaly detection**, the system identifies suspicious accounts and potential fraud rings.

The project uses **NetworkX for graph modeling** and **Isolation Forest for anomaly detection** to detect abnormal transaction behavior.

---

## Features

- Transaction network modeling using graph structures
- Graph feature extraction (Degree, PageRank, Centrality)
- Fraud detection using Isolation Forest
- Identification of suspicious accounts
- Visualization of transaction networks
- Export of suspicious accounts as CSV reports

---

## Tech Stack

- Python
- NetworkX
- Scikit-learn
- Pandas
- Matplotlib

---

## Project Structure
graph-based-fraud-detection
│
├── outputs
│ └── suspicious_accounts.csv
│
├── main.py
├── test.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore

---


## Dataset

This project uses the **IEEE Fraud Detection Dataset**.

Download the dataset from Kaggle:

https://www.kaggle.com/competitions/ieee-fraud-detection

After downloading, place the dataset inside:
data/train_transaction.csv


## Installation

Clone the repository:
git clone https://github.com/K-N-Yashwanth07/graph-based-fraud-detection.git

cd graph-based-fraud-detection

Install required libraries:
pip install -r requirements.txt

---

## Running the Project

Run the fraud detection system:
python main.py

The program will:

1. Load transaction data
2. Build the transaction graph
3. Extract graph features
4. Apply anomaly detection using Isolation Forest
5. Identify suspicious accounts
6. Generate a fraud detection report

---

## Example Output

Example suspicious accounts detected:
Suspicious Accounts:
[18132, 12695, 2803, 4461, 11839]

Generated report:
outputs/suspicious_accounts.csv

---

## Fraud Detection Approach

The detection pipeline follows these steps:

1. Transaction data preprocessing
2. Graph construction from transactions
3. Feature extraction from network structure
4. Anomaly detection using Isolation Forest
5. Identification of suspicious nodes

Graph-based analysis allows detection of **fraud rings and coordinated fraud activities** that traditional rule-based systems may miss.

---

## Future Improvements

Possible improvements for this project include:

- Real-time fraud detection system
- Graph Neural Networks (GNN) for fraud prediction
- Interactive fraud visualization dashboard
- Integration with streaming transaction data

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Yashwanth K**

GitHub:  
https://github.com/K-N-Yashwanth07
