<div align="center">

# 🏭 Multi-Class Fault Diagnosis System
### Deep Learning for Industrial Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-Classification-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-green)

[View Architecture](#-neural-network-architecture) • [Fault Classes](#-fault-classification-types)

</div>

---

## 📖 Overview

This project implements a **Multi-Class Classification System** designed to identify specific mechanical failure modes in industrial machinery. Unlike simple "Pass/Fail" detectors, this model acts as a diagnostic tool, pinpointing the *root cause* of the anomaly.

Using a Deep Neural Network (DNN), the system analyzes sensor data patterns (Vibration, Temperature, etc.) to classify equipment into one of four distinct health states. This capability is critical for **Prescriptive Maintenance**—knowing exactly *what* to fix before dispatching a technician.

---

## ⚡ Key Features

* **🔍 4-Class Diagnostics:** Capable of distinguishing between Normal, Bearing Failure, Misalignment, and Imbalance.
* **🧠 Softmax Probability:** Outputs a confidence score for *each* fault type, allowing for nuanced decision-making.
* **📈 Synthetic Data Engine:** Simulates realistic sensor signatures:
    * *Bearing Failure:* High Vibration + Moderate Temp.
    * *Misalignment:* Moderate Vibration + High Temp.
    * *Imbalance:* Extreme Vibration + Normal Temp.
* **📊 Granular Evaluation:** Includes per-class accuracy metrics and confusion matrices to identify "hard-to-detect" faults.

---

## 🏭 Fault Classification Types

The model is trained to recognize the following distinct signatures:

| Class ID | Fault Name | Sensor Characteristic | Severity |
| :--- | :--- | :--- | :--- |
| **0** | 🟢 **Normal** | Baseline readings (0-10) | None |
| **1** | 🟡 **Bearing Failure** | High Vibration, Moderate Heat | Medium |
| **2** | 🔴 **Misalignment** | Moderate Vibration, High Heat | High |
| **3** | 🟣 **Imbalance** | Very High Vibration, Normal Heat | Critical |

---

## 🏗 Neural Network Architecture

The model utilizes a Feed-Forward Neural Network with a Softmax output layer for multi-class probability distribution:

```
graph LR
    A[Input Layer<br/>(4 Features)] --> B[Dense Layer<br/>(64 Neurons, ReLU)]
    B --> C[Dense Layer<br/>(32 Neurons, ReLU)]
    C --> D[Dense Layer<br/>(16 Neurons, ReLU)]
    D --> E[Output Layer<br/>(Softmax)]
    E --> F[Class Probabilities<br/>[P0, P1, P2, P3]]
```
