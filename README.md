---
title: MRT-3 Forecast System
emoji: 🚇
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# MRT-ADAPT
**Passenger Volume Forecasting System for MRT-3 Stations**

## Overview
MRT-ADAPT is a data-driven web application designed to analyze, track, and forecast passenger volume across MRT-3 stations. By leveraging advanced machine learning sequence models, this system provides accurate predictive logic to anticipate passenger traffic, helping to optimize transit management and passenger flow analysis.

## Key Features
* **Predictive Forecasting:** Utilizes Long Short-Term Memory (LSTM) and trained 26 models for Northbound and Southbound for 13 MRT stations.
* **Interactive Dashboard:** A responsive frontend interface for visualizing historical data and future passenger traffic trends.
* **Robust Data Management:** Secure and efficient data persistence handling large datasets of transit logs.

## Tech Stack
This project was built utilizing a comprehensive full-stack and machine learning architecture:

**Backend & Architecture**
* **Language:** Python
* **Framework:** Flask
* **Database:** SQLite

**Frontend**
* HTML5, CSS3, Vanilla JavaScript

**Machine Learning & Data Processing**
* **Models:** LSTM (Long Short-Term Memory)
* **Libraries:** Pandas, NumPy, Scikit-Learn

## 🚀 Getting Started (Local Development)

To run this application locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mrt-adapt.git
cd mrt-adapt