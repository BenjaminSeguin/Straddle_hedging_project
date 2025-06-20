# 📈 Straddle Hedging Optimization via Simulation

This project implements a simulation-based framework to optimize the hedging of a short straddle position on the S&P 500. It evaluates how different hedging strategies perform in minimizing the replication error between the hedged portfolio and the short straddle payoff.

📂 All relevant scripts are included in the repository.  
▶️ To run the main experiment, execute `main_simulation.py`.

> 📝 This project was completed independently and is intended to replicate academic approaches to dynamic option hedging.

---

## 🎯 Objective

To minimize the **hedging error** when replicating a short straddle using different daily hedging strategies under realistic market dynamics. The goal is to understand how classical and ML-enhanced approaches compare in terms of performance and robustness.

---

## ⚙️ Methodology

### 1. **Option Setup**
- Underlying: S&P 500
- Instrument: 1-month short straddle (ATM call + ATM put)
- Hedge instruments: Underlying asset + risk-free asset

### 2. **Simulation Framework**
- Generate 1000 Monte Carlo price paths per month (2007–2023)
- Use historical volatility + drift estimates to calibrate the stochastic process
- Simulate daily hedging using various strategies over the life of the option

### 3. **Hedging Strategies Compared**
- **Delta Hedging** (Black-Scholes-based)
- **Static Hedging** (no rebalancing)
- **ML-Driven Hedging** using Random Forests to predict optimal hedge ratios

### 4. **Evaluation Metrics**
- Final replication error distribution (Mean, Std Dev)
- Hedging cost over time

---

## 📊 Results Summary

- ML and optimization-based strategies consistently outperformed naive delta hedging in volatile regimes.
- Delta hedging tended to underperform in presence of skew and stochastic volatility.
- Random Forest-based hedge ratio prediction was particularly effective in reducing extreme tail errors.

---

## 📦 Tools & Dependencies

- Python, NumPy, Pandas, Matplotlib
- Scikit-learn (Random Forests)

---

## 🚀 Highlights

- Applies dynamic programming concepts and ML to real-world derivatives risk management
- Implements a robust Monte Carlo engine for path simulation and strategy comparison
- Bridges quantitative finance theory with empirical ML techniques

---

Feel free to open an issue or reach out for questions or collaborations.
