# Bregman Superquantiles in Heavy-Tailed Models
Implementation and analysis for Bregman superquantiles in heavy-tailed models (Manou-Abi &amp; Slaoui). Studies empirical, kernel, and recursive estimators, asymptotic theory, parameter estimation for GPD/stable laws, and financial/insurance data applications.

[![R-project](https://img.shields.io/badge/Language-R-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for the paper:  
**"Bregman superquantiles in heavy-tailed models: kernel and recursive estimators, asymptotic theory and parameter estimation"**  
*Solym M. Manou-Abi and Yousri Slaoui*  
Laboratoire de Mathématiques et Applications (UMR CNRS 7348), Université de Poitiers, France.

---

## 📌 Overview

Superquantiles—also known as Conditional Value-at-Risk (CVaR)—and their Bregman generalizations average the tail of a distribution beyond a given quantile through a strictly convex generator. 

This repository provides the full R implementation, simulation studies, and real-world data applications comparing three types of estimators:
1. **Empirical Estimator:** Standard non-parametric baseline.
2. **Kernel-Smoothed Estimator:** Smooth non-parametric estimation with second-order bias control.
3. **Recursive (Robbins–Monro) Estimator:** Designed for data streams, reaching optimal efficiency via Newton-type gains or Polyak–Ruppert averaging.

---

## 🎯 Key Highlights

* **Asymptotic Theory:** Almost sure consistency for any bandwidth $h_n \to 0$ and Central Limit Theorems (CLT) highlighting smoothing-bias regimes.
* **Heavy-Tailed Parameter Estimation:** Estimation equations for **Generalized Pareto Distributions (GPD)** and **Stable Laws**, including exact quantile–superquantile representations under quadratic generators.
* **Infinite Variance Stabilization:** Logarithmic generators restore standard Gaussian inference in heavy-tailed settings with infinite variance.
* **Empirical Validation:** Applications to benchmark datasets including Danish fire losses, S&P 500 return losses, and Boeing stock losses.

---

## 📂 Repository Structure

```text
.
├── contam_p0.8.json                   # Configuration files for contamination scenarios (e.g., GPD p = 0.8)
├── gpd_contam.py                      # Python scripts for data contamination and simulation generation
├── stable_parameters_superquantiles.Rmd # R Markdown: Estimation and parameters for stable laws
├── application_superquantiles.Rmd     # R Markdown: Real-world financial/insurance data applications
├── bregman_simulation_study.Rmd       # R Markdown: Monte Carlo simulations and asymptotic checks
└── README.md                          # Repository documentation
