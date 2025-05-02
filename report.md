# Soccer Match Outcome Prediction - Project Report

## 1. Introduction
This project aims to predict the outcome of soccer matches (Home Win, Draw, Away Win) using historical match data, betting odds, and team/player attributes. The goal is to develop a robust prediction model that can accurately forecast match results

## 2. Methodology
Our approach to building the prediction model followed a systematic process:

1. **Exploratory Data Analysis (EDA)**
   - Initial data exploration to understand the structure and characteristics of our dataset
   - Identification of key features and potential relationships
   - Analysis of data quality and missing values
   - Visualization of important patterns and trends in the data

2. **Data Preparation**
   - Implementation of the `prepare_football_data.py` script
   - Data cleaning and preprocessing
   - Feature engineering to create meaningful predictors
   - Handling of missing values and outliers
   - Creation of a standardized dataset ready for modeling

3. **Model Development**
   - Started with a Random Forest model as our baseline
   - Implementation of model training and validation procedures
   - Performance evaluation using various metrics
   - Future plans to implement XGBoost for comparison

## 3. Results: Random forest

### 3.1 Model selection and headline metrics
After an exhaustive grid‑search (15 parameter grids × 2‑fold CV, 30 fits) the best Random Forest used:

```text
n_estimators       = 200
max_depth          = 30
max_features       = "sqrt"
min_samples_split  = 5
min_samples_leaf   = 2
```

| Data split                      |  Accuracy |  Log‑loss |
| ------------------------------- | :-------: | :-------: |
| Cross‑validation (mean, 2‑fold) | **0.629** |     —     |
| Test set (n = 5 196)            | **0.670** | **0.836** |

**Classification report (test set):**
| Class            | Precision | Recall   | F1‑score | Support |
| ---------------- | --------- | -------- | -------- | ------- |
| **Home Win**     | 0.69      | **0.87** | 0.77     | 2 384   |
| **Away Win**     | **0.67**  | 0.73     | 0.70     | 1 493   |
| **Draw**         | 0.56      | **0.24** | 0.33     | 1 319   |
| **Macro avg**    | 0.64      | 0.61     | 0.60     | 5 196   |
| **Weighted avg** | 0.65      | 0.67     | 0.64     | 5 196   |

### 3.2 Error distribution
**Key insight – Draws are the pain‑point**
* 65 % of true draws are predicted as Away Wins
* 35 % are predicted as Home Wins
* < 1 % are predicted as draws

This single class accounts for the majority of the macro‑recall deficit.

![Confusion matrix](docs/results/random_forest/random_forest_confusion_matrix.png)

The matrix confirms the “draw problem”:  
* **352 / 1 319 (27 %)** draws are called **Away Win**; **195 / 1 319 (15 %)** become **Home Win**.  
* Only **311** draws are caught correctly → recall = 0.24 (see Section 3.1).  
Addressing this single class would lift macro‑recall by ~13 pp.

---

### 3.3 Probability quality  

#### 3.3.1 Calibration  
![Calibration curve](docs/results/random_forest/random_forest_calibration.png)

Predicted probabilities for wins hug the diagonal ⇒ **well‑calibrated**.  
Draws are **over‑confident** between 0.4–0.6; the model assigns them 50 % when the empirical frequency is closer to 35 %.

#### 3.3.2 Where confidence helps (and where it doesn’t)  
![PDF by correctness](docs/results/random_forest/random_forest_pred_prob_distribution.png)

* Correct predictions are biased away from the uninformed prior (0.33).  
* Draw errors peak at *p* ≈ 0.28 – the forest “senses” uncertainty but still picks a side.

---

### 3.4 What drives the model?  
![Top‑15 feature importance](docs/results/random_forest/random_forest_feature_importance.png)

* **Elo ratings** (`home_elo`, `away_elo`, `elo_diff`) explain **≈ 27 %** of split gain.  
* **Market odds** (home/away/draw implied probabilities, `team_draw_rate`) occupy the next tier.  
* **H2H** goal metrics and **recent form** finish the list — small individually but valuable in ensemble.

> **Practical reading:** when Elo says the teams are equal and the market is split, the model is effectively guessing — exactly the scenario that generates most draw errors.

---

### 3.5 Class‑specific precision–recall  
![PR curves](docs/results/random_forest/random_forest_precision_recall.png)

* **Home Wins** keep ≥ 0.80 precision up to 50 % recall — ideal for selective betting strategies.  
* **Away Wins** trail slightly but are still exploitable.  
* **Draws** tumble below 0.50 precision once recall exceeds 0.15 — reinforcing the need for a dedicated draw detector or cost‑sensitive training.

---

### 3.6 Temporal robustness  
![Accuracy by year](docs/results/random_forest/random_forest_temporal_accuracy.png)

Accuracy drifts from **0.73 (2008)** to **0.63 (2016)**.

* **Concept drift** – tactical evolution, player turnover and rule changes dilute historical patterns.  
* **Data mix** – later seasons add leagues with stronger home advantage, shifting class priors.


## 4. Results: XG-boost



## Discussion

1. Här kan vi jämföra resultaten mellan XG-boost och random forest. Såg vi några tydliga skillnader.
2. Därefter kan vi diskutera vad som skulle vara intressant att utveckla. Spontant: inkorperera individuella spelare och deras statistik/performance. Då skulle man kunna skapa förväntade predictions på lag som inte spelat mot varandra någon gång eller påhittade lag/laguppställningar. Det skulle nog också kunna ge bättre resultat generellt. (OM m du kommer på något mer så kan du lägga till det tänker jag.) Men typ:

 **Data Enhancement**
   - Incorporation of individual player statistics and performance metrics
   - Addition of real-time match data (e.g., weather conditions, team formations)
   - Integration of more detailed historical head-to-head records
   - Inclusion of team-specific factors like injuries and suspensions


Eller dyl: 
The project demonstrates the potential of machine learning in sports prediction, while also highlighting the complexity of soccer match outcomes and the need for continuous model improvement and data enhancement. 