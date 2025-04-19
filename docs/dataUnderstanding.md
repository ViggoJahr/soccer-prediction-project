## 1. How to open the Notebook.

xxxxx
xxxxx

## 2. What have been found after running the EDA: Notebook 

### 2.1 dataUnderstanding section 

Summary of our data audit

**The core resides in the `Match` table**  

Here we find everything that is actually known **before kick‑off**: match ID, date, season/stage, home team, away team, goals, and the bookmaker odds. These fields form both the target variable (the result) and several strong predictors (odds → implied probabilities).

---

**`Team` and `Team_Attributes` are useful**

* `Team` only translates IDs into names—handy for interpreting models and visualising results.  
* `Team_Attributes` contains playing‑style metrics (build‑up speed, crossing, defence, etc.) that can be linked to each club via `team_api_id`. We pick the record that lies *closest before* the match date to avoid data leakage.

---

**We skip the player tables (`Player`, `Player_Attributes`)**  
The starting‑XI columns in `Match` are largely `NaN`, so we cannot reliably identify which players actually appeared. Imputing them would be uncertain and risk injecting noisy signals.

---

**We ignore league details and in‑game event strings**

* League information beyond a simple `league_id` adds no direct predictive power.  
* XML strings for shots, fouls, cards, possession, etc., describe events **after** kick‑off and therefore cannot be used as pre‑match features.

---

### The resulting clean, compact dataset contains

1. **Match metadata** (season, date, home/away team).  
2. **Bookmaker odds** (converted to implied probabilities).  
3. **Team attributes** for both teams (and preferably the difference between them).  
4. **Target variable:** the match outcome.








