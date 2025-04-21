## 1. How to open the Notebook.

xxxxx
xxxxx

## 2. What have been found after running the EDA: Notebook 

### 2.1 dataUnderstanding section 

#### Summary of our data audit

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

#### The resulting clean, compact dataset contains

1. **Match metadata** (season, date, home/away team).  
2. **Bookmaker odds** (converted to implied probabilities).  
3. **Team attributes** for both teams (and preferably the difference between them).  
4. **Target variable:** the match outcome.

## 2.2 Data Preparation (Step 2)

1. **Column selection** from `Match`:  
   - Kept `date`, home/away team IDs, goals, and top bookmaker odds (B365, BWH, IWH, LBH, PSH, WHH, SJH, and their draw/away equivalents).

2. **Merged team names**:  
   - Joined `Team.team_long_name` to both home and away IDs.

3. **Built attribute tables**:  
   - Extracted `team_api_id`, `date`, and tactical metrics from `Team_Attributes`.  
   - Created two copies: one prefixed `home_`, one `away_`, each sorted chronologically.

4. **Joined attributes to matches**:  
   - Merged home attributes onto each match, keeping the latest pre‑match record via `groupby` + `last`.  
   - Repeated for away attributes.  
   - Resulting `df_final`: 19 355 matches × 71 columns (no missing tactical values).

5. **Derived target and diff features**:  
   - `goal_diff = home_team_goal - away_team_goal`.  
   - `result` as `1` (home win), `0` (draw), `-1` (away win).

### 2.3 Feature Understanding (Step 3)

- **Goal distributions**: Histograms for `home_team_goal` and `away_team_goal`.  
- **Numeric attribute histograms & densities**: Plotted for all non‑ID, non‑odds numeric attributes.  
- **Baseline win probability**: Home‑win rate ≈ 0.4566.

### 2.4 Feature Relationships (Step 4)

1. **Tactical diffs**: For each key metric (speed, dribbling, passing, crossing, shooting, pressure, aggression, team width), calculated `home_<feat> - away_<feat>`.  
2. **Cleaned diffs**: Replaced infinite with `NaN` and dropped any rows missing diff values.  
3. **Sampled** 100 matches for performance, encoded `result` as a categorical hue.  
4. **Pairplot**: Explored pairwise relationships between all tactical diffs, colored by `result`.





