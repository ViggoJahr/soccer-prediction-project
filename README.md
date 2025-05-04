# Soccer Match Outcome Prediction

## 1. Overview
This project aims to predict the outcome of soccer matches (Home Win, Draw, Away Win) using historical match data, betting odds, and team/player attributes. We will explore data cleaning, feature engineering, and train a baseline model (e.g., Random Forest or Logistic Regression) to assess predictive performance.

## 2. Current Status
- We have successfully cleaned the data and are in the process of feature engineering. A preliminary model has been built, and we are currently evaluating its performance.
- The dataset has been consolidated from multiple sources, including match results, betting odds, and team attributes. 
## 3. Project Goals & Roadmap
- **Short-Term (Week 1–2):** ✅
  - Consolidate data sources and handle basic cleaning.  
  - Conduct initial exploratory data analysis (EDA).  
  - Outline possible features (form, head-to-head performance, odds, etc.).

- **Mid-Term (Week 2–3):**  *Work in progress*
  - Implement feature engineering.  
  - Build an initial baseline model.  
  - Experiment with improved models (e.g., Random Forest, XGBoost).

- **Long-Term (Week 4):**  
  - Finalize model tuning and evaluation.  
  - Prepare a demo and finalize documentation.


## 4. Repository Structure
- `README.md` – Project overview and instructions.  
- `data/` – Contains raw and cleaned datasets, along with visualizations.  
- `src/` – Core source code and scripts.  
- `docs/` – Project-related documentation.  

## 5. Quick-Start Installation Guide (Linux • macOS • Windows)

> **Goal:** run the whole project—from cloning to model results—on your local machine.
> **Note:** This guide assumes you have basic knowledge of Python and Git. If you are new to these tools, please refer to their respective documentation for installation instructions.
---

### 1. Install Python 3.9+ Install core libraries.

| Platform  | Recommended method |
|-----------|--------------------|
| **Windows** | Download the installer from [python.org](https://www.python.org/downloads/windows/) |
| **macOS**   | `brew install python` (requires [Homebrew](https://brew.sh)). |
| **Linux**   | `sudo apt-get install python3 python3-pip` (Debian/Ubuntu) or grab the tarball from [python.org](https://www.python.org). |
---

For all platforms, run the following command to install the required libraries:

```bash
# Install core libraries
pip install pandas numpy scikit-learn matplotlib
```


### 2. (Optional) Create & activate a virtual environment
You can find instructions on your own.

### 3. Clone the repository
Find the location you want the repository in, and in the terminal. Run: 

```bash
git clone https://github.com/ViggoJahr/soccer-prediction-project.git
cd soccer-match-outcome-prediction   # makes sure you are in the right location.
```

### 4. Unzip the data 
Unzip the [`archivedData.zip`](/data/archivedData.zip)
 into the project’s [`data\`](/data)
 folder & make sure that it stays in the [`data\`](/data) folder and that the name of the database is `database.sqlite`.

### 5. Run the pipeline

```bash
# Prepare cleaned / feature-ready data
python src/prepare_football_data.py

# Train & evaluate the model
python src/optimized_random_forrest.py
# or with the advanced model
python src/boosted_random_forrest.py
```

### 6. Locate your results

- **Visualizations:** `data/visualizations/` — contains all plots and charts generated during training and evaluation.  
- **Predictions:** `data/predictions.csv` — a CSV file with the model’s predicted outcomes for each match.  
- **Other artifacts:** cleaned datasets and any saved model files live in the root `data/` folder, as defined in the scripts.



## 6. Libraries & Frameworks
- **Python 3.9+**
- **NumPy** and **pandas** for data manipulation.
- **Matplotlib** for data visualization.
- **scikit-learn** for model building. (annat?)

*(Additional libraries will be added as needed.)*

## 7. Contributing / Team Members
- **Team Members:**
  - Viggo Jahr [@ViggoJahr](https://github.com/ViggoJahr)
  - Axel Prander [@amjgp](https://github.com/amjgp)

## 8. Project Management
- We use the GitHub Issue Tracker to manage tasks, bugs, and new features.
- Milestones are established for each week of the project (Weeks 1-4).
- Each new feature/task is recorded as an Issue and referenced in commit messages.
