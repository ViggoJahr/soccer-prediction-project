import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path


# Start timer
start_time = time.time()

# Connect to the database
db_path = Path(__file__).resolve().parents[1] / "data" / "database.sqlite" # works aslong as the database.sqlite is unzipped in the data-folder.
conn = sqlite3.connect(db_path)
print("Connected to database successfully!")

# Query match data
match_query = """
SELECT 
    m.id,
    m.date,
    m.season,
    m.league_id,
    m.stage,
    m.home_team_api_id,
    m.away_team_api_id,
    m.home_team_goal,
    m.away_team_goal,
    m.home_player_1, m.home_player_2, m.home_player_3, m.home_player_4, m.home_player_5,
    m.home_player_6, m.home_player_7, m.home_player_8, m.home_player_9, m.home_player_10, m.home_player_11,
    m.away_player_1, m.away_player_2, m.away_player_3, m.away_player_4, m.away_player_5,
    m.away_player_6, m.away_player_7, m.away_player_8, m.away_player_9, m.away_player_10, m.away_player_11,
    m.B365H, m.B365D, m.B365A,
    m.BWH, m.BWD, m.BWA,
    m.IWH, m.IWD, m.IWA,
    m.PSH, m.PSD, m.PSA,
    m.WHH, m.WHD, m.WHA,
    t1.team_long_name AS home_team,
    t2.team_long_name AS away_team
FROM Match m
JOIN Team t1 ON m.home_team_api_id = t1.team_api_id
JOIN Team t2 ON m.away_team_api_id = t2.team_api_id;
"""
matches = pd.read_sql_query(match_query, conn)
print(f"Loaded matches: {len(matches)}")

# Query team attributes
attr_query = """
SELECT 
    team_api_id,
    date,
    buildUpPlaySpeed,
    chanceCreationPassing,
    chanceCreationCrossing,
    chanceCreationShooting,
    defencePressure,
    defenceAggression,
    defenceTeamWidth
FROM Team_Attributes
"""
team_attrs = pd.read_sql_query(attr_query, conn)
print(f"Loaded team attributes: {len(team_attrs)}")

conn.close()

# Validate data quality
print(f"Unique teams in matches: {matches['home_team_api_id'].nunique()}")
print(f"Unique teams in Team_Attributes: {team_attrs['team_api_id'].nunique()}")
print(f"Missing odds for all bookmakers: {matches[['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA']].isna().sum().to_dict()}")
print(f"Team ID types: {matches[['home_team_api_id', 'away_team_api_id']].dtypes}")
print(f"Missing team IDs: {matches[['home_team_api_id', 'away_team_api_id']].isna().sum().to_dict()}")

# Convert team IDs to Int64
matches['home_team_api_id'] = matches['home_team_api_id'].astype('Int64')
matches['away_team_api_id'] = matches['away_team_api_id'].astype('Int64')

# Drop rows with missing team IDs
matches = matches.dropna(subset=['home_team_api_id', 'away_team_api_id'])
print(f"Matches after dropping missing team IDs: {len(matches)}")

# Convert dates
matches['date'] = pd.to_datetime(matches['date'])
team_attrs['date'] = pd.to_datetime(team_attrs['date'])

# Compute match outcome and goal difference
matches['result'] = np.select(
    [matches['home_team_goal'] > matches['away_team_goal'],
     matches['home_team_goal'] < matches['away_team_goal']],
    [1, -1],  # 1 = Home Win, -1 = Away Win, 0 = Draw
    default=0
)
matches['match_goal_diff'] = matches['home_team_goal'] - matches['away_team_goal']

# Compute player quality
player_cols = [f'home_player_{i}' for i in range(1, 12)] + [f'away_player_{i}' for i in range(1, 12)]
matches['home_player_quality'] = matches[[f'home_player_{i}' for i in range(1, 12)]].notna().sum(axis=1)
matches['away_player_quality'] = matches[[f'away_player_{i}' for i in range(1, 12)]].notna().sum(axis=1)

# Compute home/away bias
home_win_rates = matches.groupby('home_team_api_id')['result'].apply(lambda x: (x == 1).mean()).rename('home_win_rate')
away_win_rates = matches.groupby('away_team_api_id')['result'].apply(lambda x: (x == -1).mean()).rename('away_win_rate')
matches = matches.merge(home_win_rates, left_on='home_team_api_id', right_index=True, how='left')
matches = matches.merge(away_win_rates, left_on='away_team_api_id', right_index=True, how='left')

# Compute season trend
season_win_rates = matches.groupby(['home_team_api_id', 'season'])['result'].apply(lambda x: (x == 1).mean()).rename('season_win_rate')
matches = matches.merge(season_win_rates, left_on=['home_team_api_id', 'season'], right_index=True, how='left')

# Compute draw tendency
draw_rates = matches.groupby('home_team_api_id')['result'].apply(lambda x: (x == 0).mean()).rename('draw_rate_team')
matches = matches.merge(draw_rates, left_on='home_team_api_id', right_index=True, how='left')

# Compute league competitiveness
league_stats = matches.groupby('league_id')['result'].agg(
    win_rate=lambda x: (x != 0).mean(),
    draw_rate=lambda x: (x == 0).mean()
)
matches = matches.merge(league_stats, left_on='league_id', right_index=True, how='left')

# Compute high-stakes match
matches['high_stakes'] = (matches['stage'] > 30).astype(int)

# Impute missing Bet365 odds
print("Imputing missing odds...")
odds_time = time.time()
bookmakers = ['BW', 'IW', 'PS', 'WH']
for col, other_cols in [('B365H', [f'{bm}H' for bm in bookmakers]),
                        ('B365D', [f'{bm}D' for bm in bookmakers]),
                        ('B365A', [f'{bm}A' for bm in bookmakers])]:
    mask = matches[col].isna()
    matches.loc[mask, col] = matches.loc[mask, other_cols].mean(axis=1)

# Fallback to league-specific means
league_odds_means = matches.groupby('league_id')[['B365H', 'B365D', 'B365A']].mean()
matches = matches.merge(league_odds_means, left_on='league_id', right_index=True, how='left', suffixes=('', '_league_mean'))
matches['B365H'] = matches['B365H'].fillna(matches['B365H_league_mean'])
matches['B365D'] = matches['B365D'].fillna(matches['B365D_league_mean'])
matches['B365A'] = matches['B365A'].fillna(matches['B365A_league_mean'])

# Final fallback to global means
matches[['B365H', 'B365D', 'B365A']] = matches[['B365H', 'B365D', 'B365A']].fillna(matches[['B365H', 'B365D', 'B365A']].mean())

# Cap extreme odds
matches['B365H'] = matches['B365H'].clip(upper=30)
matches['B365D'] = matches['B365D'].clip(upper=15)
matches['B365A'] = matches['B365A'].clip(upper=30)

matches = matches.drop(columns=['B365H_league_mean', 'B365D_league_mean', 'B365A_league_mean'])

print(f"Missing odds after imputation: {matches[['B365H', 'B365D', 'B365A']].isna().sum().to_dict()}")
print(f"Imputed odds stats:\n{matches[['B365H', 'B365D', 'B365A']].describe()}")
matches[['B365H', 'B365D', 'B365A']].hist(bins=50)
print(f"Odds imputation time: {time.time() - odds_time:.2f} seconds")

# Compute implied probabilities
print("Computing implied probabilities...")
prob_time = time.time()
bookmakers = ['B365', 'BW', 'IW', 'PS', 'WH']
for bm in bookmakers:
    odds_cols = [f'{bm}H', f'{bm}D', f'{bm}A']
    probs = 1 / matches[odds_cols]
    probs_sum = probs.sum(axis=1)
    for col in odds_cols:
        matches[f'{col}_prob'] = probs[col] / probs_sum

matches['home_prob'] = matches[[f'{bm}H_prob' for bm in bookmakers]].mean(axis=1)
matches['draw_prob'] = matches[[f'{bm}D_prob' for bm in bookmakers]].mean(axis=1)
matches['away_prob'] = matches[[f'{bm}A_prob' for bm in bookmakers]].mean(axis=1)
matches['home_prob_median'] = matches[[f'{bm}H_prob' for bm in bookmakers]].median(axis=1)
matches['draw_prob_median'] = matches[[f'{bm}D_prob' for bm in bookmakers]].median(axis=1)
matches['away_prob_median'] = matches[[f'{bm}A_prob' for bm in bookmakers]].median(axis=1)
print(f"Probability computation time: {time.time() - prob_time:.2f} seconds")

# Optimize team attributes merge
print("Merging team attributes...")
attr_time = time.time()
tactical_cols = ['buildUpPlaySpeed', 'chanceCreationPassing', 'chanceCreationCrossing',
                 'chanceCreationShooting', 'defencePressure', 'defenceAggression', 'defenceTeamWidth']

# Impute missing teams' attributes
missing_teams = set(matches['home_team_api_id'].unique()) | set(matches['away_team_api_id'].unique()) - set(team_attrs['team_api_id'].unique())
print(f"Missing teams: {len(missing_teams)}")
for team in missing_teams:
    league = matches[matches['home_team_api_id'] == team]['league_id'].iloc[0]
    league_attrs_medians = team_attrs[team_attrs['team_api_id'].isin(
        matches[matches['league_id'] == league]['home_team_api_id'])][tactical_cols].median()
    team_attrs = pd.concat([team_attrs, pd.DataFrame({
        'team_api_id': [team],
        'date': [matches['date'].min()],
        **{col: [league_attrs_medians[col]] for col in tactical_cols}
    })])
print(f"Teams with attributes after imputation: {team_attrs['team_api_id'].nunique()}")

# Merge attributes
team_attrs = team_attrs.sort_values(['team_api_id', 'date'])
matches['match_key'] = matches['home_team_api_id'].astype(str) + '_' + matches['date'].astype(str)
team_attrs['attr_key'] = team_attrs['team_api_id'].astype(str) + '_' + team_attrs['date'].astype(str)
home_attrs = matches.merge(team_attrs, left_on=['home_team_api_id', 'date'], right_on=['team_api_id', 'date'], how='left')
away_attrs = matches.merge(team_attrs, left_on=['away_team_api_id', 'date'], right_on=['team_api_id', 'date'], how='left')

print(f"Unmatched home teams: {home_attrs['team_api_id'].isna().sum()}")
print(f"Unmatched away teams: {away_attrs['team_api_id'].isna().sum()}")

home_attrs = home_attrs[tactical_cols].add_prefix('home_')
away_attrs = away_attrs[tactical_cols].add_prefix('away_')
matches = pd.concat([matches, home_attrs, away_attrs], axis=1)

# Compute tactical differences
for col in tactical_cols:
    matches[f'diff_{col}'] = matches[f'home_{col}'] - matches[f'away_{col}']

# Impute missing attributes
print(f"Missing values before imputation:\n{matches[[f'home_{col}' for col in tactical_cols] + [f'away_{col}' for col in tactical_cols]].isna().sum()}")
float_cols = matches.select_dtypes(include=['float64']).columns
matches[float_cols] = matches[float_cols].fillna(matches[float_cols].mean())
print(f"Missing values after imputation:\n{matches[[f'home_{col}' for col in tactical_cols] + [f'away_{col}' for col in tactical_cols]].isna().sum()}")
matches = matches.drop(columns=['match_key'])
print(f"Team attributes merge time: {time.time() - attr_time:.2f} seconds")

# Compute Elo ratings
print("Computing Elo ratings...")
elo_time = time.time()
elo_ratings = {team: 1500.0 for team in matches['home_team_api_id'].unique()}
K = 20  # Elo adjustment factor

for idx, row in matches.iterrows():
    home_id = row['home_team_api_id']
    away_id = row['away_team_api_id']
    home_elo = elo_ratings[home_id]
    away_elo = elo_ratings[away_id]

    # Expected outcome (home advantage: +100)
    exp_home = 1 / (1 + 10 ** ((away_elo - (home_elo + 100)) / 400))
    exp_away = 1 - exp_home

    # Actual outcome
    if row['result'] == 1:  # Home win
        home_score, away_score = 1, 0
    elif row['result'] == -1:  # Away win
        home_score, away_score = 0, 1
    else:  # Draw
        home_score, away_score = 0.5, 0.5

    # Update Elo ratings
    elo_ratings[home_id] += K * (home_score - exp_home)
    elo_ratings[away_id] += K * (away_score - exp_away)

    # Assign to DataFrame
    matches.loc[idx, 'home_elo'] = elo_ratings[home_id]
    matches.loc[idx, 'away_elo'] = elo_ratings[away_id]
    matches.loc[idx, 'elo_diff'] = home_elo - away_elo
print(f"Elo computation time: {time.time() - elo_time:.2f} seconds")

# Compute head-to-head
print("Computing head-to-head...")
h2h_time = time.time()

# Initialize columns using pd.concat to avoid fragmentation
new_cols = pd.DataFrame(index=matches.index)
new_cols['h2h_home_wins'] = 0.0
new_cols['h2h_goal_diff'] = 0.0
new_cols['h2h_home_win_rate'] = 0.0
new_cols['h2h_home_goals'] = 0.0
new_cols['h2h_away_goals'] = 0.0
new_cols['h2h_home_goal_efficiency'] = 0.0
new_cols['h2h_draw_rate'] = 0.0
matches = pd.concat([matches, new_cols], axis=1)

matches['team_pair'] = matches.apply(lambda row: tuple(sorted([row['home_team_api_id'], row['away_team_api_id']])), axis=1)
matches = matches.sort_values('date')

def compute_h2h(group):
    group = group.sort_values('date')
    h2h_wins = []
    h2h_goal_diff = []
    h2h_win_rate = []
    h2h_home_goals = []
    h2h_away_goals = []
    h2h_home_goal_efficiency = []
    h2h_draw_rate = []
    for i in range(len(group)):
        past = group.iloc[:i].tail(5)
        if not past.empty:
            weights = np.exp(-0.5 * (group.iloc[i]['date'] - past['date']).dt.days / 365)
            home_team = group.iloc[i]['home_team_api_id']
            wins = sum((past['result'] * (past['home_team_api_id'] == home_team) + 
                        (-past['result'] * (past['away_team_api_id'] == home_team))) * weights)
            goal_diff = sum((past['home_team_goal'] - past['away_team_goal']) * 
                            (past['home_team_api_id'] == home_team) * weights +
                            (past['away_team_goal'] - past['home_team_goal']) * 
                            (past['away_team_api_id'] == home_team) * weights)
            home_goals = sum(past['home_team_goal'] * (past['home_team_api_id'] == home_team) * weights +
                             past['away_team_goal'] * (past['away_team_api_id'] == home_team) * weights)
            away_goals = sum(past['away_team_goal'] * (past['home_team_api_id'] == home_team) * weights +
                             past['home_team_goal'] * (past['away_team_api_id'] == home_team) * weights)
            draws = sum((past['result'] == 0) * weights)
            total = sum(weights * (past['result'].abs() <= 1))
            win_rate = wins / (total + 1e-10) if total > 0 else 0
            goal_efficiency = (home_goals + 1) / (away_goals + 1)  # Smoothed to avoid division by zero
            draw_rate = draws / (total + 1e-10) if total > 0 else 0
            h2h_wins.append(wins)
            h2h_goal_diff.append(goal_diff)
            h2h_win_rate.append(win_rate)
            h2h_home_goals.append(home_goals / (total + 1e-10))
            h2h_away_goals.append(away_goals / (total + 1e-10))
            h2h_home_goal_efficiency.append(goal_efficiency)
            h2h_draw_rate.append(draw_rate)
        else:
            h2h_wins.append(0)
            h2h_goal_diff.append(0)
            h2h_win_rate.append(0)
            h2h_home_goals.append(0)
            h2h_away_goals.append(0)
            h2h_home_goal_efficiency.append(0)
            h2h_draw_rate.append(0)
    return pd.DataFrame({
        'h2h_home_wins': h2h_wins,
        'h2h_goal_diff': h2h_goal_diff,
        'h2h_home_win_rate': h2h_win_rate,
        'h2h_home_goals': h2h_home_goals,
        'h2h_away_goals': h2h_away_goals,
        'h2h_home_goal_efficiency': h2h_home_goal_efficiency,
        'h2h_draw_rate': h2h_draw_rate
    }, index=group.index)

h2h_results = matches.groupby('team_pair', group_keys=False).apply(compute_h2h, include_groups=False)
matches[['h2h_home_wins', 'h2h_goal_diff', 'h2h_home_win_rate', 'h2h_home_goals', 'h2h_away_goals', 'h2h_home_goal_efficiency', 'h2h_draw_rate']] = h2h_results
print(f"Head-to-head computation time: {time.time() - h2h_time:.2f} seconds")

# Compute recent form using deque
print("Computing recent form...")
form_time = time.time()

# Initialize form feature columns
matches['home_form_points'] = 0.0
matches['home_form_goals'] = 0.0
matches['home_form_intensity'] = 0.0
matches['home_form_draw'] = 0.0
matches['away_form_points'] = 0.0
matches['away_form_goals'] = 0.0
matches['away_form_intensity'] = 0.0
matches['away_form_draw'] = 0.0

# Initialize deques for each team
team_points = {team: deque(maxlen=5) for team in matches['home_team_api_id'].unique()}
team_goals = {team: deque(maxlen=5) for team in matches['home_team_api_id'].unique()}
team_intensity = {team: deque(maxlen=5) for team in matches['home_team_api_id'].unique()}
team_draw = {team: deque(maxlen=5) for team in matches['home_team_api_id'].unique()}

# Sort matches by date
matches = matches.sort_values('date').reset_index(drop=True)

# Compute form features iteratively
for idx, row in matches.iterrows():
    home_id = row['home_team_api_id']
    away_id = row['away_team_api_id']

    # Assign form features based on past matches
    matches.loc[idx, 'home_form_points'] = np.mean(team_points[home_id]) if team_points[home_id] else 0
    matches.loc[idx, 'home_form_goals'] = np.mean(team_goals[home_id]) if team_goals[home_id] else 0
    matches.loc[idx, 'home_form_intensity'] = np.mean(team_intensity[home_id]) if team_intensity[home_id] else 0
    matches.loc[idx, 'home_form_draw'] = np.mean(team_draw[home_id]) if team_draw[home_id] else 0

    matches.loc[idx, 'away_form_points'] = np.mean(team_points[away_id]) if team_points[away_id] else 0
    matches.loc[idx, 'away_form_goals'] = np.mean(team_goals[away_id]) if team_goals[away_id] else 0
    matches.loc[idx, 'away_form_intensity'] = np.mean(team_intensity[away_id]) if team_intensity[away_id] else 0
    matches.loc[idx, 'away_form_draw'] = np.mean(team_draw[away_id]) if team_draw[away_id] else 0

    # Update deques with current match outcomes
    home_points = 3 if row['result'] == 1 else (1 if row['result'] == 0 else 0)
    away_points = 3 if row['result'] == -1 else (1 if row['result'] == 0 else 0)
    total_goals = row['home_team_goal'] + row['away_team_goal']
    is_draw = 1 if row['result'] == 0 else 0

    team_points[home_id].append(home_points)
    team_goals[home_id].append(row['home_team_goal'])
    team_intensity[home_id].append(total_goals)
    team_draw[home_id].append(is_draw)

    team_points[away_id].append(away_points)
    team_goals[away_id].append(row['away_team_goal'])
    team_intensity[away_id].append(total_goals)
    team_draw[away_id].append(is_draw)

print(f"Recent form computation time: {time.time() - form_time:.2f} seconds")

# Validate final dataset
print(f"Missing values in final dataset:\n{matches.isna().sum()}")

# Clean dataset
final_cols = ['id', 'date', 'season', 'league_id', 'stage', 'home_team_api_id', 'away_team_api_id', 'result',
              'home_prob', 'draw_prob', 'away_prob', 'home_prob_median', 'draw_prob_median', 'away_prob_median',
              'home_elo', 'away_elo', 'elo_diff',
              'home_form_points', 'away_form_points', 'home_form_goals', 'away_form_goals',
              'home_form_intensity', 'away_form_intensity', 'home_form_draw', 'away_form_draw',
              'home_player_quality', 'away_player_quality', 'home_win_rate', 'away_win_rate', 'season_win_rate',
              'draw_rate_team', 'win_rate', 'draw_rate', 'high_stakes',
              'h2h_home_wins', 'h2h_goal_diff', 'h2h_home_win_rate', 'h2h_home_goals', 'h2h_away_goals',
              'h2h_home_goal_efficiency', 'h2h_draw_rate',
              'home_chanceCreationShooting', 'away_chanceCreationShooting', 'diff_chanceCreationShooting'] + \
             [f'diff_{col}' for col in ['buildUpPlaySpeed', 'chanceCreationPassing', 'chanceCreationCrossing',
                                        'defencePressure', 'defenceAggression', 'defenceTeamWidth']]
df_final = matches[final_cols]

# Deduplicate before saving
matches = matches.drop_duplicates(subset='id')

# Save to CSV
csv_path = db_path.parent / "football_ml_dataset.csv"
df_final.to_csv(csv_path, index=False)
print(f"\nDataset saved to football_ml_dataset.csv with {len(df_final)} matches")
print("\nSample data (first 5 rows):")
print(df_final.head())
print(f"\nFeature distributions:\n{df_final[['home_prob', 'elo_diff', 'home_form_points', 'h2h_home_goal_efficiency']].describe()}")
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")