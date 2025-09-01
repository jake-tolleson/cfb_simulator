import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import sqlite3

# connect to the SQLite database
conn = sqlite3.connect("cfb_pbp_db")

# create a cursor
c = conn.cursor()

df = pd.read_sql_query(
    """
SELECT 
    week,
    season,
    game_id,
    drive_id,
    id_play as play_id,
    game_play_number,
    drive_play_number,
    home,
    away,
    pos_team,
    def_pos_team,
    play_type,
    play_text,
    rush,
    yards_gained,
    yards_to_goal,
    log_ydstogo,
    log_ydstogo * Goal_To_Go as log_ydstogo_g2g_interaction,
    log_ydstogo * down as log_ydstogo_down_interaction,
    yards_to_goal * down as yards_to_goal_down_interaction,
    EPA,
    ppa,
    down,
    distance,
    period,
    TimeSecsRem as time_left_in_half,
    adj_TimeSecsRem as time_left_in_game,
    min(adj_TimeSecsRem, 15*60) as time_left_in_game_capped,
    pos_team_score, 
    def_pos_team_score,
    score_diff,
    rz_play,
    scoring_opp,
    middle_8,
    Goal_To_Go,
    Under_two,
    case when down in (3, 4) then 1 else 0 end as is_critical_down,
    case when distance <= 2 then 1 else 0 end as is_short_distance,
    case when yards_to_goal <= 20 then 1 else 0 end as is_red_zone,
    case when pos_team_score - def_pos_team_score < 0 then 1 else 0 end as is_trailing,
    case when pos_team_score - def_pos_team_score = 0 then 1 else 0 end as is_tied,
    case when pos_team_score - def_pos_team_score > 0 then 1 else 0 end as is_leading,
    case when adj_TimeSecsRem <= 5 * 60 then 1 else 0 end as is_late_game,
    case when pos_team = home then 1 else 0 end as is_home_offense,
    case when pos_team = away then 1 else 0 end as is_away_offense,
    case when yards_to_goal > 50 then 1 else 0 end as is_own_territory,
    case when yards_to_goal < 50 then 1 else 0 end as is_opponent_territory,
    down * distance as down_distance,
    score_diff * adj_TimeSecsRem as score_time_interaction
FROM 
    cfbfastR_pbp 
WHERE 
    year = 2024
    ppa is not null
ORDER BY
    game_id, drive_id, game_play_number
""",
    conn,
)

conn.close()

# Define the split point
split_week = 10

# Create the training and test sets based on the week number
train_df = df[(df["week"] < split_week)]
test_df = df[(df["week"] >= split_week)]

features = [
    "yards_to_goal",
    "yards_gained",
    "down_distance",
    "down",
    "Goal_To_Go",
    "log_ydstogo",
    "log_ydstogo_g2g_interaction",
    "log_ydstogo_down_interaction",
    "yards_to_goal_down_interaction",
    "time_left_in_half",
    "Under_two",
]
# Separate features (X) and target (y) for both sets
X_train = train_df[features]
y_train = train_df["ppa"]
X_test = test_df[features]
y_test = test_df["ppa"]

print(f"Training on {len(X_train)} plays (Weeks 1-{split_week - 1})")
print(f"Testing on {len(X_test)} plays (Weeks {split_week} and later)")

# --- Model Training and Evaluation ---

# Use your best model (Calibrated LightGBM)
# You can use the best params you found earlier or just use solid defaults
lgbm = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7, random_state=42
)
lgbm.fit(X_train, y_train)

# save model
joblib.dump(lgbm, "ppa_model.pkl")

# predict
y_pred = lgbm.predict(X_test)

# Evaluate
print("R^2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
df["EPA"].describe()

# Hyper param tuning
param_grid = {
    "num_leaves": [15, 31, 63],
    "max_depth": [3, 5, 7, -1],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 500],
}

lgbm = lgb.LGBMRegressor(random_state=42)

grid_search = GridSearchCV(
    lgbm,
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Use best estimator for calibration and evaluation
best_lgbm = grid_search.best_estimator_

# predict
y_pred = best_lgbm.predict(X_test)

# Evaluate
print("R^2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
