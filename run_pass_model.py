import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import matplotlib.pyplot as plt
import cfbd
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
    pbp.season,
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
    case when down = 1 then 1 else 0 end as first_down_play,
    case when down = 1 then 1 else 0 end * rush as first_down_run,
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
    score_diff * adj_TimeSecsRem as score_time_interaction,
    ott.talent as offensive_talent,
    dtt.talent as defensive_talent,
    pr.usage / pbp.week as ret_usage,
    pr.passing_usage / pbp.week as ret_passing_usage,
    pr.rushing_usage / pbp.week as ret_rushing_usage,
    pr.receiving_usage / pbp.week as ret_receiving_usage
FROM 
    cfbfastR_pbp pbp
LEFT JOIN
    team_talent ott 
    ON pbp.pos_team = ott.team 
    AND pbp.season = ott.year
LEFT JOIN
    team_talent dtt 
    ON pbp.def_pos_team = dtt.team 
    AND pbp.season = dtt.year
LEFT JOIN
    player_returning pr
    ON pbp.pos_team = pr.team
    AND pbp.season = pr.season
WHERE 
    pbp.year = 2024
    and (rush = 1 or pass = 1)
ORDER BY
    pbp.season, pbp.week, drive_id, game_play_number
""",
    conn,
)

conn.close()


# Ensure you have your data from multiple seasons loaded into 'df'

# 1. CRITICAL: Sort the data chronologically
df.sort_values(by=["season", "week", "drive_id", "game_play_number"], inplace=True)

# Calculate number of runs and total plays per offense per game
game_stats = (
    df.groupby(["pos_team", "game_id"])
    .agg(
        runs=("rush", "sum"),
        plays=("rush", "count"),
        first_downs=("first_down_play", "sum"),
        first_down_runs=("first_down_run", "sum"),
    )
    .reset_index()
    .sort_values(["pos_team", "game_id"])
)

# Calculate cumulative runs and plays up to (but not including) each game
game_stats["cum_runs"] = (
    game_stats.groupby("pos_team")["runs"].cumsum().shift(1).fillna(0)
)
game_stats["cum_plays"] = (
    game_stats.groupby("pos_team")["plays"].cumsum().shift(1).fillna(0)
)
game_stats["cum_first_downs"] = (
    game_stats.groupby("pos_team")["first_downs"].cumsum().shift(1).fillna(0)
)
game_stats["cum_first_down_runs"] = (
    game_stats.groupby("pos_team")["first_down_runs"].cumsum().shift(1).fillna(0)
)

# Calculate prior run rate (weighted)
game_stats["offense_historical_run_rate"] = game_stats["cum_runs"] / game_stats[
    "cum_plays"
].replace(0, np.nan)
game_stats["offense_hist_run_rate_1st_down"] = game_stats[
    "cum_first_down_runs"
] / game_stats["cum_first_downs"].replace(0, np.nan)


# Merge prior_run_rate back to main df
df = df.merge(
    game_stats[
        [
            "pos_team",
            "game_id",
            "offense_historical_run_rate",
            "offense_hist_run_rate_1st_down",
        ]
    ],
    on=["pos_team", "game_id"],
    how="left",
)

league_avg_run_rate = 0.5
# Fill cold start with league average
df["offense_historical_run_rate"] = df["offense_historical_run_rate"].fillna(
    league_avg_run_rate
)
df["offense_hist_run_rate_1st_down"] = df["offense_hist_run_rate_1st_down"].fillna(
    league_avg_run_rate
)

# New feature: Previous play was a run
df["previous_run"] = (
    df.sort_values("game_play_number").groupby("drive_id")["rush"].shift(1).fillna(0)
)

# df["time_left_bin"] = pd.cut(
#     df["time_left_in_game"],
#     bins=[0, 120, 5 * 60, 15 * 60, 30 * 60, 60 * 60],
#     labels=["2min", "late", "end_quarter", "mid_game", "early"],
#     include_lowest=True,
# )
# df = pd.get_dummies(df, columns=["time_left_bin"], drop_first=True)

# # Offense and defense one-hot encoding
# df = pd.get_dummies(
#     df,
#     columns=["offense", "defense"],
#     drop_first=True,
# )

ppa_model = joblib.load("ppa_model.pkl")
ppa_features = [
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
df["pred_ppa"] = ppa_model.predict(df[ppa_features])


# get the ppa of previous play if the same drive, order by playNumber
df["previous_ppa"] = (
    df.sort_values("game_play_number")
    .groupby("drive_id")["pred_ppa"]
    .shift(1)
    .fillna(0)
)
df["previous_ppa_2"] = (
    df.sort_values("game_play_number")
    .groupby("drive_id")["pred_ppa"]
    .shift(2)
    .fillna(0)
)
# New feature: Change in PPA from previous play
df["ppa_change"] = df["previous_ppa_2"] - df["previous_ppa"]

# Define the split point
split_week = 10

# Create the training and test sets based on the week number
train_df = df[df["week"] < split_week]
test_df = df[df["week"] >= split_week]

features = [
    "score_diff",
    "yards_to_goal",
    "down_distance",
    "ppa_change",
    "previous_ppa",
    "previous_run",
    "is_home_offense",
    "is_away_offense",
    "offensive_talent",
    "defensive_talent",
    "ret_usage",
    "ret_passing_usage",
    "ret_rushing_usage",
    "ret_receiving_usage",
    "time_left_in_half",
    "offense_historical_run_rate",
    "offense_hist_run_rate_1st_down",
]
# Separate features (X) and target (y) for both sets
X_train = train_df[features]
y_train = train_df["rush"]
X_test = test_df[features]
y_test = test_df["rush"]

print(f"Training on {len(X_train)} plays (Weeks 1-{split_week - 1})")
print(f"Testing on {len(X_test)} plays (Weeks {split_week} and later)")

# --- Model Training and Evaluation ---

# Use your best model (Calibrated LightGBM)
# You can use the best params you found earlier or just use solid defaults
lgbm = lgb.LGBMClassifier(
    learning_rate=0.1, max_depth=-1, n_estimators=500, num_leaves=63, random_state=42
)
lgbm.fit(X_train, y_train)

calibrated_lgbm = CalibratedClassifierCV(lgbm, method="isotonic", cv=5)
calibrated_lgbm.fit(X_train, y_train)

# --- Evaluation on the Test Set ---
y_proba = calibrated_lgbm.predict_proba(X_test)[:, 1]
y_pred = calibrated_lgbm.predict(X_test)

auc_score = roc_auc_score(y_test, y_proba)

print("\n--- Results on Holdout Test Set (Weeks 9+) ---")
print(classification_report(y_test, y_pred))
print(f"ROC AUC on test set: {auc_score:.4f}")

# Save the model
joblib.dump(calibrated_lgbm, "run_pass_model.pkl")


df.loc[(df["offense"] == "Tennessee") & (df["week"] == 7)]
df.columns


# After fitting your LightGBM model (lgbm)
feature_importances = lgbm.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for easy viewing
fi_df = pd.DataFrame(
    {"feature": feature_names, "importance": feature_importances}
).sort_values(by="importance", ascending=False)

print("\nTop 20 LightGBM Feature Importances:")
print(fi_df.head(20))

# Optional: Plot feature importances
plt.figure(figsize=(10, 6))
fi_df.head(20).plot.bar(x="feature", y="importance", legend=False)
plt.title("Top 20 LightGBM Feature Importances")
plt.tight_layout()
plt.show()

# Hyper param tuning
param_grid = {
    "num_leaves": [15, 31, 63],
    "max_depth": [3, 5, 7, -1],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 500],
}

lgbm = lgb.LGBMClassifier(random_state=42)
grid_search = GridSearchCV(
    lgbm,
    param_grid,
    cv=3,
    scoring="roc_auc",
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best ROC AUC:", grid_search.best_score_)

# Use best estimator for calibration and evaluation
best_lgbm = grid_search.best_estimator_
calibrated_lgbm = CalibratedClassifierCV(best_lgbm, method="isotonic", cv=5)
calibrated_lgbm.fit(X_train, y_train)

y_pred = calibrated_lgbm.predict(X_test)
y_proba = calibrated_lgbm.predict_proba(X_test)[:, 1]

print("Tuned LightGBM Results:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
