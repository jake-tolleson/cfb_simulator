import pandas as pd
import lightgbm as lgb
from scipy.stats import percentileofscore
import numpy as np
import ast
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib
import sqlite3

# connect to the SQLite database
conn = sqlite3.connect("cfb_pbp_db")

# create a cursor
c = conn.cursor()

"conferences"                "draft_picks"               
"player_recruiting_rankings" "player_returning"           "player_usage"              
"team_recruiting_rankings"   "team_talent"                "transfer_portal"           
 "venues"                    

pd.Series(pd.read_sql_query("select * from conferences limit 10", conn).columns.tolist()).to_clipboard(index=False)

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
    def_pos_team as defense,
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
    case when down >= 3 and distance >= 7 then 1 else 0 end as obvious_pass_situation,
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
    and pass_attempt = 1
    and position_reception is not null
ORDER BY
    pbp.season, pbp.week, drive_id, game_play_number
""",
    conn,
)

conn.close()

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

# get the ppa of previous play if the same drive, order by game_play_number
df["previous_ppa"] = (
    df.sort_values("game_play_number")
    .groupby("drive_id")["pred_ppa"]
    .shift(1)
    .fillna(0)
)
df["previous_ppa_2"] = (
    df.sort_values("game_play_number").groupby("drive_id")["ppa"].shift(2).fillna(0)
)
# New feature: Change in PPA from previous play
df["ppa_change"] = df["previous_ppa_2"] - df["previous_ppa"]


pass_df = df.copy()

# Offensive passing average
off_avg_pass_yds = (
    pass_df.groupby("pos_team")["yards_gained"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)
pass_df["temp_off_avg_yds"] = off_avg_pass_yds
pass_df["rolling_off_avg_pass_yards"] = pass_df.groupby("pos_team")[
    "temp_off_avg_yds"
].shift(1)

# Defensive passing average allowed
def_avg_pass_yds_allowed = (
    pass_df.groupby("defense")["yards_gained"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)
pass_df["temp_def_avg_yds"] = def_avg_pass_yds_allowed
pass_df["rolling_def_avg_pass_yards_allowed"] = pass_df.groupby("defense")[
    "temp_def_avg_yds"
].shift(1)

# --- Step 2: Extract the pre-game value for each game ---

# For each game and each offense, the first play's rolling average is the pre-game average
pre_game_off_stats = (
    pass_df.groupby(["game_id", "pos_team"])["rolling_off_avg_pass_yards"]
    .first()
    .reset_index()
)
pre_game_off_stats.rename(
    columns={"rolling_off_avg_pass_yards": "pre_game_off_avg_pass_yards"}, inplace=True
)

pre_game_def_stats = (
    pass_df.groupby(["game_id", "defense"])["rolling_def_avg_pass_yards_allowed"]
    .first()
    .reset_index()
)
pre_game_def_stats.rename(
    columns={
        "rolling_def_avg_pass_yards_allowed": "pre_game_def_avg_pass_yards_allowed"
    },
    inplace=True,
)

# --- Step 3: Merge these static, pre-game stats back into the main DataFrame ---
df = pd.merge(df, pre_game_off_stats, on=["game_id", "pos_team"], how="left")
df = pd.merge(df, pre_game_def_stats, on=["game_id", "defense"], how="left")

# Handle the first game of the season for any team (will be NaN)
# Forward-fill can propagate a team's end-of-last-season value, but filling with a neutral
# average is safer and simpler.
league_avg_pass_yds = df["yards_gained"].mean()
df["pre_game_off_avg_pass_yards"].fillna(league_avg_pass_yds, inplace=True)
df["pre_game_def_avg_pass_yards_allowed"].fillna(league_avg_pass_yds, inplace=True)

# --- Step 4: Create the matchup feature ---
df["pre_game_pass_matchup"] = (
    df["pre_game_off_avg_pass_yards"] - df["pre_game_def_avg_pass_yards_allowed"]
)

df.columns
# --- Step 1: Train the Model ---

features = [
    "down",
    "distance",
    "yards_to_goal",
    "score_diff",
    "is_critical_down",
    "is_short_distance",
    "is_red_zone",
    "is_late_game",
    "is_home_offense",
    "is_away_offense",
    "is_own_territory",
    "is_opponent_territory",
    "pre_game_off_avg_pass_yards",
    "pre_game_def_avg_pass_yards_allowed",
    "pre_game_pass_matchup",
    "obvious_pass_situation",
    "previous_ppa",
    "ppa_change",
    "ret_usage",
    "ret_passing_usage",
    "ret_rushing_usage",
    "ret_receiving_usage",
    "time_left_in_half",
]

# Define the split point
split_week = 10
UPPER_BOUND = 60
LOWER_BOUND = -10


# Create the training and test sets based on the week number
train_df = df[(df["week"] < split_week)].dropna(subset=features + ["yards_gained"])

# Clip the y_train Series. This ONLY affects model training.
train_df["yards_gained_clipped"] = train_df["yards_gained"].clip(
    LOWER_BOUND, UPPER_BOUND
)

test_df = df[(df["week"] >= split_week)].dropna(subset=features + ["yards_gained"])
test_df["yards_gained_clipped"] = test_df["yards_gained"].clip(LOWER_BOUND, UPPER_BOUND)

# Separate features (X) and target (y) for both sets
X_train = train_df[features]
y_train = train_df["yards_gained_clipped"]
X_test = test_df[features]
y_test = test_df["yards_gained_clipped"]


# Train the LightGBM Regressor
lgbm_yards = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=7, random_state=42
)
print("Training yards gained model...")
lgbm_yards.fit(X_train, y_train)


# --- Step 3: Set up Quantile Mapping (with an extra step) ---
# Get predictions (these will be on the log scale)
all_predictions = lgbm_yards.predict(X_train)

# 2. Get the ACTUAL yards gained for all historical plays
actual_yards = test_df["yards_gained"].values

print("Distributions created. Ready for simulation.")

# Save the model and distributions
joblib.dump(lgbm_yards, "yds_gained_pass_model.pkl")
np.save("pass_all_predictions.npy", all_predictions)
np.save("pass_actual_yards.npy", actual_yards)


# This function is what you'll call inside your Monte Carlo loop
def simulate_yards_gained(play_features_df):
    """
    Takes a dataframe with a single row of play features,
    predicts the expected yards, and maps it to a realistic
    outcome using quantile mapping.
    """
    # Get the model's raw prediction (the "expected" yards)
    predicted_expected_yards = lgbm_yards.predict(play_features_df)[0]

    # Find the percentile of this prediction relative to all historical predictions
    pred_percentile = percentileofscore(all_predictions, predicted_expected_yards)

    # Find the yardage value at that same percentile in the REAL distribution
    simulated_outcome = np.percentile(actual_yards, pred_percentile)

    return round(simulated_outcome)


# --- USAGE EXAMPLE ---
row_number = 3
# Imagine in your simulation, it's a 1st & 10 pass play
sample_play = X_test.iloc[[row_number]]

simulated_yards = simulate_yards_gained(sample_play)
print(f"\nSimulated outcome for the play: {simulated_yards} yards")
y_test.iloc[row_number]

X_test.iloc[row_number]


# First, get the raw predictions on your test set
y_pred_raw = lgbm_yards.predict(X_test)

# 1. Use Standard Regression Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred_raw))
r2 = r2_score(y_test, y_pred_raw)

print("\n--- Core LGBM Regressor Evaluation ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# 2. Visualize: Actual vs. Predicted Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_raw, alpha=0.3)
plt.plot(
    [min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--"
)
plt.title("Actual vs. Predicted Yards")
plt.xlabel("Actual Yards Gained")
plt.ylabel("Predicted Expected Yards")
plt.grid(True)
plt.show()


# --- EVALUATION FOR THE SIMULATION OUTPUT ---
def vectorized_simulate_yards(X_data, model, all_preds_dist, actual_yards_dist):
    """
    Vectorized version of the simulation function.
    Processes the entire dataset at once for massive speedup.
    """
    # Step 1: Get all raw predictions for the new data in one shot
    new_predictions = model.predict(X_data)

    # Step 2: Vectorized percentile calculation
    # First, sort the historical predictions distribution
    sorted_all_preds = np.sort(all_preds_dist)
    # Use np.searchsorted to find the rank of each new prediction
    # This is a highly optimized way to do what percentileofscore does in a loop
    ranks = np.searchsorted(sorted_all_preds, new_predictions, side="right")
    percentiles = ranks / len(sorted_all_preds) * 100

    # Step 3: Get all simulated outcomes from the real distribution in one shot
    simulated_outcomes = np.percentile(actual_yards_dist, percentiles)

    return np.round(simulated_outcomes)


print("\n--- Simulation Output Evaluation ---")
print("Generating simulated outcomes for the entire test set...")
# Call the new vectorized function instead of the loop
simulated_outcomes = vectorized_simulate_yards(
    X_test, lgbm_yards, all_predictions, actual_yards
)

# Now you can proceed with your comparisons as before
real_dist = pd.Series(actual_yards)
sim_dist = pd.Series(simulated_outcomes)

print("\nComparing Descriptive Statistics:")
stats_df = pd.DataFrame(
    {"Real Yards": real_dist.describe(), "Simulated Yards": sim_dist.describe()}
)
print(stats_df)

# 2. Visualize: Distribution Comparison Plot
plt.figure(figsize=(12, 7))
sns.histplot(real_dist, color="skyblue", kde=True, label="Real Yards", stat="density")
sns.histplot(sim_dist, color="red", kde=True, label="Simulated Yards", stat="density")
plt.title("Distribution of Real vs. Simulated Yards Gained")
plt.xlabel("Yards Gained")
plt.legend()
plt.show()

# 3. Visualize: Q-Q Plot (Quantile-Quantile)
plt.figure(figsize=(8, 8))
sm.qqplot(real_dist, line="45", other=sim_dist, fmt="o", ms=4)
plt.title("Q-Q Plot: Real vs. Simulated Quantiles")
plt.xlabel("Real Quantiles")
plt.ylabel("Simulated Quantiles")
plt.grid(True)
plt.show()
