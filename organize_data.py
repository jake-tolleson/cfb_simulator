import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
from random import random
from statsmodels.sandbox.distributions.extras import pdf_mvsk
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.stats as stats
import datetime
from copy import deepcopy
from time import time
import re


class Game:
    def __init__(self, home_team, away_team):
        self.home = home_team
        self.away = away_team
        self.plays = {}

    def is_same_game(self, row):
        if row["home"] == self.home and row["away"] == self.away:
            return True
        return False

    def process_row(self, row, next_row=None):
        def add_half_details():
            if row["period"] <= 2:
                play["half"] = 1
                play["overtime"] = 0
            elif row["period"] in [3, 4]:
                play["half"] = 2
                play["overtime"] = 0
            elif row["period"] > 4:
                play["half"] = None
                play["overtime"] = 1
            else:
                print("ERROR #0 - row['game_half'] != 'Half1' or 'Half2' or 'Overtime'")
                play["half"] = None

        def add_pass_details():
            if row["play_type"] in [
                "Rushing Touchdown",
                "Rush",
                "Fumble Recovery (Opponent)",
                "Fumble Recovery (Own)",
                "Fumble Return Touchdown",
            ]:
                play["run"] = 1
                play["short_pass"] = None
                play["deep_pass"] = None
                play["run_vs_pass"] = 1

            elif row["play_type"] in [
                "Pass Incompletion",
                "Pass Completion",
                "Pass Reception",
                "Passing Touchdown",
                "Interception Return Touchdown",
                "Interception",
                "Pass Interception Return",
                "Pass Interception",
            ]:
                x = np.random.random()
                if x < 0.9:
                    play["run"] = 0
                    play["short_pass"] = 1
                    play["deep_pass"] = 0
                    play["run_vs_pass"] = 0
                else:
                    play["run"] = 0
                    play["short_pass"] = 0
                    play["deep_pass"] = 1
                    play["run_vs_pass"] = 0
            else:
                # this is reached if its a field goal or so
                play["run"] = 0
                play["short_pass"] = 0
                play["deep_pass"] = 0
                play["run_vs_pass"] = None

            if row["play_type"] == "Sack":
                play["run_vs_pass"] = 0

        def add_type_of_play_details():
            if row["play_type"] in [
                "End Period",
                "End of Half",
                "End of Regulation",
                "End of Game",
            ]:
                play["type"] = "end_of_period"
            elif row["play_type"] in [
                "Defensive 2pt Conversion",
                "2pt Conversion",
                "Two Point Pass",
                "Two Point Rush",
            ]:
                play["type"] = "2pt"
                if row["play_type"] == "2pt Conversion":
                    play["2pt_success"] = 1
                else:
                    play["2pt_success"] = 0
            elif row["play_type"] in [
                "Blocked PAT",
                "Extra Point Missed",
                "Extra Point Good",
            ]:
                play["type"] = "1pt"
                if row["play_type"] == "Extra Point Good":
                    play["1pt_success"] = 1
                else:
                    play["1pt_success"] = 0
            else:
                play["type"] = row["play_type"]

        def add_home_away_score_details():
            if row["drive_is_home_offense"] == True:
                play["home_score"] = row["offense_score"]
                play["away_score"] = row["defense_score"]
            else:
                play["home_score"] = row["defense_score"]
                play["away_score"] = row["offense_score"]

            if play["home_score"] > play["away_score"]:
                play["home_winning"] = 1
                play["away_winning"] = 0
                play["scoreless_game"] = 0
                play["tie"] = 0
            elif play["away_score"] > play["home_score"]:
                play["home_winning"] = 0
                play["away_winning"] = 1
                play["scoreless_game"] = 0
                play["tie"] = 0
            elif play["away_score"] == 0:
                play["home_winning"] = None
                play["away_winning"] = None
                play["scoreless_game"] = 1
                play["tie"] = 1
            else:
                play["home_winning"] = None
                play["away_winning"] = None
                play["scoreless_game"] = 0
                play["tie"] = 1

        def add_pos_def_score_details():
            play["pos_score"] = row["offense_score"]
            play["def_score"] = row["defense_score"]
            play["pos_minus_def_score"] = play["pos_score"] - play["def_score"]
            if play["pos_score"] > play["def_score"]:
                play["pos_winning"] = 1
                play["def_winning"] = 0
                play["tie"] = 0
            elif play["away_score"] > play["home_score"]:
                play["pos_winning"] = 0
                play["def_winning"] = 1
                play["tie"] = 0
            else:
                play["pos_winning"] = 0
                play["def_winning"] = 0
                play["tie"] = 1

        def add_down_converted():
            if row["distance"] <= row["yards_gained"]:
                play["converted"] = True
            else:
                play["converted"] = False

        def add_td():
            if (row["scoring"] == True) & (
                row["drive_end_offense_score"] >= row["drive_start_offense_score"] + 6
            ):
                play["pos_td"] = 1
            else:
                play["pos_td"] = 0
            if (row["scoring"] == True) & (
                row["drive_end_defense_score"] >= row["drive_start_defense_score"] + 6
            ):
                play["def_td"] = 1
            else:
                play["def_td"] = 0

        def add_time_of_play():
            if next_row is None:
                play["play_t_length"] = play["qt_left"]
                play["last_play_of_game"] = 1
            else:
                play["play_t_length"] = row["time_left"] - next_row["time_left"]
                play["last_play_of_game"] = 0
                if play["play_t_length"] > 65 or play["play_t_length"] < 0:
                    play["play_t_length"] = None

        def add_penalty():
            if row["play_type"] == "Penalty":
                play["penalty"] = 1
                if row["yards_gained"] < 0:
                    play["penalty_team"] = row["offense_play"]
                else:
                    play["penalty_team"] = row["defense_play"]
                if play["penalty_team"] == play["pos_team"]:
                    play["penalty_pos"] = 1
                else:
                    play["penalty_pos"] = 0
                play["penalty_yd"] = row["yards_gained"]
                if row["yards_gained"] == 0:
                    play["penalty_acpt"] = 0
                else:
                    play["penalty_acpt"] = 1
            else:
                play["penalty"] = 0

        def add_turnover():
            if row["play_type"] in [
                "Fumble Return Touchdown",
                "Fumble Recovery (Opponent)",
                "Fumble Recovery (Own)",
            ]:
                play["fumble"] = 1
            if row["play_type"] == "Fumble Recovery (Opponent)":
                play["fumble_lost"] = 1
                play["fumble_spot_yds"] = row["yards_gained"]
            else:
                set_none(["fumble_lost", "fumble_spot_yds"])

            if row["play_type"] in [
                "Interception",
                "Pass Interception",
                "Pass Interception Return",
                "Interception Return Touchdown",
            ]:
                play["interception"] = 1
            else:
                play["interception"] = 0
            if play["interception"] == 1:
                play["int_spot_yds"] = row["yards_gained"]
            else:
                play["int_spot_yds"] = None

        def add_punt():
            if row["play_type"] in [
                "Blocked Punt",
                "Punt",
                "Punt Return Touchdown",
                "Blocked Punt Touchdown",
                "Punt Return",
            ]:
                play["punt"] = 1
                if row["play_type"] in ["Blocked Punt", "Blocked Punt Touchdown"]:
                    play["punt_blocked"] = 1
                else:
                    play["punt_blocked"] = 0
                if "touchback" in str(row["play_text"]):
                    play["punt_yds"] = play["goal_yd"] - 20
                elif str(row["play_text"]) == "nan":
                    play["punt_yds"] = 0
                elif re.search(r"\d+", row["play_text"]) is not None:
                    play["punt_yds"] = re.search(r"\d+", row["play_text"]).group()
                else:
                    play["punt_yds"] = 0
            else:
                set_none(["punt_blocked", "punt_yds"])

        def add_FG():
            play["pos_down_three_or_less"] = (
                1 if -3 <= play["pos_minus_def_score"] <= 0 else 0
            )
            if row["play_type"] in [
                "Missed Field Goal Return",
                "Blocked Field Goal",
                "Field Goal Good",
                "Field Goal Missed",
                "Missed Field Goal Return Touchdown",
                "Blocked Field Goal Touchdown",
            ]:
                play["FG_attempted"] = 1
                if row["play_type"] == "Field Goal Good":
                    play["FG_made"] = 1
                else:
                    play["FG_made"] = 0
            else:
                play["FG_attempted"] = 0
                play["FG_made"] = None

        def set_none(list_of_keys):
            for key in list_of_keys:
                play[key] = None

        def add_week():
            play["week"] = row["wk"]

        play = {}
        play["qt_left"] = row["clock.minutes"] * 60 + row["clock.seconds"]
        play["time_left"] = row["time_left"]
        play["home"] = self.home
        play["away"] = self.away
        play["down"] = row["down"]
        play["yd_gain"] = row["yardsGained"]
        play["ydstogo"] = row["distance"]
        play["id"] = "g" + str(row["gameId"]) + "p" + str(row["id"])
        play["pos_team"] = row["offense_play"]
        play["def_team"] = row["defense_play"]
        play["goal_yd"] = row["yards_to_goal"]
        play["qtr"] = row["period"]
        add_half_details()
        add_pass_details()
        add_type_of_play_details()
        add_home_away_score_details()
        add_pos_def_score_details()
        add_down_converted()
        add_td()
        add_time_of_play()
        add_penalty()
        add_turnover()
        add_punt()
        add_FG()
        add_week()
        self.plays[play["id"]] = play


def iterate_df(df):
    print("Running iterate_df...")
    print("Number of rows:", df.shape[0])
    games = []
    plays = {}
    i = 0
    df.index = np.arange(
        0, len(df)
    )  # index will start at some high number if not year 2009, the first year of data xl
    for idx, row in tqdm(df.iterrows()):
        if not i:
            game = Game(row["home"], row["away"])
            i += 1
        if not game.is_same_game(row):
            games.append(game)
            game = Game(row["home"], row["away"])
        elif idx + 1 < df.shape[0]:
            game.process_row(row, next_row=df.loc[idx + 1])
        else:
            game.process_row(row)
        plays.update(game.plays)

    plays_df = pd.DataFrame.from_dict(plays, "index")

    return plays_df


if __name__ == "__main__":
    # iterate_df
    SMALL = False
