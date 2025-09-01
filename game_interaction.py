from random import random
import pandas as pd
from organize_data import iterate_df
from predictors import New_Predictor
import statistics as stat
from termcolor import colored
import pickle
import joblib

# TODO: Safeties, Touchbacks, Penatlies!!


class Game:
    def __init__(self, home_team, away_team, df, print=True, max_effective_time=900):
        self.home = home_team
        self.away = away_team
        if random() > 0.5:
            self.pos_team = self.home
            self.def_team = self.away
        else:
            self.pos_team = self.away
            self.def_team = self.home
        self.is_kickoff = True
        self.is_extra_point = True
        self.scores = {self.home: 0, self.away: 0}
        self.ydline_100_pos = None
        self.first_down = None
        self.new_predictor = New_Predictor()
        self.ppa_model = joblib.load("ppa_model.pkl")
        self.time_left = 3600
        self.effective_time_left = 900
        self.max_effective_time = max_effective_time
        self.first_down_100 = None
        self.down = None
        self.print = print

        self.pos_minus_def_score = None
        self.yd_to_go = None
        self.qtr = 1
        self.number_of_plays = 0
        self.previous_run = 0
        self.previous_ppa = 0
        self.previous_ppa_2 = 0

    def update_basics(self):
        self.pos_minus_def_score = (
            self.scores[self.pos_team] - self.scores[self.def_team]
        )
        self.pos_down_three_or_less = 1 if -3 <= self.pos_minus_def_score <= 0 else 0
        self.pos_winning = 1 if self.pos_minus_def_score > 0 else 0
        self.ydstogo = self.ydline_100_pos - self.first_down_100
        if 4 - int(self.time_left / 900) > self.qtr:
            print(colored("\tQuarter:" + str(self.qtr + 1), "blue"))
        self.qtr = 4 - int(self.time_left / 900)
        if self.time_left > self.max_effective_time:
            self.effective_time_left = self.max_effective_time
        else:
            self.effective_time_left = self.time_left

    def simulate_play(self):  # self.pos_team is receiving
        if self.check_end_game():
            return True
        if self.is_kickoff:
            self.kickoff()
            self.is_kickoff = False
            return
        self.update_basics()
        is_FG = False
        is_punt = False  # self.check_and_do_punt()
        if is_punt:
            run_pass = False
        else:
            is_FG, FG_made = False, False  # self.check_and_do_FG()
            run_pass = not is_FG

        if run_pass:
            self.number_of_plays += 1
            if self.print:
                print("play count:", self.number_of_plays)
            is_run = self.predict_run_vs_pass()
            # penalty, penalty_yds, penalty_team = self.penalty_check()
            # fumble_or_int, ball_lost, yd_change = self.turnover_check(is_run)
            # if penalty:
            #    self.move_yds(penalty_yds, False, f'Penalty on {penalty_team} for {penalty_yds}.')
            # elif fumble_or_int:
            #    self.move_yds(yd_change, ball_lost, move_yd_str='Fumbled into a 1st down:')
            # else:
            self.run_pass_play(is_run)
            t_play = 5  # int(self.get_play_t_length(is_run) * 1.0) # *1.15 this lowers the number of plays. BIG SHAME!!!
        else:
            t_play = 10

        if not is_FG:
            self.check_TD()

        # print('Play length:', self.get_play_t_length(is_run))
        if t_play < 0 or t_play > 75:
            for i in range(0, 100):
                print(colored("ERROR IN T_PLAY", "green"))
        self.time_left = self.time_left - t_play
        self.check_4th_down()

    def check_punt(self):
        if self.down != 4:  # catches ~99.999% of plays I imagine
            return False, None
        else:
            is_punt, real_p = self.basic_predictors["punt"].predict(
                pd.DataFrame.from_dict(
                    {
                        "punt": [None],
                        "pos_minus_def_score": [self.pos_minus_def_score],
                        "time_left": [self.effective_time_left],
                        "ydstogo": [self.ydstogo],
                        "pos_winning": [self.pos_winning],
                        "pos_team": [self.pos_team],
                        "def_team": [self.def_team],
                        "goal_yd": [self.ydline_100_pos],
                    }
                ),
                get_real_p=True,
            )
            if self.print:
                print("p(punt) =", real_p)
            # TODO: implement blocked punts
            if is_punt:
                punt_yds = self.basic_predictors["punt_yds"].predict(
                    pd.DataFrame.from_dict(
                        {"punt_yds": [None], "goal_yd": [self.ydline_100_pos]}
                    )
                )
                # if self.print:
                print(
                    self.pos_team,
                    "punts from the",
                    int(self.ydline_100_pos),
                    "it goes:",
                    int(punt_yds),
                    "yds.",
                )
                return is_punt, punt_yds
            else:
                return is_punt, None

    def check_and_do_punt(self):
        is_punt, punt_yds = self.check_punt()
        if is_punt:
            self.move_yds(punt_yds, True, "Punting the ball. Possession of")
        if self.ydline_100_pos > 100:
            self.move_yds(
                self.ydline_100_pos - 80,
                False,
                "Touchback. Ball placed at 80 yard line",
            )
        return is_punt

    def check_FG(self):
        if self.down != 4:  # catches ~99.999% of plays I imagine
            return False, None
        elif self.ydline_100_pos > 60:
            return False, None
        else:
            is_FG, real_p = self.basic_predictors["FG_attempted"].predict(
                pd.DataFrame.from_dict(
                    {
                        "FG_attempted": [None],
                        "pos_minus_def_score": [self.pos_minus_def_score],
                        "pos_down_three_or_less": [self.pos_down_three_or_less],
                        "time_left": [self.effective_time_left],
                        "pos_team": [self.pos_team],
                        "def_team": [self.def_team],
                        "ydstogo": [self.ydstogo],
                        "goal_yd": [self.ydline_100_pos],
                    }
                ),
                get_real_p=True,
            )
            if self.print:
                print("p(FG_attempt) =", real_p)
            if is_FG:
                FG_made = self.basic_predictors["FG_made"].predict(
                    pd.DataFrame.from_dict(
                        {
                            "FG_made": [None],
                            "pos_team": [self.pos_team],
                            "goal_yd": [self.ydline_100_pos],
                        }
                    )
                )
                return is_FG, FG_made
            else:
                return is_FG, None

    def check_and_do_FG(self):
        is_FG, FG_made = self.check_FG()
        if is_FG:
            if FG_made:
                self.is_kickoff = True
                self.scores[self.pos_team] += 3
                print(
                    colored(
                        "FG is good from "
                        + str(int(self.ydline_100_pos))
                        + ". Scores: "
                        + str(self.get_score_str()),
                        "green",
                    )
                )
            else:
                print(
                    colored(
                        "FG failed from "
                        + str(int(self.ydline_100_pos))
                        + ". Scores: "
                        + str(self.get_score_str()),
                        "red",
                    )
                )

                self.ydline_100_pos += 10
            self.transfer_possession(dont_print=True)
        return is_FG, FG_made

    def move_yds(self, yd_change, ball_lost, move_yd_str=""):
        self.ydline_100_pos = float(int(self.ydline_100_pos - yd_change))
        if ball_lost:
            self.transfer_possession()
        else:
            if self.is_first_down():
                self.reset_downs()
                print(move_yd_str, self.pos_team, "ball. 1st and 10 to go.")
            else:
                self.down += 1

    def check_TD(self):
        if self.ydline_100_pos < 0:
            self.scores[self.pos_team] += 7
            if self.print:
                print()
            print(
                colored(
                    "Touchdown "
                    + str(self.pos_team)
                    + "!! Scores: "
                    + str(self.get_score_str()),
                    "green",
                )
            )
            self.transfer_possession(dont_print=True)
            self.kickoff()

    def penalty_check(self):
        is_penalty = self.basic_predictors["penalty"].predict(
            pd.DataFrame.from_dict(
                {
                    "penalty": [None],
                    "pos_team": [self.pos_team],
                    "def_team": [self.def_team],
                    "goal_yd": [self.ydline_100_pos],
                    "ydstogo": [self.ydstogo],
                    "down": [self.down],
                    "pos_minus_def_score": [self.pos_minus_def_score],
                    "time_left": [self.effective_time_left],
                }
            )
        )
        is_penalty = 0
        if is_penalty:
            poss_team_penalty = self.basic_predictors["penalty_poss"].predict(
                pd.DataFrame.from_dict(
                    {
                        "penalty": [None],
                        "pos_team": [self.pos_team],
                        "def_team": [self.def_team],
                        "goal_yd": [self.ydline_100_pos],
                        "ydstogo": [self.ydstogo],
                        "down": [self.down],
                        "pos_minus_def_score": [self.pos_minus_def_score],
                        "time_left": [self.effective_time_left],
                    }
                )
            )

            penalty_accepted = self.basic_predictors["penalty_accepted"].predict(
                pd.DataFrame.from_dict(
                    {
                        "penalty_acpt": [None],
                        "penalty_pos": [poss_team_penalty],
                        "goal_yd": [self.ydline_100_pos],
                        "ydstogo": [self.ydstogo],
                        "down": [self.down],
                        "pos_minus_def_score": [self.pos_minus_def_score],
                        "time_left": [self.effective_time_left],
                    }
                )
            )
            if poss_team_penalty:
                penalty_team = self.pos_team
            else:
                penalty_team = self.def_team
            if penalty_accepted:
                penalty_yds = 5 * (
                    abs(
                        self.basic_predictors["penalty_yds"].predict(
                            pd.DataFrame.from_dict(
                                {
                                    "yd_gain": [None],
                                    "penalty_pos": [poss_team_penalty],
                                    "goal_yd": [self.ydline_100_pos],
                                    "ydstogo": [self.ydstogo],
                                    "down": [self.down],
                                    "time_left": [self.effective_time_left],
                                }
                            )
                        )
                    )
                    / 5
                )
                if poss_team_penalty:
                    return True, -penalty_yds, penalty_team
                else:
                    return True, penalty_yds, penalty_team
            else:
                return None, None, None
        else:
            return None, None, None

    def turnover_check(self, is_run):
        is_fumble = self.basic_predictors["fumble"].predict(
            pd.DataFrame.from_dict(
                {
                    "fumble": [None],
                    "pos_team": [self.pos_team],
                    "pos_winning": [self.pos_winning],
                    "run_vs_pass": [is_run],
                    "time_left": [self.time_left],
                }
            )
        )
        if is_fumble:
            # is_fumble_lost = self.basic_predictors['fumble_lost'].predict(pd.DataFrame.from_dict({'fumble_lost': [None],
            #                                                                                  'def_team': [self.def_team],
            #                                                                                  'run_vs_pass': [is_run],
            #                                                                                  'pos_winning': [self.pos_winning]}))
            is_fumble_lost = random() > 0.5
            # TODO: implement Distribution manager here
            try:
                fumble_spot_yds = self.basic_predictors["fumble_yds"].predict(
                    pd.DataFrame.from_dict(
                        {
                            "fumble_spot_yds": [None],
                            "def_team": [self.def_team],
                            "fumble_lost": [is_fumble_lost],
                        }
                    )
                )
            except:
                fumble_spot_yds = 0
            if is_fumble_lost:
                print(
                    colored(
                        self.pos_team + " fumbled! Recovered by " + self.def_team + "!",
                        "red",
                    )
                )
            else:
                print(
                    colored(
                        self.pos_team + " fumbled! But they recovered it back.",
                        "magenta",
                    )
                )

            return True, is_fumble_lost, int(fumble_spot_yds + 0.5)

        if is_run:
            return None, None, None
        is_int = self.basic_predictors["interception"].predict(
            pd.DataFrame.from_dict(
                {
                    "interception": [None],
                    "pos_team": [self.pos_team],
                    "def_team": [self.def_team],
                    "time_left": [self.effective_time_left],
                }
            )
        )
        if is_int:
            # TODO: implement Distribution manager here
            try:
                int_spot_yds = self.basic_predictors["int_yds"].predict(
                    pd.DataFrame.from_dict(
                        {"int_spot_yds": [None], "def_team": [self.def_team]}
                    )
                )
            except:
                int_spot_yds = 0
                print(
                    colored(
                        self.pos_team + " intercepted by " + self.def_team + "!", "red"
                    )
                )
            return True, True, int(int_spot_yds + 0.5)
        else:
            return None, None, None

    def kickoff(self):
        self.ydline_100_pos = 75
        self.first_down_100 = 65
        self.down = 1
        if self.print:
            print(
                "Kickoff:",
                self.pos_team,
                "gets the ball at their",
                self.ydline_100_pos,
                "yard line. 1st down.",
            )

    def predict_run_vs_pass(self):
        features = {
            "score_diff": [self.pos_minus_def_score],
            "yardsToGoal": [self.ydline_100_pos],
            "down_distance": [self.down * self.ydstogo],
            "ppa_change": [self.previous_ppa - self.previous_ppa_2],
            "is_critical_down": [1 if self.down in [3, 4] else 0],
            "is_short_distance": [1 if self.ydstogo <= 2 else 0],
            "is_red_zone": [1 if self.ydline_100_pos <= 20 else 0],
            "is_trailing": [1 if self.pos_minus_def_score < 0 else 0],
            "is_tied": [1 if self.pos_minus_def_score == 0 else 0],
            "previous_ppa": [self.previous_ppa],
            "is_leading": [1 if self.pos_minus_def_score > 0 else 0],
            "is_late_game": [1 if self.time_left <= 300 else 0],
            "previous_run": [self.previous_run],
            "is_home_offense": [1 if self.pos_team == self.home else 0],
            "is_away_offense": [1 if self.pos_team == self.away else 0],
            "is_own_territory": [1 if self.ydline_100_pos > 50 else 0],
            "is_midfield": [1 if self.ydline_100_pos == 50 else 0],
            "is_opponent_territory": [1 if self.ydline_100_pos < 50 else 0],
        }
        features_df = pd.DataFrame.from_dict(features)
        is_run = self.new_predictor.predict_play_type(features_df)
        if self.print:
            print("p(run)=", is_run)
        return is_run

    def run_pass_play(self, is_run):
        features = {
            "down": [self.down],
            "distance": [self.ydstogo],
            "yardsToGoal": [self.ydline_100_pos],
            "score_diff": [self.pos_minus_def_score],
            "time_left_in_game": [self.time_left],
            "is_critical_down": [1 if self.down in [3, 4] else 0],
            "is_short_distance": [1 if self.ydstogo <= 2 else 0],
            "is_red_zone": [1 if self.ydline_100_pos <= 20 else 0],
            "is_late_game": [1 if self.time_left <= 300 else 0],
            "is_home_offense": [1 if self.pos_team == self.home else 0],
            "is_away_offense": [1 if self.pos_team == self.away else 0],
            "is_own_territory": [1 if self.ydline_100_pos > 50 else 0],
            "is_midfield": [1 if self.ydline_100_pos == 50 else 0],
            "is_opponent_territory": [1 if self.ydline_100_pos < 50 else 0],
            "pre_game_off_avg_pass_yards": [0],  # Placeholder
            "pre_game_def_avg_pass_yards_allowed": [0],  # Placeholder
            "pre_game_pass_matchup": [0],  # Placeholder
            "obvious_pass_situation": [
                1 if self.down >= 3 and self.ydstogo >= 7 else 0
            ],
            "previous_ppa": [self.previous_ppa],
            "ppa_change": [self.previous_ppa - self.previous_ppa_2],
            "run": [is_run],
            "pre_game_off_avg_rush_yards": [0],  # Placeholder
            "pre_game_def_avg_rush_yards_allowed": [0],  # Placeholder
            "pre_game_run_matchup": [0],  # Placeholder
            "obvious_run_situation": [1 if self.down >= 3 and self.ydstogo <= 2 else 0],
        }
        features_df = pd.DataFrame.from_dict(features)
        yd_gain_pred = self.new_predictor.predict_yards_gained(is_run, features_df)

        ppa_features = {
            "score_diff": [self.pos_minus_def_score],
            "yardsToGoal": [self.ydline_100_pos],
            "yardsGained": [yd_gain_pred],
            "down_distance": [self.down * self.ydstogo],
            "is_critical_down": [1 if self.down in [3, 4] else 0],
            "is_short_distance": [1 if self.ydstogo <= 2 else 0],
            "is_red_zone": [1 if self.ydline_100_pos <= 20 else 0],
            "is_trailing": [1 if self.pos_minus_def_score < 0 else 0],
            "is_tied": [1 if self.pos_minus_def_score == 0 else 0],
            "is_leading": [1 if self.pos_minus_def_score > 0 else 0],
            "is_late_game": [1 if self.time_left <= 300 else 0],
            "run": [is_run],
            "is_home_offense": [1 if self.pos_team == self.home else 0],
            "is_away_offense": [1 if self.pos_team == self.away else 0],
            "is_own_territory": [1 if self.ydline_100_pos > 50 else 0],
            "is_midfield": [1 if self.ydline_100_pos == 50 else 0],
            "is_opponent_territory": [1 if self.ydline_100_pos < 50 else 0],
        }
        ppa_features_df = pd.DataFrame.from_dict(ppa_features)
        new_ppa = self.ppa_model.predict(ppa_features_df)[0]

        self.previous_ppa_2 = self.previous_ppa
        self.previous_ppa = new_ppa

        self.ydline_100_pos -= yd_gain_pred
        if self.is_first_down():
            self.reset_downs()
        else:
            self.down += 1
        if self.down <= 4:
            if self.print:
                print(
                    self.is_run_to_str(is_run, yd_gain_pred),
                    self.pos_team,
                    "gains",
                    yd_gain_pred,
                    "to",
                    self.ydline_100_pos,
                    "yd line.",
                    self.get_togo_str(),
                )
        else:
            print(
                self.is_run_to_str(is_run, yd_gain_pred),
                self.pos_team,
                "gains",
                yd_gain_pred,
                "to",
                self.ydline_100_pos,
                "yd line. Transferring possession!",
            )
            # python colored print()

    def check_4th_down(self):
        if self.down == 5:
            self.transfer_possession()

    def reset_downs(self):
        self.first_down_100 = self.ydline_100_pos - 10
        self.down = 1

    def transfer_possession(self, dont_print=False):
        self.ydline_100_pos = 100 - self.ydline_100_pos
        temp = self.def_team
        self.def_team = self.pos_team
        self.pos_team = temp
        self.reset_downs()
        if self.print:
            if not dont_print:
                print(
                    self.pos_team,
                    "gets the ball at",
                    str(self.ydline_100_pos) + ".",
                    self.get_togo_str(),
                )

    def is_first_down(self):
        if self.ydline_100_pos <= self.first_down_100:
            return True
        else:
            return False

    def get_togo_str(self):
        return (
            str(self.down)
            + " down "
            + str(self.ydline_100_pos - self.first_down_100)
            + " to go. Time: "
            + str(self.time_left)
        )

    def get_score_str(self):
        return (
            self.home
            + ": "
            + str(self.scores[self.home])
            + ", "
            + self.away
            + ": "
            + str(self.scores[self.away])
        )

    def check_end_game(self):
        if self.time_left <= 0:
            if self.print:
                print()
                print("Game over. Final score:", self.get_score_str())
            return True

    def get_play_t_length(self, is_run):
        current_play_df = pd.DataFrame.from_dict(
            {
                "time_left": [self.effective_time_left],
                "pos_team": self.pos_team,
                "play_t_length": -999,
                "run_vs_pass": is_run,
                "pos_winning": self.pos_winning,
            }
        )
        tile = self.PMs["play_t_length"].predict(current_play_df)
        if self.print:
            print("---")
            print("play len tile:", tile)
            print("t:", self.DMs["play_t_length"].get_val(is_run, tile))
        return self.DMs["play_t_length"].get_val(is_run, tile)

    def is_run_to_str(self, is_run, yd_gain_pred):
        if is_run:
            return "Run play."
        else:
            if yd_gain_pred < 0:
                return "Sack."
            else:
                return "Pass play."


class Game_Stats_Manager:
    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team
        self.received_first = None
        self.interceptions = {self.home_team: 0, self.away_team: 0}
        self.fumbles = {self.home_team: 0, self.away_team: 0}
        self.yards = {self.home_team: 0, self.away_team: 0}
        self.number_of_plays = 0

        # TODO: sacks

    def add_interception(self, team):
        self.interceptions[team] += 1

    def add_fumble(self, team):
        self.fumbles[team] += 1

    def add_yards(self, team, yards):
        self.yards[team] += yards

    def set_received_first(self, team):
        self.received_first = team

    def add_number_of_plays(self):
        self.number_of_plays += 1


def duo_processor(df, team_A, team_B, percentile_break=0.1):
    A_minus_B_history = []
    for sim_num in range(0, 250):
        game = Game(
            team_A, team_B, df, print=True, max_effective_time=max_effective_time
        )
        for i in range(0, 300):
            is_game_ended = game.simulate_play()
            if is_game_ended:
                final_score = game.scores
                break
        A_minus_B_history.append(final_score[team_A] - final_score[team_B])
        print(
            "Median", team_A, "minus", team_B, "score:", stat.median(A_minus_B_history)
        )
        A_minus_B_history.sort()
        percentile_results_str = "n = " + str(len(A_minus_B_history)) + ". "
        for percentile in frange(0, 1.0, step=percentile_break):
            n_tile = int(len(A_minus_B_history) * percentile)
            score_dif = A_minus_B_history[n_tile]
            percentile_results_str = (
                percentile_results_str
                + " | "
                + str(round(percentile, 2))
                + "%: = "
                + str(score_dif)
            )
        print(percentile_results_str)


def frange(start, stop, step=1.0):
    while start < stop:
        yield start
        start += step


def simulate_game(df, team_A=None, team_B=None, print=True):
    if team_A is None:
        team_A = TEAM_A
    if team_B is None:
        team_B = TEAM_B
    game = Game(team_A, team_B, df, print=print, max_effective_time=max_effective_time)
    final_score = None
    for i in range(0, 200):
        is_game_ended = game.simulate_play()
        if is_game_ended:
            final_score = game.scores
            break
    return final_score[team_A], final_score[team_B]


def load_df_and_simulate_game(team_A, team_B, year, week):
    df = get_df(year, week)
    return simulate_game(df, team_A=team_A, team_B=team_B)


def get_df(year, week):

    week = (
        week - 1
    )  # 1.10.2020 game_interaction.py only works with data from weeks [1:5] inclusive. So start on week 6
    file_path = "pbp\\" + str(year) + "_" + str(week) + "_pbp.pkl"
    with open(file_path, "rb") as file:
        df = pickle.load(file)
    return df


# TODO: add FGs
# TODO: add pos_team and def_team as predictors
# TODO: analyze season of play

TEAM_A = "Boston College"
TEAM_B = "Duke"

if __name__ == "__main__":
    import numpy as np

    df = pd.DataFrame()
    fbs_conf = [
        "SEC",
        "Mountain West",
        "American Athletic",
        "Pac-12",
        "Mid-American",
        "Sun Belt",
        "ACC",
        "Big 12",
        "Big Ten",
        "FBS Independents",
        "Conference USA",
    ]
    for wk in range(1, 6):
        nxt_wk = pd.read_csv(f"raw_data/2023_{wk}_pbp.csv", engine="pyarrow")
        nxt_wk = nxt_wk.loc[
            (nxt_wk.offense_conference.isin(fbs_conf))
            & (nxt_wk.defense_conference.isin(fbs_conf))
        ]
        df = pd.concat([df, nxt_wk])

    df = pd.concat(
        [
            pd.read_csv(
                f"/Users/jaketolleson/Library/CloudStorage/GoogleDrive-jaketolleson7@gmail.com/My Drive/01_Projects/CFB/starter_pack/data/plays/2024/regular_{i}_plays.csv"
            )
            for i in range(10, 16)
        ],
        ignore_index=True,
    )
    df = iterate_df(df)
    # df.to_csv('sim_df.csv',index=False)
    # df = pd.read_csv('sim_df.csv')
    df = df.astype({"punt_yds": float})
    df["punt"] = df["punt"].fillna(0.0)
    df["fumble"] = df["fumble"].fillna(0.0)
    max_effective_time = 900
    # if we have items with time_left = 3600 or really big that will skew the regressions too heavily
    df["time_left"] = df["time_left"].apply(
        lambda x: max_effective_time if x > max_effective_time else x
    )
    df.to_csv("sim_df.csv", index=False)
    duo_processor(df, TEAM_A, TEAM_B)
    A_scores = []
    B_scores = []
    A_wins = []
    for j in range(0, 250):
        A, B = simulate_game(df, print=True)
        A_scores.append(A)
        B_scores.append(B)
        if A > B:
            A_wins.append(1)
        elif A == B:
            A_wins.append(0.5)
        else:
            A_wins.append(0)
        print()
        print("--*---*---")
        print(
            "Average",
            TEAM_A,
            "scores:",
            stat.mean(A_scores),
            ",",
            TEAM_B,
            "scores:",
            stat.mean(B_scores),
        )
        print("Proportion of", TEAM_A, "wins:", stat.mean(A_wins))
        print("iteration:", j)

    df = pd.read_csv("sim_df.csv")
    new_games = pd.read_csv("new_espn_probs.csv")
    new_games["home_prob"] = np.nan
    new_games["home_mean"] = np.nan
    new_games["home_std"] = np.nan
    new_games["away_mean"] = np.nan
    new_games["away_std"] = np.nan
    new_games["pred_spread"] = np.nan

    # Sim one game
    max_effective_time = 900
    TEAM_A = "Eastern Michigan"
    TEAM_B = "Akron"
    A_scores = []
    B_scores = []
    A_wins = []
    game = Game(TEAM_A, TEAM_B, df, print=True, max_effective_time=max_effective_time)

    for row in new_games.index:
        TEAM_A = new_games.loc[row, "home_team"]
        TEAM_B = new_games.loc[row, "away_team"]
        A_scores = []
        B_scores = []
        A_wins = []
        game = Game(
            TEAM_A, TEAM_B, df, print=False, max_effective_time=max_effective_time
        )

        for j in range(512):
            final_score = None
            for i in range(0, 201):
                is_game_ended = game.simulate_play()
                if is_game_ended:
                    final_score = game.scores
                    A = final_score[TEAM_A]
                    B = final_score[TEAM_B]
                    A_scores.append(A)
                    B_scores.append(B)
                    if A > B:
                        A_wins.append(1)
                    elif A == B:
                        A_wins.append(0.5)
                    else:
                        A_wins.append(0)
                    break
            if random() > 0.5:
                game.pos_team = game.home
                game.def_team = game.away
            else:
                game.pos_team = game.away
                game.def_team = game.home
            game.is_kickoff = True
            game.is_extra_point = True
            game.scores = {game.home: 0, game.away: 0}
            game.ydline_100_pos = None
            game.first_down = None
            game.time_left = 3600
            game.effective_time_left = 900
            game.max_effective_time = max_effective_time
            game.first_down_100 = None
            game.down = None

            game.pos_minus_def_score = None
            game.yd_to_go = None
            game.qtr = 1
            game.number_of_plays = 0
            print()
            print("--*---*---")
            print(
                "Average",
                TEAM_A,
                "scores:",
                stat.mean(A_scores),
                ",",
                TEAM_B,
                "scores:",
                stat.mean(B_scores),
            )
            print("Proportion of", TEAM_A, "wins:", stat.mean(A_wins))
            print("iteration:", j)

        pd.DataFrame(
            dict(
                [
                    (k, pd.Series(v))
                    for k, v in {
                        "home_team": [TEAM_A],
                        "away_team": [TEAM_B],
                        "home_score": A_scores,
                        "away_score": B_scores,
                    }.items()
                ]
            )
        ).fillna(method="ffill").to_csv(
            f"new_games/{TEAM_A}vs{TEAM_B}.csv", index=False
        )

# dist_A = distfit()
# dist_B = distfit()

# # Search for best theoretical fit on your emperical data
# dist_A.fit_transform(np.array(A_scores))
# dist_B.fit_transform(np.array(B_scores))

# home_scores = dist_A.generate(10000000)
# away_scores = dist_B.generate(10000000)
# diff_scores = away_scores - home_scores

# new_games.loc[row, 'pred_spread'] = np.median(diff_scores)
# new_games.loc[row, 'home_prob'] = np.mean(diff_scores < 0)
# new_games.loc[row, 'home_mean'] = np.mean(home_scores)
# new_games.loc[row, 'home_std'] = np.std(home_scores)
# new_games.loc[row, 'away_mean'] = np.mean(away_scores)
# new_games.loc[row, 'away_std'] = np.std(away_scores)


# import numpy as np
# from statsmodels.stats.proportion import proportion_confint
# import matplotlib.pyplot as plt
# from distfit import distfit

# -(5*(200000*np.log(-stat.mean(A_wins)/(stat.mean(A_wins)-1))+ 16519))/123143
# np.sort(df.home.unique())
# stat.mean(B_scores) - stat.mean(A_scores)
# differences = np.array(B_scores) - np.array(A_scores)
# np.mean(differences < -1.0)

# np.mean(np.array(A_scores) > 40)


# proportion_confint(sum(A_wins), len(A_wins))

# # Initialize
# dist_A = distfit()
# dist_B = distfit()

# # Search for best theoretical fit on your emperical data
# dist_A.fit_transform(np.array(A_scores))
# dist_B.fit_transform(np.array(B_scores))

# home_scores = dist_A.generate(10000000)
# away_scores = dist_B.generate(10000000)
# diff_scores = away_scores - home_scores

# np.mean(diff_scores)
# np.mean(diff_scores > 0)
# proportion_confint(sum(diff_scores < 0), len(diff_scores))
