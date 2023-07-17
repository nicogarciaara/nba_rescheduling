from model_utils import League, Scheduler
from ttp_model import TTPModel, get_disruptions_and_non_disruptions
import numpy as np
import warnings
import datetime
import cplex
import pandas as pd
import os
import sys
import pickle
import time
warnings.filterwarnings('ignore')
cwd = os.getcwd()
eda_wd = cwd.replace('models\\ttp model', 'eda')
sys.path.insert(0, eda_wd)
from scheduling_rules import *


def check_multi(df):
    teams = df['home'].unique()
    df_multi = pd.DataFrame()

    for team in teams:
        df_team = df[(df['home'] == team) | (df['visitor'] == team)]
        df_group = df_team.groupby('original_date').size().reset_index(name='n_games')
        df_group = df_group[df_group['n_games'] > 1]
        df_team = pd.merge(df_team, df_group, how='inner', on='original_date')
        df_team['Team'] = team
        df_multi = pd.concat([df_multi, df_team], ignore_index=True)
    return df_multi

# hola nico te amo:)
if __name__ == '__main__':
    leagues = ['nba']
    objectives = ['basic', 'double', 'squared']
    objectives = ['basic']
    for objective in objectives:
        #for distance_mode in ['mid', 'low']:
        for distance_mode in ['high']:
            #for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
            #for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
            for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
                #for reschedule_mode in ['monthly', 'post_all_star', 'ten_days']:
                #for reschedule_mode in ['monthly', 'post_all_star']:
                for reschedule_mode in ['post_all_star']:
                    for max_mods_per_tour in [2]:
                        #for feasibility_days in [5, 7, 10]:
                        for feasibility_days in [10]:
                            for league in leagues:
                                #for n_window in [0, 3, 5]:
                                #for n_window in [1, 2, 3, 4, 5]:
                                for n_window in [0]:
                                    for asterisk in [1]:
                                        for max_non_dis_mods in [1000]:
                                            if league == 'nba':
                                                if reschedule_mode == 'monthly':

                                                    starts = [datetime.datetime(2020, 12, 1),
                                                              datetime.datetime(2021, 1, 1),
                                                              datetime.datetime(2021, 2, 1),
                                                              datetime.datetime(2021, 3, 1),
                                                              datetime.datetime(2021, 4, 1),
                                                              datetime.datetime(2021, 5, 1)]
                                                    ends = [datetime.datetime(2020, 12, 31),
                                                            datetime.datetime(2021, 1, 31),
                                                            datetime.datetime(2021, 2, 28),
                                                            datetime.datetime(2021, 3, 31),
                                                            datetime.datetime(2021, 4, 30),
                                                            datetime.datetime(2021, 5, 31)]
                                                elif reschedule_mode == 'post_all_star':
                                                    starts = [datetime.datetime(2020, 12, 1),
                                                              datetime.datetime(2021, 3, 5),
                                                              datetime.datetime(2021, 4, 1),
                                                              datetime.datetime(2021, 5, 1)]
                                                    ends = [datetime.datetime(2021, 3, 4),
                                                            datetime.datetime(2021, 3, 31),
                                                            datetime.datetime(2021, 4, 30),
                                                            datetime.datetime(2021, 5, 31)]
                                                else:
                                                    starts = [datetime.datetime(2020, 12, 1),
                                                              datetime.datetime(2021, 1, 1),
                                                              datetime.datetime(2021, 1, 11),
                                                              datetime.datetime(2021, 1, 21),
                                                              datetime.datetime(2021, 2, 1),
                                                              datetime.datetime(2021, 2, 11),
                                                              datetime.datetime(2021, 2, 21),
                                                              datetime.datetime(2021, 3, 1),
                                                              datetime.datetime(2021, 3, 11),
                                                              datetime.datetime(2021, 3, 21),
                                                              datetime.datetime(2021, 4, 1),
                                                              datetime.datetime(2021, 4, 11),
                                                              datetime.datetime(2021, 4, 21),
                                                              datetime.datetime(2021, 5, 1),
                                                              datetime.datetime(2021, 5, 11),
                                                              datetime.datetime(2021, 5, 21)
                                                              ]
                                                    ends = [datetime.datetime(2020, 12, 31),
                                                            datetime.datetime(2021, 1, 10),
                                                            datetime.datetime(2021, 1, 20),
                                                            datetime.datetime(2021, 1, 31),
                                                            datetime.datetime(2021, 2, 10),
                                                            datetime.datetime(2021, 2, 20),
                                                            datetime.datetime(2021, 2, 28),
                                                            datetime.datetime(2021, 3, 10),
                                                            datetime.datetime(2021, 3, 20),
                                                            datetime.datetime(2021, 3, 31),
                                                            datetime.datetime(2021, 4, 10),
                                                            datetime.datetime(2021, 4, 20),
                                                            datetime.datetime(2021, 4, 30),
                                                            datetime.datetime(2021, 5, 10),
                                                            datetime.datetime(2021, 5, 20),
                                                            datetime.datetime(2021, 5, 31)]
                                            else:
                                                starts = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 2, 1),
                                                          datetime.datetime(2021, 3, 1), datetime.datetime(2021, 4, 1), datetime.datetime(2021, 5, 1)]
                                                ends = [datetime.datetime(2021, 1, 31), datetime.datetime(2021, 2, 28),
                                                        datetime.datetime(2021, 3, 31), datetime.datetime(2021, 4, 30), datetime.datetime(2021, 5, 31)]
                                            start_time = time.time()
                                            L = League(league)
                                            if instance == 'basic':
                                                fixture = L.load_schedule()
                                            else:
                                                fixture = pd.read_csv(f'./other_instances/nba_schedule_{instance}.csv')
                                                fixture['original_date'] = pd.to_datetime(fixture['original_date'])
                                                fixture['game_date'] = pd.to_datetime(fixture['game_date'])
                                                fixture['final_date'] = pd.to_datetime(fixture['final_date'])
                                            fixture.loc[(fixture['home'] == 'Dallas Mavericks') & (
                                                    fixture['visitor'] == 'Detroit Pistons') & (
                                                fixture['original_date'] == datetime.datetime(2021, 2, 17)
                                            ), 'original_date'] = datetime.datetime(2021, 4, 21)

                                            fixture.loc[(fixture['home'] == 'Charlotte Hornets') & (
                                                    fixture['visitor'] == 'Chicago Bulls') & (
                                                fixture['original_date'] == datetime.datetime(2021, 2, 17)
                                            ), 'original_date'] = datetime.datetime(2021, 5, 6)
                                            if asterisk == 1:
                                                fixture['original_date_month'] = fixture['original_date'].dt.month
                                                fixture['game_date_month'] = fixture['game_date'].dt.month
                                                fixture['equal_month'] = 1*(fixture['original_date_month'] == fixture['game_date_month'])
                                                fixture.loc[(fixture['equal_month'] == 1), 'original_date'] = fixture['game_date']
                                                fixture.loc[(fixture['equal_month'] == 1), 'reschedule'] = 0

                                                fixture.loc[(fixture['reschedule'] == 1) & (fixture['game_date'] <= ends[0]), 'original_date'] = fixture['game_date']
                                                fixture.loc[(fixture['reschedule'] == 1) & (
                                                            fixture['game_date'] <= ends[0]), 'reschedule'] = 0

                                                fixture['day_difference'] = (fixture['game_date'] - fixture['original_date']).dt.days
                                                fixture.drop(columns=['original_date_month', 'game_date_month', 'equal_month'], inplace=True)

                                            fixture_detroit = fixture[fixture['visitor'] == 'Detroit Pistons']
                                            S = Scheduler(league, custom_fixture=fixture)
                                            covid_windows = S.calculate_resched_windows().copy()
                                            original_fixture = fixture.copy()

                                            all_needed_reschedules = []
                                            # We make reschedules for the matches of that month

                                            for s in range(len(starts)):
                                                check = 0
                                                start_date = starts[s]
                                                end_date = ends[s]
                                                print(f"Rescheduling matches for league {league} - objective {objective},  "
                                                      f"between {start_date.date()} and {end_date.date()} - distance mode {distance_mode} "
                                                      f"- instance {instance} - reschedule_mode {reschedule_mode} - adjustment days {n_window}"
                                                      f"- max mods per tour {max_mods_per_tour} - feasibility days {feasibility_days}")

                                                disruptions, non_disruptions = get_disruptions_and_non_disruptions(fixture, covid_windows,
                                                                                                                   start_date, end_date)
                                                disruptions
                                                match_buffer = []

                                                # Create the model and the lp problem
                                                M = TTPModel(league, custom_fixture=fixture, start_date=start_date, end_date=end_date,
                                                             distance_mode=distance_mode, disruptions=disruptions,
                                                             non_disruptions=non_disruptions, max_mods_per_tour=max_mods_per_tour,
                                                             max_adj_days=n_window, feasibility_days=feasibility_days,
                                                             max_non_dis_mods=max_non_dis_mods)
                                                dis = M.disruptions
                                                prob_lp = cplex.Cplex()

                                                # We create the variables that will go into the model
                                                x_var_dict, matches_to_be_scheduled, \
                                                diff_games_dict, non_matched_matches = M.create_decision_variables_dict(
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    objective=objective,
                                                    match_buffer=[],
                                                    max_adj_days=n_window
                                                )

                                                if reschedule_mode == 'post_all_star' and n_window == 5 and instance == '15_games_in_march':
                                                    output_df, x_variables = M.solve_lp(x_var_dict, diff_games_dict, prob_lp, objective, mip_gap=0.02)
                                                else:
                                                    output_df, x_variables = M.solve_lp(x_var_dict, diff_games_dict, prob_lp, objective)
                                                output_df_diff = output_df[output_df['proposed_date'] != output_df['original_date']]

                                                fixture = pd.merge(fixture, output_df[['home', 'visitor', 'game_date',
                                                                                       'proposed_date', 'model_reschedule']], how='left',
                                                                   on=['home', 'visitor', 'game_date'])

                                                # Update date on schedule in order to consider new dates in the feasibility calculation
                                                fixture.loc[fixture['model_reschedule'] == 1, 'original_date'] = fixture['proposed_date']
                                                # Delete new columns in order to make future merges possible
                                                fixture.drop(columns=['proposed_date', 'model_reschedule'], inplace=True)

                                                # We check the matches that will need a new reschedule and add it to our list
                                                new_reschedules_list = M.calculate_needed_reschedules(output_df)
                                                all_needed_reschedules = all_needed_reschedules + new_reschedules_list

                                                output = {
                                                    'fixture_old': original_fixture,
                                                    'fixture_new': fixture,
                                                    'x_var_dict': x_var_dict,
                                                    'x_variables': x_variables,
                                                    'covid_windows': covid_windows,
                                                    'output_df_diff': output_df_diff,
                                                    'disruptions': disruptions,
                                                    'non_disruptions': non_disruptions,
                                                    'variables_by_match': M.get_variables_by_match(x_var_dict)
                                                }
                                                with open(f'./debug/{start_date.date()}.pickle', 'wb') as handle:
                                                    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

                                            check = 1

                                            if check == 1:
                                                end_time = time.time()
                                                time_taken = (end_time - start_time)/60
                                                fixture['time'] = time_taken
                                                fixture.to_csv(f'./output/BasicModel_{league}_{objective}_{distance_mode}_{instance}_{reschedule_mode}_{n_window}_{max_mods_per_tour}_{asterisk}_{feasibility_days}_{max_non_dis_mods}Test.csv', index=False, encoding='utf-8 sig')

