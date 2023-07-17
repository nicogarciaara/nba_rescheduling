import pandas as pd
import numpy as np
import os
import warnings
import sys
from model_utils import League
warnings.filterwarnings('ignore')
cwd = os.getcwd()
eda_wd = cwd.replace('models\\ttp model', 'eda')
sys.path.insert(0, eda_wd)
import analysis_utils as au
from calculate_distance_and_breaks import *
from calculate_k_balance import *
from compare_schedules import *
from scheduling_rules import *
import pandas as pd
import datetime

if __name__ == '__main__':
    leagues = ['nba', 'nhl']
    objs = {'nba': ['basic', 'min_date'], 'nhl': ['basic']}
    objs = {'nba': ['basic', 'min_date'], 'nhl': ['basic']}
    distance_mode = {'nba': ['low', 'high'], 'nhl': []}

    # Read base CSVs that will be used for comparison (as they include the KPIs from the original schedules)
    distance_full = pd.read_csv(f'{eda_wd}/output_for_tableau/distance_analysis.csv')
    breaks_full = pd.read_csv(f'{eda_wd}/output_for_tableau/breaks_analysis.csv')
    balance_full = pd.read_csv(f'{eda_wd}/output_for_tableau/balance_analysis.csv')
    difference_full = pd.read_csv(f'{eda_wd}/output_for_tableau/day_difference_analysis.csv')
    schedule_full = pd.read_csv(f'{eda_wd}/output_for_tableau/games_by_date_analysis.csv')
    schedule_full['game_date'] = pd.to_datetime(schedule_full['game_date'])
    rules_full = pd.DataFrame()
    schedules_most_reschedules_teams = pd.DataFrame()
    execution_times = pd.DataFrame()

    for league in ['nba']:
        for obj in objs[league]:
            #for distance_mode in ['low', 'mid', 'high']:
            for distance_mode in ['high']:
                #for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
                #for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
                for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
                    #for reschedule_mode in ['monthly', 'post_all_star', 'ten_days']:
                    #for reschedule_mode in ['monthly', 'post_all_star']:
                    for reschedule_mode in ['post_all_star']:
                        #for max_mods_per_tour in [1, 2, 3]:
                        for max_mods_per_tour in [2]:
                            #for feasibility_days in [5, 7, 10]:
                            for feasibility_days in [10]:
                                #for n_window in [0, 3, 5]:
                                for n_window in [0, 3]:
                                    for asterisk in [0, 1]:
                                        for max_non_dis_mods in [5, 10]:
                                            for overlap_tours in [False]:
                                                print(obj, distance_mode, instance, reschedule_mode, n_window, max_mods_per_tour, asterisk, feasibility_days, asterisk, max_non_dis_mods, overlap_tours)
                                                check = 0
                                                try:
                                                    if n_window != -1:
                                                        df = pd.read_csv(f'./output/BasicModel_{league}_{obj}_{distance_mode}_{instance}_{reschedule_mode}_{n_window}_{max_mods_per_tour}_{asterisk}_{feasibility_days}_{max_non_dis_mods}_{overlap_tours}Test.csv')
                                                        #df = pd.read_csv(
                                                        #    f'./output/BasicModel_{league}_{obj}_{distance_mode}_{instance}_{reschedule_mode}_{n_window}_{max_mods_per_tour}_{asterisk}_{feasibility_days}.csv')
                                                    else:
                                                        df = pd.read_csv(
                                                            f'./output/BasicModel_{league}_{obj}_{distance_mode}_{instance}_{reschedule_mode}_{n_window}.csv')
                                                except:
                                                    df = 1
                                                    print("Cannot load", obj, distance_mode, instance, reschedule_mode, n_window, max_mods_per_tour, asterisk, feasibility_days)
                                                if type(df) != int:
                                                    exec_time = np.mean(df['time'])
                                                    if 'final_date' in df.columns:
                                                        df.rename(columns={'game_date': 'aux_date'}, inplace=True)
                                                        df.rename(columns={'final_date': 'game_date'}, inplace=True)
                                                        df['aux_date'] = pd.to_datetime(df['aux_date'])

                                                    df['game_date'] = pd.to_datetime(df['game_date'])
                                                    df['original_date'] = pd.to_datetime(df['original_date'])
                                                    df.rename(columns={'original_date': 'new_date'}, inplace=True)
                                                    df

                                                    # Read the original schedule
                                                    L = League(league)
                                                    schedule = L.load_schedule()
                                                    schedule['Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode} - {n_window} - {max_mods_per_tour} - {feasibility_days} - {asterisk} - {max_non_dis_mods} - {overlap_tours}"
                                                    schedule['Schedule Owner'] = 'NBAs'
                                                    schedule['League'] = league.upper()

                                                    # Merge both dataframes and rename columns
                                                    df = pd.merge(df, schedule[['home', 'visitor', 'game_date', 'original_date']], how='left',
                                                                  on=['home', 'visitor', 'game_date'])
                                                    df.rename(columns={'game_date': 'final_date', 'new_date': 'game_date'}, inplace=True)
                                                    df['day_difference'] = (df['game_date'] - df['original_date']).dt.days
                                                    if instance == 'basic':
                                                        df['final_day_difference'] = (df['final_date'] - df['original_date']).dt.days
                                                    else:
                                                        df['final_day_difference'] = (df['aux_date'] - df['original_date']).dt.days
                                                    df['PlusLastDate'] = 0
                                                    df.loc[
                                                        df['game_date'] > datetime.datetime(2021, 5, 16),
                                                        'PlusLastDate'] = 1
                                                    df_post = df[df['PlusLastDate'] == 1]
                                                    df_reschedule = df[df['reschedule'] == 1]
                                                    df_post
                                                    df_different = df[df['game_date'] != df['original_date']]
                                                    df_different = df_different[df_different['reschedule'] == 0]
                                                    rescheds = len(df_different)
                                                    # reschedule = 0
                                                    df_reschedule = df[(df['reschedule'] == 1) & (df['final_day_difference'] > 6)]

                                                    home_reschedules = df_different.groupby('home').size().reset_index(
                                                        name='reschedules').rename(columns={'home': 'team'})
                                                    away_reschedules = df_different.groupby('visitor').size().reset_index(
                                                        name='reschedules').rename(columns={'visitor': 'team'})
                                                    total_reschedules = pd.concat([home_reschedules, away_reschedules], ignore_index=True)
                                                    reschedules_by_team = total_reschedules.groupby('team')['reschedules'].sum().reset_index()
                                                    reschedules_by_team = reschedules_by_team.sort_values(by='reschedules', ascending=False)
                                                    top_reschedules = reschedules_by_team[reschedules_by_team['reschedules'] ==
                                                                                          np.max(reschedules_by_team['reschedules'])]
                                                    teams_with_most_reschedules = list(top_reschedules['team'])
                                                    #teams_with_most_reschedules = [teams_with_most_reschedules[0]]
                                                    df.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode} - {n_window} - {max_mods_per_tour} - {feasibility_days} - {asterisk} - {max_non_dis_mods} - {overlap_tours}"
                                                    df.loc[:, 'Schedule Owner'] = 'Us'
                                                    df.loc[:, 'League'] = league.upper()

                                                    df_top = df[(df['home'].isin(teams_with_most_reschedules)) | (
                                                        df['visitor'].isin(teams_with_most_reschedules))]
                                                    schedule_top = schedule[(schedule['home'].isin(teams_with_most_reschedules)) | (
                                                        df['visitor'].isin(teams_with_most_reschedules))]

                                                    schedules_most_reschedules_teams = pd.concat([
                                                        schedules_most_reschedules_teams, df_top, schedule_top], ignore_index=True)

                                                    # Calculate the different KPIs, first defining the necessity
                                                    teams = list(df['home'].unique())
                                                    dist_matrix = L.get_distance_matrix()
                                                    tournament_days = list(pd.date_range(np.min(df['game_date']), np.max(df['game_date'])))

                                                    df_distance = calculate_distance(df, dist_matrix, teams)
                                                    df_distance.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode} - {n_window} - {max_mods_per_tour} - {feasibility_days} - {asterisk} - {max_non_dis_mods} - {overlap_tours}"
                                                    df_distance.loc[:, 'League'] = league.upper()

                                                    df_breaks = calculate_breaks(df, teams)
                                                    df_breaks.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode} - {n_window} - {max_mods_per_tour} - {feasibility_days} - {asterisk} - {max_non_dis_mods} - {overlap_tours}"
                                                    df_breaks.loc[:, 'League'] = league.upper()

                                                    df_balance = calculate_k_balance(df, league, teams, tournament_days, games='all')
                                                    df_balance.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode} - {n_window} - {max_mods_per_tour} - {feasibility_days} - {asterisk} - {max_non_dis_mods} - {overlap_tours}"
                                                    df_balance.loc[:, 'League'] = league.upper()
                                                    df_balance.loc[:, 'Balance 7-day rolling mean'] = df_balance['diff'].rolling(
                                                        7, min_periods=1).mean()

                                                    df_diff = analyze_days_between_matches(df)
                                                    df_diff.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode} - {n_window} - {max_mods_per_tour} - {feasibility_days} - {asterisk} - {max_non_dis_mods} - {overlap_tours}"
                                                    df_diff.loc[:, 'League'] = league.upper()

                                                    # Concat with original measurements
                                                    distance_full = pd.concat([distance_full, df_distance], ignore_index=True)
                                                    breaks_full = pd.concat([breaks_full, df_breaks], ignore_index=True)
                                                    balance_full = pd.concat([balance_full, df_balance], ignore_index=True)
                                                    difference_full = pd.concat([difference_full, df_diff], ignore_index=True)

                                                    for col in schedule_full.columns:
                                                        if col in df.columns:
                                                            pass
                                                        else:
                                                            df[col] = ''
                                                    #df = df[list(schedule_full.columns)]
                                                    schedule_full = pd.concat([schedule_full, df], ignore_index=True)


                                                    
                                                    # Calculate scheduling rules so we can validate them we are not making any mistake
                                                    df_max_days = calculate_max_games_per_team(df, tournament_days, teams)
                                                    df_stats = df_max_days
                                    
                                                    # Create a max for each column
                                                    df_rules = pd.DataFrame()
                                                    df_rules.loc[:, 'Schedule Type'] = [f"{obj} - {distance_mode} - {instance} - {reschedule_mode} - {n_window}"]
                                                    df_rules.loc[:, 'League'] = [league.upper()]
                                                    stats_columns = [x for x in df_stats.columns if x not in ['Team']]
                                                    for col in stats_columns:
                                                        df_rules[col] = np.max(df_stats[col])
                                                    rules_full = pd.concat([rules_full, df_rules], ignore_index=True)
                                                    
                                                    print()

                                                    aux_time = pd.DataFrame(data={
                                                        'instance': [instance],
                                                        'n_window': [n_window],
                                                        'asterisk': [asterisk],
                                                        'time': [exec_time]
                                                    })
                                                    execution_times = pd.concat([execution_times, aux_time],
                                                                                ignore_index=True)



            distance_full.to_csv('./results_output/distance_analysis.csv', index=False, encoding='utf-8 sig')
            breaks_full.to_csv('./results_output/breaks_analysis.csv', index=False, encoding='utf-8 sig')
            balance_full.to_csv('./results_output/balance_analysis.csv', index=False, encoding='utf-8 sig')
            difference_full.to_csv('./results_output/day_difference_analysis.csv', index=False, encoding='utf-8 sig')
            schedule_full.to_csv('./results_output/all_schedules.csv', index=False, encoding='utf-8 sig')
            rules_full.to_csv('./results_output/schedule_rules.csv', index=False, encoding='utf-8 sig')
            schedules_most_reschedules_teams.to_csv('./results_output/teams_with_more_reschedules.csv', index=False,
                                                    encoding='utf-8 sig')

            execution_times.to_csv('./results_output/execution_times.csv', index=False, encoding='utf-8 sig')



