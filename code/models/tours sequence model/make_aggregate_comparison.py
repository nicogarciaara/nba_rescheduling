import pandas as pd
import datetime

if __name__ == '__main__':
    distance_full = pd.read_csv('./results_output/distance_analysis.csv')
    breaks_full = pd.read_csv('./results_output/breaks_analysis.csv')
    balance_full = pd.read_csv('./results_output/balance_analysis.csv')
    difference_full = pd.read_csv('./results_output/day_difference_analysis.csv')
    schedule_full = pd.read_csv('./results_output/all_schedules.csv')
    schedule_full['game_date'] = pd.to_datetime(schedule_full['game_date'])
    schedule_full
    schedules_most_reschedules_teams = pd.read_csv('./results_output/teams_with_more_reschedules.csv')
    schedules_most_reschedules_teams = schedules_most_reschedules_teams.drop_duplicates()
    schedules_most_reschedules_teams['game_date'] = pd.to_datetime(schedules_most_reschedules_teams['game_date'])

    # Calculate aggregates
    distance_agg = distance_full[distance_full['Team'] == 'all']
    distance_actual = distance_agg[(distance_agg['League'] == 'NBA') & (distance_agg['Schedule Type'] == 'Planned')].reset_index(drop=True)
    distance_ref = distance_actual['Distance'][0]
    distance_agg['Diff'] = distance_agg['Distance']/distance_ref - 1
    distance_agg[["obj", "distance_mode", "instance", "reschedule_mode", "n_window", "max_mods_per_tour", "feasibility_days", "asterisk", 'max_non_dis_mods', 'overlap_tours']] = distance_agg['Schedule Type'].str.split(' - ', expand=True)
    distance_agg.to_csv('./model_output/DistanceAnalysis.csv', index=False, encoding='utf-8 sig')

    breaks_agg = breaks_full[breaks_full['Team'] == 'all']
    breaks_agg[
        ["obj", "distance_mode", "instance", "reschedule_mode", "n_window", "max_mods_per_tour", "feasibility_days",
         "asterisk", 'max_non_dis_mods', 'overlap_tours']] = breaks_agg['Schedule Type'].str.split(' - ', expand=True)
    breaks_agg.to_csv('./model_output/BreaksAnalysis.csv', index=False, encoding='utf-8 sig')

    balance_agg = balance_full.groupby(['League', 'Schedule Type'])['Balance 7-day rolling mean'].sum().reset_index()
    difference_agg = difference_full.copy()
    difference_agg['Std'] = difference_agg['Day Difference'].copy()
    difference_agg = difference_agg.groupby(['League', 'Schedule Type']).agg({
        'Day Difference': 'mean',
        'Std': 'std'
    }).reset_index()
    schedule_full_may = schedule_full[schedule_full['game_date'] >= datetime.datetime(2021, 4, 1)]
    schedule_full_may_agg = schedule_full_may.groupby(['League', 'Schedule Type']).size().reset_index(name='n_games')

    schedule_full_plus = schedule_full[schedule_full['game_date'] >= datetime.datetime(2021, 5, 20)]
    schedule_full_plus['Count'] = 1
    schedule_full_plus_agg = schedule_full_plus.groupby(['League', 'Schedule Type']).agg({
        'Count': 'sum',
        'game_date': 'max'
    }).reset_index()

    schedules_most_reschedules_teams_may = schedules_most_reschedules_teams[schedules_most_reschedules_teams['game_date'] >= datetime.datetime(2021, 4, 1)]
    schedules_most_reschedules_teams_agg = schedules_most_reschedules_teams_may.groupby(['League', 'Schedule Type', 'Schedule Owner']).size().reset_index(name='n_games')

    schedule_full_plus = schedule_full.copy()
    schedule_full_plus['Count'] = 1
    schedule_full_plus['PlusLastDate'] = 0
    schedule_full_plus.loc[schedule_full_plus['game_date'] > datetime.datetime(2021, 5, 16), 'PlusLastDate'] = 1
    schedule_full_agg = schedule_full_plus.groupby(['League', 'Schedule Type']).agg({
        'Count': 'sum',
        'PlusLastDate': 'sum',
        'game_date': 'max'
    }).reset_index()
    schedule_full_agg['NewDays'] = (schedule_full_agg['game_date'] - datetime.datetime(2021, 5, 16)).dt.days

    schedule_different = schedule_full[schedule_full['game_date'] != schedule_full['original_date']]
    schedule_different = schedule_different[schedule_different['reschedule'] == 0]
    schedule_different = schedule_different.groupby('Schedule Type').size().reset_index(name='reschedule_nondis')
    schedule_full_agg = pd.merge(schedule_full_agg, schedule_different, how='left', on='Schedule Type')

    schedule_full_agg[
        ["obj", "distance_mode", "instance", "reschedule_mode", "n_window", "max_mods_per_tour", "feasibility_days",
         "asterisk", 'max_non_dis_mods', 'overlap_tours']] = schedule_full_agg['Schedule Type'].str.split(' - ', expand=True)

    schedule_full_agg.to_csv('./model_output/FullScheduleAnalysis.csv', index=False, encoding='utf-8 sig')