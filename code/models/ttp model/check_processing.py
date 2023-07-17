import pickle
import pandas as pd
import numpy as np
import datetime
from model_utils import League, Scheduler
from ttp_model import TTPModel

if __name__ == '__main__':
    starts = [datetime.datetime(2021, 3, 1)]

    for start in starts:
        with open(f'{start.date()}.pickle', 'rb') as handle:
            results = pickle.load(handle)

        # We evaluate each element of the dictionary
        fixture_old = results['fixture_old']
        fixture = results['fixture_new']
        disruptions = results['disruptions']
        non_disruptions = results['non_disruptions']
        x_var_dict = results['x_var_dict']
        x_variables = results['x_variables']
        variables_by_match = results['variables_by_match']

        fixture['original_date'] = pd.to_datetime(fixture['original_date'])
        all_dates = list(pd.date_range(
            start=np.min(fixture['original_date']),
            end=np.max(fixture['original_date'])))

        L = League('nba')
        original_fixture = L.load_schedule()
        S = Scheduler('nba', original_fixture)
        windows = S.calculate_resched_windows()

        teams = fixture['home'].unique()

        """
        df_multi = pd.DataFrame()

        for team in teams:
            df_team = fixture[(fixture['home'] == team) | (fixture['visitor'] == team)]

            for i in range(len(all_dates) - 2):
                start = all_dates[i]
                end = all_dates[i + 2]

                df_filt = df_team[(df_team['original_date'] >= start) & (df_team['original_date'] <= end)]
                if df_filt.shape[0] == 3:
                    df_multi = pd.concat([df_multi, df_filt], ignore_index=True)

        """
        vars_disruption = variables_by_match[('Toronto Raptors', 'Detroit Pistons', datetime.datetime(2021, 3, 2),
                                              datetime.datetime(2021, 3, 3))]
        vars_non_disruption_1 = variables_by_match[('Sacramento Kings', 'Detroit Pistons',
                                                    datetime.datetime(2021, 4, 8),
                                                    datetime.datetime(2021, 4, 8))]
        vars_match = []
        for nd in non_disruptions:
            if nd['game'] == ('Portland Trail Blazers', 'Detroit Pistons'):
                vars_match.append(nd)
        vars_non_disruption_2 = variables_by_match[('Portland Trail Blazers', 'Detroit Pistons',
                                                    datetime.datetime(2021, 4, 9),
                                                    datetime.datetime(2021, 4, 9))]

        vars_groups = vars_disruption + vars_non_disruption_1 + vars_non_disruption_2
        variables = []
        for var in vars_groups:
            idx = x_var_dict[var]
            if round(x_variables[idx]) == 1:
                variables.append(var)
        for var in variables:
            print(var)
        fixture_detroit = fixture[fixture['visitor'] == 'Detroit Pistons']
        fixture_detroit
