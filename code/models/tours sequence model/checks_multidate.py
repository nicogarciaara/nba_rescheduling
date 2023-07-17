import pandas as pd
from model_utils import League, Scheduler
import numpy as np
from ttp_model import TTPModel

if __name__ == '__main__':
	df = pd.read_csv("./output/BasicModel_nba_basic_mid_basic_post_all_star_5.csv")
	df['original_date'] = pd.to_datetime(df['original_date'])
	all_dates = list(pd.date_range(
		start=np.min(df['original_date']),
		end=np.max(df['original_date'])))

	L = League('nba')
	fixture = L.load_schedule()
	S = Scheduler('nba', fixture)
	windows = S.calculate_resched_windows()

	teams = df['home'].unique()

	df_multi = pd.DataFrame()

	for team in teams:
		aas = df['home'].unique()
		aas
		df_team = df[(df['home'] == team) | (df['visitor'] == team)]
		full_window = windows[team]
		no_dates = []
		for window in full_window:
			for e in window:
				no_dates.append(e)

		df_group = df_team.groupby('original_date').size().reset_index(name='n_games')
		df_group = df_group[df_group['n_games'] > 1]
		df_multi = pd.concat([df_multi, df_team], ignore_index=True)

		for i in range(len(all_dates) - 2):
			start = all_dates[i]
			end = all_dates[i + 2]
			end

			df_filt = df_team[(df_team['original_date'] >= start) & (df_team['original_date'] <= end)]
			print(len(df_filt))
			if df_filt.shape[0] == 3:
				df_filt
				df_multi = pd.concat([df_multi, df_filt], ignore_index=True)
	df_multi
	#df_multi.to_csv('./output/MULTICHECKS.csv', index=False, sep=';')