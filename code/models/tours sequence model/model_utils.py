import pandas as pd
import os
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')


class League:
    def __init__(self, league, custom_schedule=pd.DataFrame()):
        """
        Initiate the League class

        Parameters
        ----------
        league: str
            String indicating the league whose schedule we want to load. Must be one of the following:
                - 'nba'
                - 'nhl'
        custom_schedule: pd.DataFrame
            Dataframe containing the custom schedule that we want to use for our model

        """
        self.league = league
        self.custom_schedule = custom_schedule

    def load_schedule(self):
        """
        Loads a dataframe that has the played and the planned schedule and another one


        Returns
        -------
        df_fixture: pd.DataFrame
            Planned and played schedule
        """
        if len(self.custom_schedule) > 0:
            df_fixture = self.custom_schedule
        else:
            # 2e load the schedule
            schedule_file_dir = os.getcwd()
            if "models" in schedule_file_dir:
                schedule_file_dir = schedule_file_dir.replace('code\\models\\ttp model', f'data\\schedules\\{self.league}')
            else:
                schedule_file_dir = schedule_file_dir + f"/data\\schedules\\{self.league}"
            df_fixture = pd.read_csv(f'{schedule_file_dir}/{self.league}_original_and_true_schedule.csv')

        # Format date columns
        df_fixture['original_date'] = pd.to_datetime(df_fixture['original_date'])
        df_fixture['original_date'] = df_fixture['original_date'].fillna(df_fixture['game_date'])
        df_fixture['game_date'] = pd.to_datetime(df_fixture['game_date'])
        return df_fixture

    def load_rules(self):
        """
        Loads a dataframe that has the scheduling rules (how many games are played, at maximum, in a range of X days)

        Returns
        -------
        df_rules: pd.DataFrame
            Scheduling rules
        """

        # We load the schedule rules
        rules_file_dir = os.getcwd()
        if "models" in rules_file_dir:
            rules_file_dir = rules_file_dir.replace('models\\ttp model', f'eda\\results')
        else:
            rules_file_dir = rules_file_dir + "\\code\\eda\\results"
        df_rules = pd.read_csv(f'{rules_file_dir}/{self.league}_schedule_rules.csv')

        return df_rules

    def get_disruptions(self):
        """
        Calculate for every rescheduled match, if it was a disruption (a game is a disruption if in the new date, we
        change the order of that game, compared to the old date)

        Returns
        -------
        disruption_games: list
            Array whose elements are dictionaries with information about the rescheduled games.
            The dictionary that has the following structure
            {game: (home_team, away_team),
            original_date: original datetime,
            new_date: played datetime}
        non_disruption_games: list
            Array whose elements are dictionaries with information about the non-rescheduled games.
            The dictionary that has the following structure
            {game: (home_team, away_team),
            original_date: original datetime}

        """
        # Load schedule
        df_schedule = self.load_schedule()

        teams = list(df_schedule['home'].unique())

        # Create output list
        disruption_games = []
        non_disruption_games = []
        id_no_disruption = 0
        id_disruption = 0

        # The procedure will be the following:
        # - For each team, we check their games games
        # - For each rescheduled game we check the previous and next game, if it is the same it is not a disruption
        for team in teams:
            # First, we filter games of that particular team
            df_team = df_schedule[(df_schedule['home'] == team) | (df_schedule['visitor'] == team)].reset_index(drop=True)

            # We filter reschedules (only checking the ones that weren't rescheduled to a previous date)
            df_reschedules = df_team[((df_team['reschedule'] == 1) & (df_team['day_difference'] > 0))]
            df_reschedules['considered'] = 1

            df_team_no_reschedules = pd.merge(df_team, df_reschedules, how='left',
                                              on=['home', 'visitor', 'original_date', 'game_date'])
            df_team_no_reschedules['considered'] = df_team_no_reschedules['considered'].fillna(0)
            df_team_no_reschedules = df_team_no_reschedules[df_team_no_reschedules['considered'] == 0]
            if len(df_reschedules) + len(df_team_no_reschedules) != 72:
                df_team_no_reschedules

            for index, row in df_reschedules.iterrows():
                if row['home'] == team:
                    original_date = row['original_date']
                    new_date = row['game_date']

                    # We check the previous game
                    prev_games_old = df_team[df_team['original_date'] < original_date].sort_values(by='original_date',
                                                                                                   ascending=False).head(1)
                    prev_games_old = prev_games_old.reset_index(drop=True)
                    # We add a clause in case this is the first game of the season
                    if original_date != np.min(df_team['original_date']):
                        prev_date_old = prev_games_old['original_date'][0]
                    else:
                        prev_date_old = 'NAN'

                    # We check the previous game in the new schedule
                    prev_games_new = df_team[df_team['game_date'] < new_date].sort_values(by='game_date',
                                                                                          ascending=False).head(1)
                    prev_games_new = prev_games_new.reset_index(drop=True)
                    if new_date != np.min(df_team['game_date']):
                        prev_date_new = prev_games_new['game_date'][0]
                    else:
                        prev_date_new = 'NAN'

                    # We filter now the next game, first for the original schedule
                    next_games_old = df_team[df_team['original_date'] > original_date].sort_values(
                        by='original_date').head(1)
                    next_games_old = next_games_old.reset_index(drop=True)

                    if original_date != np.max(df_team['original_date']):
                        next_date_old = next_games_old['original_date'][0]
                    else:
                        next_date_old = 'NAN'

                    next_games_new = df_team[df_team['game_date'] > new_date].sort_values(by='game_date').head(1)
                    next_games_new = next_games_new.reset_index(drop=True)
                    if new_date != np.max(df_team['game_date']):
                        next_date_new = next_games_new['game_date'][0]
                    else:
                        next_date_new = 'NAN'

                    # We add a disruption if rivals change and dates change
                    if prev_date_old != prev_date_new and next_date_old != next_date_new:
                        disruption_games.append(
                            {'game': (team, row['visitor']),
                             'original_date': row['original_date'],
                             'proposed_date': row['original_date'],
                             'game_date': row['game_date'],
                             'id_match': id_disruption}
                        )
                        id_disruption += 1
                    else:
                        non_disruption_games.append(
                            {'game': (team, row['visitor']),
                             'original_date': row['original_date'],
                             'proposed_date': row['original_date'],
                             'id_match': id_no_disruption}
                        )
                        id_no_disruption += 1
            for index, row in df_team_no_reschedules.iterrows():
                if row['home'] == team:
                    original_date = row['original_date']
                    non_disruption_games.append(
                        {'game': (team, row['visitor']),
                         'original_date': row['original_date'],
                         'proposed_date': row['original_date'],
                         'id_match': id_no_disruption}
                    )
                    id_no_disruption += 1
        return disruption_games, non_disruption_games

    def get_available_dates(self):
        """
        We create a date range that has all dates between the first and last of the original schedule

        Returns
        -------
        league_dates: list
            Possible dates of the original schedule
        """
        df_schedule = self.load_schedule()
        league_dates = list(pd.date_range(np.min(df_schedule['original_date']), np.max(df_schedule['original_date'])))
        return league_dates

    def get_max_games_rules(self):
        """
        Creates a dictionary that saves how many games at max can be played in a span of time

        Returns
        -------
        max_games_dict: dict
            Information with maximum number of games in a span of time.
            Each item of a dictionary has the following structure:
            (home_away_condition, number_of_dates): number_of_games
        """
        # Load rules
        df_rules = self.load_rules()
        max_games_dict = {}

        for col_name in df_rules.columns:
            col_name_split = col_name.split('_')

            # As the name of the column is "Max_games_days_condition" (e.g. Max_games_1_home),
            # we use that to populate our dictionary
            if 'Max' in col_name_split:
                max_games_dict[(col_name_split[len(col_name_split) - 1],
                                int(col_name_split[len(col_name_split) - 2]))] = np.max(df_rules[col_name])

        return max_games_dict

    def get_back_to_back_rules(self):
        """
        Creates a dictionary that saves how many back to back games can be played with a particular home-away condition

        Returns
        -------
        back_to_backs_dict: dict
            Information with the maximum number of back to backs with a particular home away condition
            The dictionary has the following structure
            home_away_condition: number_of_back_to_backs
        """
        # Load rules
        df_rules = self.load_rules()
        back_to_backs_dict = {}

        for col_name in df_rules.columns:
            col_name_split = col_name.split('_')

            # As the column name is "Back2Backs_condition" (e.g. Back2Backs_home),
            # we use that to populate our dictionary
            if 'Back2Backs' in col_name_split:
                back_to_backs_dict[col_name_split[len(col_name_split) - 1]] = np.max(df_rules[col_name])
        return back_to_backs_dict

    def get_distance_matrix(self):
        """
        Generates a dictionary with the distance between two teams

        Returns
        -------
        dist_matrix: dict
            Distance matrix between teams. The dictionary has the following structure
            (team_a, team_b): distance_between_a_and_b
        """
        # As distances are saved in a dataframe, we load that first
        file_dir = os.getcwd()
        if "models" in file_dir:
            file_dir = file_dir.replace('code\\models\\ttp model', f'data\\teams\\{self.league}')
        else:
            file_dir = file_dir + f"\\data\\teams\\{self.league}"
        
        dist_matrix_df = pd.read_csv(f'{file_dir}\\{self.league}_distances_matrix.csv')

        # List of teams
        teams = list(dist_matrix_df['Equipo'])

        dist_matrix = {}

        # We populate the dictionary
        for team_i in teams:
            for j in range(len(dist_matrix_df)):
                team_j = dist_matrix_df['Equipo'][j]
                dist_matrix[(team_i, team_j)] = dist_matrix_df[team_i][j]
        return dist_matrix


class Scheduler:
    def __init__(self, league, custom_fixture=None):
        """
        Initializes the Scheduler class

        Parameters
        ----------
        league: str
            String indicating the league whose schedule we want to load. Must be one of the following:
                - 'nba'
                - 'nhl'
        custom_fixture (optional): pd.DataFrame
            If specified, we use a custom fixture for the schedule. This will be useful when we are building
            models iteratively
        """
        self.league = league
        L = League(league, custom_schedule=custom_fixture)
        self.df_fixture = custom_fixture
        """
        # We import things from the League class
        try:
            if custom_fixture.shape[0] > 0:
                L = League(league, custom_schedule=custom_fixture)
                self.df_fixture = custom_fixture
            else:
                L = League(league)
                self.df_fixture = L.load_schedule()
        except:
            L = League(league)
            self.df_fixture = L.load_schedule()
        """
        self.disruptions, self.non_disruptions = L.get_disruptions()
        self.league_dates = L.get_available_dates()
        self.max_games_rules = L.get_max_games_rules()
        self.back_to_back_rules = L.get_back_to_back_rules()
        self.dist_matrix = L.get_distance_matrix()
        self.teams = list(self.df_fixture['home'].unique())

    def get_tours_by_team(self):
        """
        Creates a dictionary that has, by team, a list of lists that has the dates of each tour
        A tour is a series of matches with the same home/away condition that have three days or less between games

        Returns
        -------
        tours_dict: dict
            Tours by team. The dictionary has the following structure
            team: [tour_1, tour_2, tour_3]
        """
        tours_dict = {}
        for team in self.teams:
            tours_dict[team] = []

            # We filter the games of this team
            team_games = self.df_fixture[((self.df_fixture['home'] == team) | (self.df_fixture['visitor'] == team))]
            team_games = team_games.sort_values(by='original_date').reset_index(drop=True)

            # We create a column that has the previous game date
            team_games['prev_date'] = team_games['original_date'].shift(1)
            team_games['diff'] = (team_games['original_date'] - team_games['prev_date']).dt.days
            team_games['diff'] = team_games['diff'].fillna(0)
            team_games = team_games.reset_index(drop=True)

            # Check the condition of the first game
            if team_games['home'][0] == team:
                prev_condition = 'H'
            else:
                prev_condition = 'A'

            tour_games = [{'game': (team_games['home'][0],
                                    team_games['visitor'][0]),
                           'original_date': team_games['original_date'][0],
                           'game_date': team_games['game_date'][0],
                           'proposed_date': team_games['original_date'][0],

                           }
                          ]

            for i in range(1, len(team_games)):
                # Check the condition of the game
                if team_games['home'][i] == team:
                    condition = 'H'
                else:
                    condition = 'A'
                # If it's the same condition, we add a new game to the tour
                if condition == prev_condition and team_games['diff'][i] < 4:
                    tour_games.append({
                        'game': (team_games['home'][i],
                                 team_games['visitor'][i]),
                        'original_date': team_games['original_date'][i],
                        'game_date': team_games['game_date'][i],
                        'proposed_date': team_games['original_date'][i],
                           })
                else:
                    # If not, we finish the tour and create a new one
                    if len(tour_games) > 0:
                        tours_dict[team].append(tour_games)
                    tour_games = [{
                        'game': (team_games['home'][i],
                                 team_games['visitor'][i]),
                        'original_date': team_games['original_date'][i],
                        'game_date': team_games['game_date'][i],
                        'proposed_date': team_games['original_date'][i]
                                   }]
                    prev_condition = condition
        return tours_dict
    
    def get_away_tours(self, tours_dict, end_date):
        """
        Filters the tours_dict dictionary to only include away tours that start after the date we are standing on

        Parameters
        ----------
        tours_dict (dict): Tours by team. The dictionary has the following structure
            team: [tour_1, tour_2, tour_3]
        end_date (str or datetime): Date when we do the rescheduling
        
        Returns
        -------
        away_future_tours_dict: Tours by team, filtered by date and away condition. The dictionary has the following structure
            team: [tour_1, tour_2, tour_3]
        """
        away_future_tours_dict = {}
        for team in self.teams:
            away_future_tours_dict[team] = []
            for tour in tours_dict[team]:
                # Check first game of the tour
                first_game = tour[0]
                if first_game['game'][0] == team:
                    condition = 'home'
                else:
                    condition = 'away'
                game_date = first_game['original_date']
                if condition == 'away' and game_date > end_date:
                    away_future_tours_dict[team].append(tour)
        return away_future_tours_dict

    def calculate_resched_windows(self):
        """
        Additionally to the hard rules a schedule has, we calculate sets of dates in which games will not be played due
        to reschedules. Therefore, if a rescheduled game is scheduled in a window of days where there was a COVID
        outbreak, a new reschedule should occur

        Returns
        -------
        resched_windows_dict: dict
            Information by team of the windows in which there were reschedules
        """
        resched_windows_dict = {}
        for team in self.teams:
            resched_windows_dict[team] = []

            team_games = self.df_fixture[((self.df_fixture['home'] == team) | (self.df_fixture['visitor'] == team))]
            #team_games.loc[team_games['day_difference'] == 1, 'reschedule'] = 0
            # We filter reschedules (only checking the ones that weren't rescheduled to a previous date)
            df_reschedules = team_games[((team_games['reschedule'] == 1) & (team_games['day_difference'] > 0))]

            for index, row in df_reschedules.iterrows():
                new_date = row['original_date']

                # We check the previous game of the reschedule
                prev_game = team_games[(team_games['game_date'] < new_date) & (
                        team_games['reschedule'] == 0)].sort_values(by='game_date', ascending=False).head(1)
                prev_game = prev_game.reset_index(drop=True)

                # We check the next game of the reschedule
                next_game = team_games[(team_games['game_date'] > new_date) & (
                        team_games['reschedule'] == 0)].sort_values(by='game_date').head(1)
                next_game = next_game.reset_index(drop=True)

                # Create the date range between both dates and append it
                if len(prev_game) > 0 and len(next_game) > 0:
                    window = list(pd.date_range(prev_game['game_date'][0] + datetime.timedelta(days=1),
                                                next_game['game_date'][0] - datetime.timedelta(days=1)))
                elif len(prev_game) > 0 and len(next_game) == 0:
                    window = list(pd.date_range(prev_game['game_date'][0] + datetime.timedelta(days=1),
                                                prev_game['game_date'][0] + datetime.timedelta(days=10)))
                elif len(prev_game) == 0 and len(next_game) > 0:
                    window = list(pd.date_range(next_game['game_date'][0] - datetime.timedelta(days=10),
                                                next_game['game_date'][0] - datetime.timedelta(days=1)))
                else:
                    window = []

                if window not in resched_windows_dict[team]:
                    resched_windows_dict[team].append(window)

        return resched_windows_dict

