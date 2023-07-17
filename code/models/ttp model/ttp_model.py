import pandas as pd
from model_utils import Scheduler
import datetime
import numpy as np
from tqdm import tqdm


class TTPModel:
    def __init__(self, league, custom_fixture=None, start_date=datetime.datetime(2021, 1, 1),
                 end_date=datetime.datetime(2021, 1, 31), distance_mode='mid', disruptions=[], non_disruptions=[],
                 max_mods_per_tour=100, max_adj_days=0, feasibility_days=10, max_non_dis_mods=0, overlap_tours=True):
        """
        Initiate basic model class

        Parameters
        ----------
        league: str
            String indicating the league whose schedule we want to load. Must be one of the following:
                - 'nba'
                - 'nhl'
        custom_fixture (optional): pd.DataFrame
            If specified, we use a custom fixture for the schedule. This will be useful when we are building
            models iteratively
        start_date: datetime.datetime
            Start datetime for the models that will be rescheduled
        end_date: datetime.datetime
            End datetime for the models that will be rescheduled
        distance_mode: str
            Indicates artificial modes that will be used to differentiate the distance tolerance.
            Must be one of the following:
                - low
                - mid
                - high
        disruptions: list
            If defined, it refers to an array whose elements are dictionaries with information about the
            rescheduled games.
            The dictionary that has the following structure
                {game: (home_team, away_team),
                original_date: original datetime,
                new_date: played datetime}
            If it is not defined, we calculate it
        non_disruptions: list
            If defined, it refers to an array whose elements are dictionaries with information about the
            non-rescheduled games.
            The dictionary that has the following structure
                {game: (home_team, away_team),
                original_date: original datetime,
                new_date: played datetime}
            If it is not defined, we calculate it
        max_mods_per_tour: int
            Max number of modified of non disruptions whose date can be modified per tour
        max_adj_days: int
            Maximum number of days that a non-disruption game can be changed
        feasibility_days: int
            Minimum number of days we ask for a particular candidate day to
        max_non_dis_mods: int
            Maximum number of non-disrupted matches whose date can be modified
        overlap_tours: bool
            Variable that indicates if we let tours overlap or not
        """
        self.league = league

        # Create other classes
        S = Scheduler(league, custom_fixture=custom_fixture)
        self.df_fixture = S.df_fixture
        self.disruptions = disruptions
        self.non_disruptions = non_disruptions
        self.league_dates = S.league_dates
        self.max_games_rules = S.max_games_rules
        self.back_to_back_rules = S.back_to_back_rules
        self.dist_matrix = S.dist_matrix
        self.teams = list(self.df_fixture['home'].unique())
        self.tours_dict = S.get_tours_by_team()
        self.resched_windows_dict = S.calculate_resched_windows()
        self.start_date = start_date
        self.end_date = end_date
        self.distance_mode = distance_mode
        self.max_mods_per_tour = max_mods_per_tour
        self.max_adj_days = max_adj_days
        self.feasibility_days = feasibility_days
        self.max_non_dis_mods = max_non_dis_mods
        self.overlap_tours = overlap_tours

        # Create a set of extended dates, that is equal to a date range that has the next 180 days after the end of the
        # original planned season
        self.extended_dates = list(pd.date_range(start=np.max(self.league_dates) + datetime.timedelta(days=1),
                                                 periods=180))

    def check_match_schedule_feasibility(self, team_games, potential_date):
        """
        For a potential new date for a match, this method checks if this date would break a particular
        scheduling rule

        Parameters
        ----------
        team_games: pd.DataFrame
            Schedule of a particular team
        potential_date
            Date we want to check if it would be a feasible one

        Returns
        -------
        valid_date: bool
            If True, it indicates that that this particular date is valid
        """
        # We initialize the output to True
        valid_date = True

        # We do for every range between 1 and 7 days
        for n_days in range(1, 8):
            # We check each interval in which the potential day is involved, starting when
            # potential day is the last day of the interval
            start = potential_date - datetime.timedelta(days=n_days-1)
            end = potential_date

            while start <= potential_date:
                filt_games = team_games[(team_games['original_date'] >= start) & (team_games['original_date'] <= end)]

                # If we already are having the maximum number of allowed games, then valid_date should be False
                if len(filt_games) >= self.max_games_rules[('all', n_days)]:
                    valid_date = False
                    return valid_date
                start = start + datetime.timedelta(days=1)
                end = end + datetime.timedelta(days=1)

        return valid_date


    def check_distance_feasibility(self, games_to_chack, margin=0.2):
        """
        For each disruption and each possible day for each team, we see if it is a desirable day to put the match in.
        Basically, we calculate the distance each team would incur if we put the rescheduled game there and see if
        that's acceptable (the distance would be acceptable if it is lower than the original distance multiplied by
        1 + margin

        Parameters
        ----------
        games_to_chack: list
            Games that should be considered to evaluate its distance feasibility
        margin: float
            Maximum acceptable percentage level of difference between the original distance traveled and the new one in
            the new model

        Returns
        -------
        match_distance_feasibility: dict
            Has information per match and team of the days in which it is reasonable to have a match.
            The dictionary has the following structure
            match: (team_1: [list_of_feasible_days], team_2: [list_of_feasible_days]
        """
        match_distance_feasibility = {}
        # For every disruption game
        for match in games_to_chack:
            home_team = match['game'][0]
            away_team = match['game'][1]

            # Create a team dictionary of stats
            team_stats = {'home': {'team': home_team}, 'away': {'team': away_team}}

            # First, we calculate the distance traveled by each team. The distance will be equal to
            # Distance between home team of the previous game and home team of this game +
            # Distance between home team of this game and the home team of the next game
            for team in team_stats:
                team_games = self.df_fixture[((self.df_fixture['home'] == team_stats[team]['team']) | (
                        self.df_fixture['visitor'] == team_stats[team]['team']))]

                # We see check the previous and the next game
                prev_game = team_games[team_games['original_date'] < match['original_date']].sort_values(
                    by='original_date', ascending=False).head(1).reset_index(drop=True)
                next_game = team_games[team_games['original_date'] > match['original_date']].sort_values(
                    by='original_date').head(1).reset_index(drop=True)
                if len(prev_game) > 0:
                    prev_home = prev_game['home'][0]
                else:
                    prev_home = team_stats[team]['team']

                if len(next_game) > 0:
                    next_home = next_game['home'][0]
                else:
                    next_home = team_stats[team]['team']
                distance = self.dist_matrix[(prev_home, home_team)] + self.dist_matrix[(home_team, next_home)]
                team_stats[team]['distance'] = distance

                # In order to avoid restricting too much the space when we have to reschedule a home game, we calculate
                # the closest distance between this team and another
                closest_distance = 1e10
                for team_pair in self.dist_matrix:
                    if team_stats[team]['team'] in team_pair and self.dist_matrix[team_pair] > 0:
                        if self.dist_matrix[team_pair] < closest_distance:
                            closest_distance = self.dist_matrix[team_pair]

                # Create a list where we will add feasible days
                possible_days = []

                # For each potential day, we calculate the distance that we would have
                for potential_day in self.league_dates:

                    if potential_day > self.end_date:
                        # Check potential previous and next game
                        pot_prev_game = team_games[team_games['original_date'] < potential_day].sort_values(
                            by='original_date', ascending=False).head(1).reset_index(drop=True)
                        pot_next_game = team_games[team_games['original_date'] > potential_day].sort_values(
                            by='original_date').head(1).reset_index(drop=True)
                        if len(pot_prev_game) > 0 and len(pot_next_game) > 0:

                            # Calculate distance in the same way
                            pot_prev_home = pot_prev_game['home'][0]
                            pot_next_home = pot_next_game['home'][0]
                            pot_distance = self.dist_matrix[(pot_prev_home, home_team)] + \
                                           self.dist_matrix[(home_team, pot_next_home)]
                            pot_distance_1 = np.min([self.dist_matrix[(pot_prev_home, home_team)],
                                                     self.dist_matrix[(home_team, pot_next_home)]])
                            pot_distance_2 = np.max([self.dist_matrix[(pot_prev_home, home_team)],
                                                     self.dist_matrix[(home_team, pot_next_home)]])

                            # If distance is reasonable, we add this to our list of potential dayss
                            if distance == 0:
                                reference = closest_distance
                            else:
                                reference = distance
                            if pot_distance <= reference * (1 + margin) or abs(pot_distance_2/pot_distance_1 - 1) <= margin:
                                if self.max_adj_days == -10:
                                    valid_date = self.check_match_schedule_feasibility(team_games, potential_day)
                                else:
                                    valid_date = True
                                if valid_date:
                                    possible_days.append(potential_day)

                if match['game_date'] not in possible_days:
                    possible_days

                if margin < 2500:
                    if len(possible_days) > self.feasibility_days:
                        match_distance_feasibility[(team_stats[team]['team'], match['original_date'],
                                                    match['game_date'])] = possible_days
                        if match['game_date'] not in possible_days:
                            possible_days
                    else:
                        match_distance_feasibility[(team_stats[team]['team'], match['original_date'],
                                                    match['game_date'])] = []
                else:
                    match_distance_feasibility[(team_stats[team]['team'], match['original_date'],
                                                match['game_date'])] = possible_days
                    if match['game_date'] not in possible_days:
                        possible_days
        match_distance_feasibility
        return match_distance_feasibility

    def add_variables_dict_according_to_distance_threshold(self, matches_to_schedule, match_distance_feasibility,
                                                           idx, x_var_dict, x_var_dict_inv, end_date):
        """
        Creation of variables according to the selected dictionary of distance feasibility

        Parameters
        ----------
        matches_to_schedule: list
            Matches that are going to be rescheduled
        match_distance_feasibility: dict
            Has information per match and team of the days in which it is reasonable to have a match.
            The dictionary has the following structure
            match: (team_1: [list_of_feasible_days], team_2: [list_of_feasible_days]
        idx: int
            Maximum index of the variables that were created
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        x_var_dict_inv: dict
            Inverse dictionary of decision variables that will be included in the model. Each item in the
            dictionary will have the following structure
            index: (home_team, away_team, original_date, game_date, proposed_date)
        end_date: datetime.datetime
            End date of the window of games that we want to reschedule

        Returns
        -------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        x_var_dict_inv: dict
            Inverse dictionary of decision variables that will be included in the model. Each item in the
            dictionary will have the following structure
            index: (home_team, away_team, original_date, game_date, proposed_date)
        non_matched_matches: list
            Matches that under the current distance feasibility restrictions weren't assigned any potential date
        idx: int
            Maximum index of the variables that were created

        """
        # We create a list of matches that weren't matched to any date
        non_matched_macthes = []

        # For every match we will do the following:
        # Check every date and see if
        #   The date is in both team available's date
        #   The date is greater than end_date
        for match in matches_to_schedule:
            home_team = match['game'][0]
            away_team = match['game'][1]

            # Check which date we will evaluate
            date_to_check = 'original_date'

            # We check the conditions
            check = 0
            for pot_date in self.league_dates:
                if pot_date > end_date and \
                        pot_date in match_distance_feasibility[(home_team,
                                                                match[date_to_check], match['game_date'])] and \
                        pot_date in match_distance_feasibility[(away_team, match[date_to_check], match['game_date'])]:

                    # If all conditions apply, we add the match to the variables dict
                    x_var_dict[(home_team, away_team, match[date_to_check], match['game_date'], pot_date)] = idx
                    x_var_dict_inv[idx] = (home_team, away_team, match[date_to_check], match['game_date'], pot_date)
                    idx += 1
                    check = 1
            if check == 0:
                non_matched_macthes.append(match)

        return x_var_dict, x_var_dict_inv, non_matched_macthes, idx

    def create_decision_variables_dict(self, start_date, end_date, objective, match_buffer=None, max_adj_days=1):
        """
        Creates a dictionary in which we save an index for each decision variable that we will include.
        As we will run this model monthly, we will try to imitate the real thing:
            - We will create variables for every suspended match between start_date and end_date

        This matches will include the ones whose original date was between start_date and end_date and any rescheduled
        match who was scheduled between a new COVID outbreak. In this sense, it would be as we are in end_date + 1

        Parameters
        ----------
        start_date: datetime.datetime
            Start date of the window of games that we want to reschedule
        end_date: datetime.datetime
            End date of the window of games that we want to reschedule
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'unitary': Will try to maximize the number of scheduled games
                - 'basic': Will try to minimize the difference between the original date and the new date
                - 'double': Will try to minimize the double of the difference between the original date and the new date
                - 'squared': Will try to minimize the square of the difference between the original and the new date
        match_buffer: list
            List of games that we want to reschedule again because the reschedule date was during a new COVID outbreak
        max_adj_days: int
            Maximum number of days that a non-disruption game can be changed

        Returns
        -------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index.
        matches_to_be_scheduled: list
            Matches that are rescheduled
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        non_matched_macthes5: list
            List with information of matches that we couldn't find any potential days
        """
        # First we will get the matches to be scheduled, checking if they are in the match buffer and
        # if their date was between the dates
        if match_buffer:
            matches_to_be_scheduled = match_buffer
        else:
            matches_to_be_scheduled = []
        for res in self.disruptions:
            if start_date <= res['original_date'] <= end_date and res not in matches_to_be_scheduled:
                matches_to_be_scheduled.append(res)


        # We calculate the available dates per team, with different level of preference
        if self.distance_mode == 'low':
            match_distance_feasibility_1 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0)
            lengths = []
            for match in match_distance_feasibility_1:
                lengths.append(len(match_distance_feasibility_1[match]))
            match_distance_feasibility_2 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.2)
            match_distance_feasibility_3 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.7)
        elif self.distance_mode == 'mid':
            match_distance_feasibility_1 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.2)
            match_distance_feasibility_2 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.7)
            match_distance_feasibility_3 = self.check_distance_feasibility(matches_to_be_scheduled, margin=1)
        else:
            match_distance_feasibility_1 = self.check_distance_feasibility(matches_to_be_scheduled, margin=1500)
            match_distance_feasibility_2 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2000)
            match_distance_feasibility_3 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2200)

        match_distance_feasibility_4 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2400)
        match_distance_feasibility_5 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2500)

        # For every match we will do the following:
        # Check every date and see if
        #   The date is in both team available's date
        #   The date is greater than end_date
        x_var_dict = {}
        x_var_dict_inv = {}
        idx = 0

        # We populate our dictionaries according to the information that we have
        x_var_dict, x_var_dict_inv, non_matched_macthes1, idx = self.add_variables_dict_according_to_distance_threshold(
            matches_to_be_scheduled, match_distance_feasibility_1, idx, x_var_dict, x_var_dict_inv, end_date
        )


        x_var_dict, x_var_dict_inv, non_matched_macthes2, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes1, match_distance_feasibility_2, idx, x_var_dict, x_var_dict_inv, end_date
        )

        x_var_dict, x_var_dict_inv, non_matched_macthes3, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes2, match_distance_feasibility_3, idx, x_var_dict, x_var_dict_inv, end_date
        )

        x_var_dict, x_var_dict_inv, non_matched_macthes4, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes3, match_distance_feasibility_4, idx, x_var_dict, x_var_dict_inv, end_date
        )

        x_var_dict, x_var_dict_inv, non_matched_macthes5, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes4, match_distance_feasibility_5, idx, x_var_dict, x_var_dict_inv, end_date
        )

        # Add variables of additional dates
        for match in matches_to_be_scheduled:
            home_team = match['game'][0]
            away_team = match['game'][1]
            for pot_date in self.extended_dates:
                x_var_dict[(home_team, away_team, match['original_date'], match['game_date'], pot_date)] = idx
                x_var_dict_inv[idx] = (home_team, away_team, match['original_date'], match['game_date'], pot_date)
                idx += 1
        # We allow non-disruption matches to be changed only max_adj days

        # We define the possible delta days of each non-disruption, ranging from -max_adj_days to max_adj_days
        if max_adj_days != 0:
            possible_changes = []
            delta_days = max_adj_days * (-1)
            #delta_days = 0
            while delta_days <= max_adj_days:
                possible_changes.append(delta_days)
                delta_days += 1
        else:
            possible_changes = [0]

        if len(self.non_disruptions) > 0:
            for match in self.non_disruptions:
                home_team = match['game'][0]
                away_team = match['game'][1]
                game_date = match['original_date']
                final_date = match['game_date']
                if game_date >= end_date + datetime.timedelta(days=1):
                    for n_days in possible_changes:
                        if (game_date + datetime.timedelta(days=n_days)) > end_date:
                            x_var_dict[(home_team, away_team, game_date, final_date,
                                        game_date + datetime.timedelta(days=n_days))] = idx
                            x_var_dict_inv[idx] = (home_team, away_team, game_date, final_date,
                                                   game_date + datetime.timedelta(days=n_days))
                            idx += 1

        # We add the indeces if we use the objective function in which we are trying to balance the number of games
        # per team
        diff_games_dict = {}
        if objective == 'balanced':
            for day in self.league_dates + self.extended_dates:
                diff_games_dict[day] = idx
                idx += 1

        return x_var_dict, matches_to_be_scheduled, diff_games_dict, non_matched_macthes5

    def create_non_disruption_games_by_team(self):
        """
        We create a dictionary that has, by team, a dataframe with the information of the non-disruption games

        Returns
        -------
        non_dis_by_team_dict: dict
            Dictionary that has each team as a key and a dataframe of its non disruption games as a value
        """
        non_dis_by_team_dict = {}
        for team in self.teams:
            homes = []
            visitors = []
            original_dates = []
            game_dates = []

            # For each non disruption, we check if that team is playing this disruption
            for match in self.non_disruptions:
                home_team = match['game'][0]
                away_team = match['game'][1]
                game_date = match['original_date']
                final_date = match['game_date']

                # If the team plays and we are checking at a relevant date, we add the information to our lists
                if home_team == team or away_team == team:
                    if game_date >= self.end_date + datetime.timedelta(days=1):
                        homes.append(home_team)
                        visitors.append(away_team)
                        original_dates.append(game_date)
                        game_dates.append(final_date)
            # We create the dataframe
            games_df = pd.DataFrame(data={
                'home': homes,
                'visitor': visitors,
                'original_date': original_dates,
                'game_date': game_dates
            })
            # We sort the dataframe
            games_df = games_df.sort_values(by='original_date').reset_index(drop=True)
            non_dis_by_team_dict[team] = games_df
        return non_dis_by_team_dict

    def get_variables_by_team(self, x_var_dict):
        """
        Creates a dictionary that has by team, the keys of x_var_dict that refer to a team's games.

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index.

        Returns
        -------
        team_var_dict: dict
            Dictionary with the decision variables keys that relate to a particular team. Each item in the dictionary
            will have the following structure:
            team: [list of x_var_dict.keys() related to team]
        """
        team_var_dict = {}

        for team in self.teams:
            team_var_dict[team] = []

            # We populate the dictionary
            for x in x_var_dict:
                # Remember the structure of each key in x_var_dict:
                # (home_team, away_team, original_date, game_date, proposed_date)

                # We check if the variable has the team
                if x[0] == team or x[1] == team:
                    team_var_dict[team].append(x)
        return team_var_dict

    def get_variables_by_match(self, x_var_dict):
        """
        Creates a dictionary that has by match, a list of the variables associated with it. We will identify each match
        by the tuple (home_team, away_team, original_time, game_time)

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index.

        Returns
        -------
        game_var_dict: dict
            Dictionary that associates an individual match with the match assignment variables that relate to it.
            The dictionary will have the following structure:
            (home_team, away_team, original_date, game_date): [list of x_var_dict.keys() related to match]
        """

        game_var_dict = {}
        # First, we do this for disruptions
        for match in self.disruptions:
            # Check the attributes of each disruption
            home_team_match = match['game'][0]
            away_team_match = match['game'][1]
            original_date_match = match['original_date']
            game_date_match = match['game_date']

            # Create a list for these attributes on the output dictionary
            game_var_dict[(home_team_match, away_team_match, original_date_match, game_date_match)] = []

            # For every match assignment variable, we check its attributes and compare them to the match we are checking
            for x in x_var_dict:
                home = x[0]
                away = x[1]
                o_date = x[2]
                g_date = x[3]
                p_date = x[4]

                # If the attributes are the same, then we add the variable to the list associated with this match
                if home_team_match == home and away_team_match == away and original_date_match == o_date and \
                    game_date_match == g_date:
                    game_var_dict[(home_team_match, away_team_match, original_date_match,
                                   game_date_match)].append(
                        (home, away, o_date, g_date, p_date)
                    )

        # We repeat the process for non-disruptions
        for match in self.non_disruptions:
            # Check the attributes of each match
            home_team_match = match['game'][0]
            away_team_match = match['game'][1]
            original_date_match = match['original_date']
            game_date_match = match['game_date']

            # Create a list for these attributes on the output dictionary
            game_var_dict[(home_team_match, away_team_match, original_date_match, game_date_match)] = []

            # For every match assignment variable, we check its attributes and compare them to the match we are checking
            for x in x_var_dict:
                home = x[0]
                away = x[1]
                o_date = x[2]
                g_date = x[3]
                p_date = x[4]

                # If the attributes are the same, then we add the variable to the list associated with this match
                if home_team_match == home and away_team_match == away and original_date_match == o_date and \
                    game_date_match == g_date:
                    game_var_dict[(home_team_match, away_team_match, original_date_match,
                                   game_date_match)].append(
                        (home, away, o_date, g_date, p_date)
                    )
        return game_var_dict

    def add_schedule_rules_constraints_home(self, x_var_dict, prob_lp, n_days):
        """
        Adds a set of constraint that limits the number of games in a particular set of days. For example, for each set
        of consecutive days, we can't have more than two games. A constraint will be created per team, days and number
        of days. For example this constraint

        sum_{i} x_it + sum_{i} x_it+1 <= 2 \foreach t, i \in GamesOfTeamA

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        n_days: int
            The number of days that will be considered for a particular constraint

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Create a list of possible dates
        possible_dates = self.league_dates + self.extended_dates

        for team in tqdm(self.teams):
            filt_games = self.df_fixture[self.df_fixture['home'] == team]

            # We build a constraint per team and day-window
            for i in range(len(possible_dates) - n_days + 1):
                initial_day = possible_dates[i]

                if initial_day >= (self.start_date - datetime.timedelta(days=7)):

                    # We calculate the number of games that are already played on this window in order to substract them
                    # from the right hand side. For example, if only two matches can be played in a span of three days and
                    # already there is a fixed game, then from our options, we can only add one additional game, not two
                    start = initial_day
                    end = possible_dates[i + n_days - 1]

                    filt_days = filt_games[(filt_games['original_date'] >= start) & (filt_games['original_date'] <= end)]
                    filt_days = filt_days[filt_days['original_date'] <= self.end_date]
                    n_games = len(filt_days)

                    ind = []
                    val = []
                    # For each variable, we check if the team we are checking is at home and the potential date is the one
                    # we are seeing
                    for var in x_var_dict:
                        if team == var[0] and initial_day == var[4]:
                            ind.append(x_var_dict[var])
                            val.append(1)

                    # We check now for the rest of the days
                    for n in range(1, n_days):
                        new_day = possible_dates[i + n]

                        # We check for any potential reschedule if this team is in that reschedule
                        for var in x_var_dict:
                            if team == var[0] and new_day == var[4]:
                                ind.append(x_var_dict[var])
                                val.append(1)

                    # We check if we have variables in order to add our constraint
                    if len(ind) > 0:
                        row = [ind, val]

                        # We add the constraint, checking the number of played games and the maximum allowed
                        prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'],
                                                       rhs=[self.max_games_rules[('home', n_days)] - n_games])

        return prob_lp

    def add_schedule_rules_constraints_away(self, x_var_dict, prob_lp, n_days):
        """
        Adds a set of constraint that limits the number of games in a particular set of days. For example, for each set
        of consecutive days, we can't have more than two games. A constraint will be created per team, days and number
        of days. For example this constraint

        sum_{i} x_it + sum_{i} x_it+1 <= 2 \foreach t, i \in GamesOfTeamA

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        n_days: int
            The number of days that will be considered for a particular constraint

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Create a list of possible dates
        possible_dates = self.league_dates + self.extended_dates

        for team in tqdm(self.teams):
            filt_games = self.df_fixture[self.df_fixture['visitor'] == team]
            # We build a constraint per team and day-window
            for i in range(len(possible_dates) - n_days + 1):
                initial_day = possible_dates[i]

                if initial_day >= (self.start_date - datetime.timedelta(days=7)):

                    # We calculate the number of games that are already played on this window in order to substract them
                    # from the right hand side. For example, if only two matches can be played in a span of three days and
                    # already there is a fixed game, then from our options, we can only add one additional game, not two
                    start = initial_day
                    end = possible_dates[i + n_days - 1]

                    filt_days = filt_games[(filt_games['original_date'] >= start) & (filt_games['original_date'] <= end)]
                    filt_days = filt_days[filt_days['original_date'] <= self.end_date]
                    n_games = len(filt_days)

                    ind = []
                    x_ind = []
                    val = []
                    # For each variable, we check if the team we are checking is at home and the potential date is the one
                    # we are seeing
                    for var in x_var_dict:
                        if team == var[1] and initial_day == var[4]:
                            ind.append(x_var_dict[var])
                            x_ind.append(x_var_dict[var])
                            val.append(1)

                    # We check now for the rest of the days
                    for n in range(1, n_days):
                        new_day = possible_dates[i + n]

                        # We check for any potential reschedule if this team is in that reschedule
                        for var in x_var_dict:
                            if team == var[1] and new_day == var[4]:
                                ind.append(x_var_dict[var])
                                val.append(1)

                    # We check if we have variables in order to add our constraint
                    if len(ind) > 0:
                        row = [ind, val]

                        # We add the constraint, checking the number of played games and the maximum allowed
                        prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'],
                                                       rhs=[self.max_games_rules[('away', n_days)] - n_games])

        return prob_lp

    def add_schedule_rules_constraints_all(self, x_var_dict, prob_lp, n_days):
        """
        Adds a set of constraint that limits the number of games in a particular set of days. For example, for each set
        of consecutive days, we can't have more than two games. A constraint will be created per team, days and number
        of days. For example this constraint

        sum_{i} x_it + sum_{i} x_it+1 <= 2 \foreach t, i \in GamesOfTeamA

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        n_days: int
            The number of days that will be considered for a particular constraint

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Create a list of possible dates
        possible_dates = self.league_dates + self.extended_dates

        for team in tqdm(self.teams):
            filt_games = self.df_fixture[((self.df_fixture['home'] == team) | (
                    self.df_fixture['visitor'] == team))]

            # We build a constraint per team and day-window
            for i in range(len(possible_dates) - n_days + 1):
                initial_day = possible_dates[i]
                games = []
                if initial_day >= (self.start_date - datetime.timedelta(days=7)):

                    # We calculate the number of games that are already played on this window in order to substract them
                    # from the right hand side. For example, if only two matches can be played in a span of three days and
                    # already there is a fixed game, then from our options, we can only add one additional game, not two
                    start = initial_day
                    end = possible_dates[i + n_days - 1]

                    filt_days = filt_games[(filt_games['original_date'] >= start) & (filt_games['original_date'] <= end)]
                    filt_days = filt_days[filt_days['original_date'] <= self.end_date]
                    n_games = len(filt_days)
                    ind = []
                    val = []
                    # For each variable, we check if the team we are checking is at home and the potential date is the one
                    # we are seeing
                    for var in x_var_dict:
                        if team == var[0] or team == var[1]:
                            if initial_day == var[4]:
                                ind.append(x_var_dict[var])
                                val.append(1)
                                games.append(var)

                    # We check now for the rest of the days
                    for n in range(1, n_days):
                        new_day = possible_dates[i + n]

                        # We check for any potential reschedule if this team is in that reschedule
                        for var in x_var_dict:
                            if team == var[0] or team == var[1]:
                                if new_day == var[4]:
                                    ind.append(x_var_dict[var])
                                    val.append(1)
                                    games.append(var)

                        if n_games > 0 and len(ind) > 0:
                            bound = self.max_games_rules[('all', n_days)]
                            n_games

                    # We check if we have variables in order to add our constraint
                    if len(ind) > 0:
                        row = [ind, val]

                        # We add the constraint, checking the number of played games and the maximum allowed
                        prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'],
                                                       rhs=[self.max_games_rules[('all', n_days)] - n_games])

        return prob_lp

    def one_match_per_day(self, x_var_dict, prob_lp):
        """
        Function that creates constraints that limit the number of games that a team can play on a single day to 1-
        If A is a set that contains the variables of a team, then

        sum(i \\in A) x_it <= 1 \foreach i, t

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Create a list of possible dates
        possible_dates = self.league_dates + self.extended_dates
        for team in self.teams:
            for day in possible_dates:
                # Create a constraint per team and day
                if day <= self.end_date:
                    filt_games = self.df_fixture[((self.df_fixture['home'] == team) | (
                            self.df_fixture['visitor'] == team))]
                    filt_games = filt_games[filt_games['original_date'] == day]
                    bound = len(filt_games)
                else:
                    bound = 0
                ind = []
                val = []
                # We populate the lists evaluating each variable
                for x in x_var_dict:
                    if x[0] == team or x[1] == team:
                        if x[4] == day:
                            ind.append(x_var_dict[x])
                            val.append(1)
                if len(ind) > 0:
                    row = [ind, val]
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[1 - bound])
        return prob_lp

    def each_match_is_played_once(self, x_var_dict, prob_lp):
        """
        Function that create constraints that force each games to be played exactly once.
        This can be expressed mathematically in the following way:

        sum(t) x_it = 1 \foreach i

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # We calculate the number of variables per team
        game_var_dict = self.get_variables_by_match(x_var_dict)

        for game in game_var_dict:
            # We create a constraint per match
            ind = []
            val = []
            # We add each associated variable
            for var in game_var_dict[game]:
                ind.append(x_var_dict[var])
                val.append(1)
            if len(ind) > 0:
                row = [ind, val]
                prob_lp.linear_constraints.add(lin_expr=[row], senses=['E'], rhs=[1])
        return prob_lp

    def no_games_on_prohibited_dates(self, x_var_dict, prob_lp):
        """
        Constraint that forces to have no games on dates when there are no games

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # List of dates that shouldn't have games
        dates_without_matches = [datetime.datetime(2020, 12, 24), datetime.datetime(2021, 3, 5),
                                 datetime.datetime(2021, 3, 6), datetime.datetime(2021, 3, 7),
                                 datetime.datetime(2021, 3, 8), datetime.datetime(2021, 3, 9)]
        ind = []
        val = []
        # We check each variable and see if we should add it
        for x in x_var_dict:
            if x[4] in dates_without_matches:
                ind.append(x_var_dict[x])
                val.append(1)
        if len(ind) > 0:
            row = [ind, val]
            prob_lp.linear_constraints.add(lin_expr=[row], senses=['E'], rhs=[0])
        return prob_lp

    def add_max_mods_per_tour_constraint(self, x_var_dict, prob_lp):
        """
        Adds a set of constraints that limits the number of non_disruption date modifications
        per tour to max_mods_per_tour

        If we have the set W of non-disruption variables of a tour whose proposed_date is different
        to original_date, we can define the constraint mathematically

        sum{i, t \\in W} x_it \leq 2

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        game_var_dict = self.get_variables_by_match(x_var_dict)

        # We evaluate each team and tour, creating one constraint per team and tour
        for team in self.teams:
            for tour in self.tours_dict[team]:
                ind = []
                val = []
                # For each match in the tour, we evaluate it if it is a non_disruption
                for match in tour:
                    if match in self.non_disruptions:
                        # For every variable, we check the ones where the proposed date is different to the original one
                        match_vars = game_var_dict[(match['game'][0], match['game'][1],
                                                    match['original_date'], match['game_date'])]
                        for mvar in match_vars:
                            # If the dates are different, we add the variable to our constraint
                            if mvar[4] != mvar[2]:
                                ind.append(x_var_dict[mvar])
                                val.append(1)
                if len(ind) > 1:
                    row = [ind, val]
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[self.max_mods_per_tour])
        return prob_lp

    def add_dont_overlap_tours(self, x_var_dict, prob_lp):
        """
        Adds a set of constraints that doesnt allow tours to overlap

        If we have a match i from a tour which originally starts at time t, then a match j of another tour
        that would start afterthat cannot be played before match i

        x_it + x_it-d <= 1

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        game_var_dict = self.get_variables_by_match(x_var_dict)

        for team in self.teams:
            # We check each pair of tours and see which is the one that starts before
            for tour_i in self.tours_dict[team]:
                for tour_j in self.tours_dict[team]:
                    # Tour i is the first tour
                    first_date_i = tour_i[0]['original_date']
                    first_date_j = tour_j[0]['original_date']

                    if first_date_i < first_date_j:
                        # For each pair of matches
                        for match_i in tour_i:
                            for match_j in tour_j:
                                if match_i in self.non_disruptions and match_j in self.non_disruptions:
                                    # We first check the day difference between both matches
                                    diff = abs(match_i['original_date']-match_j['original_date']).days

                                    # They will have a chance to overlap if this difference is lower than 2 times
                                    # n_window
                                    if diff <= 10000:
                                        # We check the variables of each match
                                        match_vars_i = game_var_dict[(match_i['game'][0], match_i['game'][1],
                                                                    match_i['original_date'], match_i['game_date'])]
                                        match_vars_j = game_var_dict[(match_j['game'][0], match_j['game'][1],
                                                                      match_j['original_date'], match_j['game_date'])]
                                        for var_i in match_vars_i:
                                            for var_j in match_vars_j:
                                                # (home_team, away_team, match[date_to_check], match['game_date'], pot_date)
                                                # If the second game is supposed to be played before, then we change this
                                                if var_j[4] < var_i[4]:
                                                    ind = [
                                                        x_var_dict[var_i],
                                                        x_var_dict[var_j]
                                                    ]
                                                    val = [1, 1]
                                                    row = [ind, val]
                                                    prob_lp.linear_constraints.add(lin_expr=[row],
                                                                                   senses=['L'], rhs=[1])

        return prob_lp


    def add_balanced_objective_function_constraint(self, x_var_dict, prob_lp, diff_games_dict):
        """
        We create a constraint that relates the objective function that balances the number of games played by each
        team. If set A has the set of games of team te

        This constraint will be equal to
        Dt <= sum{t<t} xbar_it + x_it \for each te \in teams, te \in teams

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        all_dates = list(self.league_dates) + list(self.extended_dates)
        day_counter = 1
        for day in all_dates:
            day_counter += 1
            if day > self.end_date:
                for team in self.teams:
                    # Create lists of indices and values
                    ind = []
                    val = []
                    team_games = self.df_fixture[((self.df_fixture['home'] == team) | (
                            self.df_fixture['visitor'] == team))]

                    # We count the games that have been played until today
                    games_played = team_games[team_games['original_date'] <= self.end_date]
                    n_games_played = len(games_played)

                    # We add the matches that we can reschedule that are prior to the date that we are looking
                    for var in x_var_dict:
                        # We consider this match if the propsed date is prior or equal to the date we are looking
                        if var[4] <= day.date():
                            if var[0] == team or var[1] == team:
                                ind.append(x_var_dict[var])
                                val.append(1)

                    # Additionally, we add the variable corresponding to the day we are looking
                    ind.append(diff_games_dict[day])
                    val.append(-1)

                    # If we have "x" variables, we add the constraint
                    if len(ind) > 0:
                        row = [ind, val]
                        prob_lp.linear_constraints.add(lin_expr=[row], senses=['G'],
                                                       rhs=[-n_games_played])

        return prob_lp

    def add_one_match_per_window(self, x_var_dict, prob_lp):
        """
        To control distance, we consider that no more than one game should be scheduled in a window (considering a
        window to be a set of consecutive of dates that a team has available for games). Considering a team's window W,
        this would be represented in the following way

        sum(i, t in W) x_it <= 1, \foreach W

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        non_dis_by_team_dict = self.create_non_disruption_games_by_team()
        team_var_dict = self.get_variables_by_team(x_var_dict)

        for team in self.teams:
            teams_df = non_dis_by_team_dict[team]
            team_vars = team_var_dict[team]

            # For every pair of consecutive dates in this df
            for i in range(len(teams_df) - 1):
                ind = []
                val = []
                dates = []
                start_date = teams_df['original_date'][i]
                end_date = teams_df['original_date'][i+1]
                # For every variable, if it is between these dates, we add it
                for variable in team_vars:
                    if start_date < variable[4] < end_date:
                        ind.append(x_var_dict[variable])
                        val.append(1)
                        dates.append(variable[4])
                if len(ind) > 1:
                    row = [ind, val]
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[1])
        return prob_lp

    def max_number_of_modifications(self, x_var_dict, prob_lp):
        """
        Sets the maximum number of non-disruptions whose dates are modified

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        game_var_dict = self.get_variables_by_match(x_var_dict)

        # We check each variable of non disruptions if their new date is different to the original one
        ind = []
        val = []
        for match in self.non_disruptions:
            # For every variable, we check the ones where the proposed date is different to the original one
            match_vars = game_var_dict[(match['game'][0], match['game'][1],
                                        match['original_date'], match['game_date'])]
            for mvar in match_vars:
                # If the dates are different, we add the variable to our constraint
                if mvar[4] != mvar[2]:
                    ind.append(x_var_dict[mvar])
                    val.append(1)
        if len(ind) > 1:
            row = [ind, val]
            prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[self.max_non_dis_mods])
        return prob_lp

    def add_constraint_matrix(self, x_var_dict, diff_games_dict, prob_lp, objective):
        """
        Adds constraint matrix to the problem, calling all the different methods

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        prob_lp: cplex.Cplex
            Cplex problem
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'basic': Will try to maximize the number of scheduled games
                - 'min_diff': Will try to minimize the difference between the original date and the new date
                - 'balanced': We try to make the daily match assignment balanced: trying to minimize the number of games
                              is played by the team that has played the most games

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Add every created constraint
        if not self.overlap_tours:
            prob_lp = self.add_dont_overlap_tours(x_var_dict, prob_lp)
        if objective == 'balanced':
            prob_lp = self.add_balanced_objective_function_constraint(x_var_dict, prob_lp, diff_games_dict)

        for n_days in list(range(1, 5)):
            #prob_lp = self.add_schedule_rules_constraints_home(x_var_dict, prob_lp, n_days)
            #prob_lp = self.add_schedule_rules_constraints_away(x_var_dict, prob_lp, n_days)
            prob_lp = self.add_schedule_rules_constraints_all(x_var_dict, prob_lp, n_days)

        #prob_lp = self.one_match_per_day(x_var_dict, prob_lp)
        prob_lp = self.each_match_is_played_once(x_var_dict, prob_lp)
        prob_lp = self.no_games_on_prohibited_dates(x_var_dict, prob_lp)
        prob_lp = self.add_max_mods_per_tour_constraint(x_var_dict, prob_lp)
        prob_lp = self.max_number_of_modifications(x_var_dict, prob_lp)

        if self.max_adj_days == -1:
            prob_lp = self.add_one_match_per_window(x_var_dict, prob_lp)

        return prob_lp

    def populate_by_row(self, x_var_dict, diff_games_dict, prob_lp, objective='ttp'):
        """
        Function that generates the model, creating the objective function and the needed constraints

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        prob_lp: cplex.Cplex
            Cplex problem
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'unitary': Will try to maximize the number of scheduled games
                - 'basic': Will try to minimize the difference between the original date and the new date
                - 'double': Will try to minimize the double of the difference between the original date and the new date
                - 'squared': Will try to minimize the square of the difference between the original and the new date

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Here the objective funcion will be equal to
        # min sum_{diff_it*x_it}.
        # The only thing that changes here is the objective function coefficient diff_it, that is equal to the
        # difference between the potential date and the original date
        coef = []
        lower_bounds = []
        upper_bounds = []
        types = []
        names = []
        for var in x_var_dict:
            # As the key of each item of x_var_dict is equal to a tuple with the following information
            # (home_team, away_team, original_date, played_date, proposed_date), this coefficient will be equal to
            # proposed_date - original_date, i.e. the fifth element of the tuple minus the third
            if objective == 'basic':
                if var[4] in self.extended_dates:
                    coef.append(100*abs(var[4]-var[2]).days)
                    #coef.append(100 * abs(var[4] - datetime.datetime(2020, 1, 1)).days)
                else:
                    coef.append(1 * abs(var[4] - var[2]).days)
                    #coef.append(1 * abs(var[4] - datetime.datetime(2020, 1, 1)).days)
            elif objective == 'min_date':
                if var[4] in self.extended_dates:
                    coef.append(100 * abs(var[4] - datetime.datetime(2020, 1, 1)).days)
                else:
                    coef.append(1 * abs(var[4] - datetime.datetime(2020, 1, 1)).days)

            elif objective == 'double':
                coef.append(2*abs(var[4] - var[2]).days)
            elif objective == 'squared':
                coef.append(abs(var[4] - var[2]).days ** 2)
            elif objective == 'unitary':
                if var[4] in self.extended_dates:
                    coef.append(abs(var[4] - var[2]).days)
                else:
                    coef.append(1)
            elif objective == 'balanced':
                coef.append(0)
            lower_bounds.append(0)
            upper_bounds.append(1)
            types.append('B')
            names.append(f'{var[0]}_{var[1]}_{var[2]}_{var[3]}_{var[4]}')

        if objective == 'balanced':
            for var in diff_games_dict:
                if var not in self.league_dates:
                    coef.append(1)
                else:
                    coef.append(10)
                lower_bounds.append(0)
                upper_bounds.append(100)
                types.append('I')
                names.append(str(var))

        # Add the variables to the problem and set the problem sense
        prob_lp.variables.add(obj=coef, lb=lower_bounds, ub=upper_bounds, types=types, names=names)
        if objective != 'balanced':
            prob_lp.objective.set_sense(prob_lp.objective.sense.minimize)
        else:
            prob_lp.objective.set_sense(prob_lp.objective.sense.maximize)

        prob_lp = self.add_constraint_matrix(x_var_dict, diff_games_dict, prob_lp, objective)
        prob_lp.write(f"RescheduleFixture_{objective}.lp")

        return prob_lp

    def solve_lp(self, x_var_dict, diff_games_dict, prob_lp, objective, mip_gap=None):
        """
        Creates and solves the linear programming problem

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        prob_lp: cplex.Cplex
            Cplex problem
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'unitary': Will try to maximize the number of scheduled games
                - 'basic': Will try to minimize the difference between the original date and the new date
                - 'double': Will try to minimize the double of the difference between the original date and the new date
                - 'squared': Will try to minimize the square of the difference between the original and the new date
        mip_gap: float
            Maximum gap accepted by the model

        Returns
        -------
        output_df: pd.DataFrame
            Information of the new dates of each match
        """
        # Create the problem
        prob_lp = self.populate_by_row(x_var_dict, diff_games_dict, prob_lp, objective)

        # Solve the problem
        #prob_lp.parameters.timelimit.set(30)
        #if mip_gap:
            #prob_lp.parameters.mip.tolerances.mipgap.set(mip_gap)
        #prob_lp.parameters.mip.tolerances.mipgap.set(0.05)
        # Setamos un limite en memoria del modelo
        #prob_lp.parameters.mip.limits.treememory.set(14000
        prob_lp.solve()

        # Get the solution variables
        x_variables = prob_lp.solution.get_values()

        # We'll create an output dataframe that has the proposed dates. We create lists where will we save each
        # proposed dates
        homes = []
        visitors = []
        original_dates = []
        game_dates = []
        proposed_dates = []
        reschedule = []

        variables = []

        # We check each variable to see its results
        for var in x_var_dict:
            if round(x_variables[x_var_dict[var]]) == 1:
                homes.append(var[0])
                visitors.append(var[1])
                original_dates.append(var[2])
                game_dates.append(var[3])
                proposed_dates.append(var[4])
                reschedule.append(1)

                if 'Detroit Pistons' == var[1]:
                    variables.append(var)

        # Create output dataframe
        output_df = pd.DataFrame({
            'home': homes,
            'visitor': visitors,
            'original_date': original_dates,
            'game_date': game_dates,
            'proposed_date': proposed_dates,
            'model_reschedule': reschedule,
        })
        return output_df, x_variables

    def calculate_needed_reschedules(self, output_df):
        """
        Calculates the matches that need to be rescheduled in the future because during the proposed reschedule date,
        some team suffers a COVID outbreak

        Parameters
        ----------
        output_df: pd.DataFrame
            Information of the new dates of each match

        Returns
        -------
        new_reschedules_list: list
            List with information of the matches that will need to be rescheduled again
        """
        new_reschedules_list = []

        for index, row in output_df.iterrows():
            # Create a variable that will indicate if we need a new reschedule
            check = 1
            home = row['home']
            visitor = row['visitor']
            # For both teams
            for team in [home, visitor]:
                # We check the "COVID" windows
                resched_windows = self.resched_windows_dict[team]

                # If the proposed date is in any of the windows, we add this match to the list of games that need to be
                # rescheduled
                for window in resched_windows:
                    if row['proposed_date'] in window and row['proposed_date'] != row['game_date']:
                        check = 0

            if check == 0:
                match_info = {
                    'game': (row['home'], row['visitor']),
                    'original_date': row['proposed_date'],
                    'game_date': row['game_date']
                }
                new_reschedules_list.append(match_info)
        return new_reschedules_list

    def update_matches_dictionaries(self, x_var_dict, x_variables):
        """
        Considering that the disruptions and non disruptions dictionaries originally are calculated without evaluating
        the model's output, we update the related date of each game

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        x_variables: list
            Decision variables output of the model
        Returns
        -------
        disruptions: list
            Array whose elements are dictionaries with information about the rescheduled games.
            The dictionary that has the following structure
            {game: (home_team, away_team),
            original_date: original datetime,
            new_date: played datetime}
        non_disruptions: list
            Array whose elements are dictionaries with information about the non-rescheduled games.
            The dictionary that has the following structure
            {game: (home_team, away_team),
            original_date: original datetime}
        """
        disruptions = []
        non_disruptions = []
        ids_disruptions = []
        ids_non_disruptions = []

        # For each variable, we check if its value is equal to 1
        for var in x_var_dict:
            if round(x_variables[x_var_dict[var]]) == 1:
                # We check the attributes of the variable that was chosen by the model
                home_team = var[0]
                away_team = var[1]
                original_date = var[2]
                game_date = var[3]
                proposed_date = var[4]
                id_match = var[5]

                # We setup the potential disruptions or non-disruptions dict
                pot_dis = {
                    'game': (home_team, away_team),
                    'original_date': original_date,
                    'game_date': game_date,
                    'id_match': id_match}
                pot_non_dis = {
                    'game': (home_team, away_team),
                    'original_date': original_date,
                    'id_match': id_match
                }

                # We check in which dictionary this game is
                if pot_dis in self.disruptions:
                    new_dis = {
                        'game': (home_team, away_team),
                        'original_date': proposed_date,
                        'game_date': game_date,
                        'id_match': id_match
                    }
                    # If it is, we add it
                    disruptions.append(new_dis)
                    ids_disruptions.append(id_match)
                if pot_non_dis in self.non_disruptions:
                    new_non_dis = {
                        'game': (home_team, away_team),
                        'original_date': proposed_date,
                        'id_match': id_match
                    }
                    non_disruptions.append(new_non_dis)
                    ids_non_disruptions.append(id_match)

        # For disruptions and non-disruptions that weren't modified during the process, we keep things this way
        for dis in self.disruptions:
            if dis['id_match'] not in ids_disruptions:
                disruptions.append(dis)
        for non_dis in self.non_disruptions:
            if non_dis['id_match'] not in ids_non_disruptions:
                non_disruptions.append(non_dis)

        return disruptions, non_disruptions


def get_disruptions_and_non_disruptions(fixture, covid_windows, start_date, end_date):
    """
    As we are re-setting all the schedule for past the end date, we calculate disruptions (past matches that were
    supposed to be played during a COVID window) and non-disruptions (every match after the end date)

    Parameters
    ----------
    fixture: pd.DataFrame
        Planned schedule
    covid_windows: dict
        Information by team of the windows in which there were reschedules
    start_date: datetime.datetime
        Start date of the period that we are going to evaluate, looking for matches to reschedule
    end_date
        End date of the period that we are going to evaluate, looking for matches to reschedule

    Returns
    -------
    disruptions: list
        Array whose elements are dictionaries with information about the rescheduled games.
        The dictionary that has the following structure
        {game: (home_team, away_team),
        original_date: original datetime,
        new_date: played datetime}
    non_disruptions: list
        Array whose elements are dictionaries with information about the non-rescheduled games.
        The dictionary that has the following structure
        {game: (home_team, away_team),
        original_date: original datetime}
    """
    # Create output dataframes
    disruptions = []
    non_disruptions = []

    # To evaluate disruptions, we will check the schedule that was planned to be played between
    # start and end date (the period that we are evaluating)
    df_evaluated_past = fixture[(fixture['original_date'] >= start_date) & (
            fixture['original_date'] <= end_date)]
    df_future = fixture[fixture['original_date'] > end_date]

    # For each match in df_evaluated_past, we check if the original date is in the COVID window of each team
    for index, row in df_evaluated_past.iterrows():
        home_team = row['home']
        away_team = row['visitor']
        original_date = row['original_date']
        game_date = row['game_date']

        # As covid windows is a list of lists, we append each element to a single list
        prohibited_dates = []
        for window in covid_windows[home_team]:
            for element in window:
                if element not in prohibited_dates:
                    prohibited_dates.append(element)
        for window in covid_windows[away_team]:
            for element in window:
                if element not in prohibited_dates:
                    prohibited_dates.append(element)
        # If the game is in any of the windows of a team, then is a disruption that we need to reschedule
        if original_date in prohibited_dates or original_date.date() in prohibited_dates:
            disruptions.append({
                'game': (home_team, away_team),
                'original_date': original_date,
                'game_date': game_date
            })

    # Now, we add every non-disruption to our non disruption list
    for index, row in df_future.iterrows():
        home_team = row['home']
        away_team = row['visitor']
        original_date = row['original_date']
        game_date = row['game_date']
        non_disruptions.append({
            'game': (home_team, away_team),
            'original_date': original_date,
            'game_date': game_date
        })

    return disruptions, non_disruptions
