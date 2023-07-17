import pandas as pd
from model_utils import Scheduler
import datetime
import numpy as np
from tqdm import tqdm
from itertools import combinations_with_replacement


class TourSequenceModel:
    def __init__(self, league, custom_fixture=None, start_date=datetime.datetime(2021, 1, 1),
                 end_date=datetime.datetime(2021, 1, 31), disruptions=[], non_disruptions=[],
                 max_mods_per_tour=100, max_adj_days=0, max_non_dis_mods=0, overlap_tours=True):
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
        custom_fixture['original_date'] = pd.to_datetime(custom_fixture['original_date'])
        self.max_date = np.max(custom_fixture['original_date'])
        self.df_fixture = S.df_fixture
        self.disruptions = disruptions
        self.non_disruptions = non_disruptions
        self.league_dates = S.league_dates
        self.max_games_rules = S.max_games_rules
        self.back_to_back_rules = S.back_to_back_rules
        self.dist_matrix = S.dist_matrix
        self.teams = list(self.df_fixture['home'].unique())
        self.tours_dict = S.get_tours_by_team()
        self.away_future_tours_dict = S.get_away_tours(self.tours_dict, end_date)
        self.resched_windows_dict = S.calculate_resched_windows()
        self.start_date = start_date
        self.end_date = end_date
        self.max_mods_per_tour = max_mods_per_tour
        self.max_adj_days = max_adj_days
        self.max_non_dis_mods = max_non_dis_mods
        self.overlap_tours = overlap_tours

        # Create a set of extended dates, that is equal to a date range that has the next 180 days after the end of the
        # original planned season
        self.extended_dates = list(pd.date_range(start=np.max(self.league_dates) + datetime.timedelta(days=1),
                                                 periods=180))
        
    def validate_tour_feasibility(self, tour):
        """
        Validates if a tour is feasible or not

        Args:
            tour (list): Sequence of matches
        """
        tour_start_date = tour[0]['proposed_date']
        tour_end_date = tour[len(tour) - 1]['proposed_date']
        tour_dates = list(pd.date_range(tour_start_date, tour_end_date))
        
        # For each day, we validate how many matches there are
        for d in tour_dates:
            for delta in range(3):
                window_start_date = d
                window_end_date = d + datetime.timedelta(days=delta)
                matches_to_check = [x for x in tour if ((x['proposed_date'] >= window_start_date) & ((x['proposed_date'] <= window_end_date)))]
                
                # Compare number of matches with maximum allowable number of matches
                max_allowable_games = self.max_games_rules[('all', delta + 1)]
                if len(matches_to_check) > max_allowable_games:
                    max_allowable_games
                    return False
        
        return True
    
    def calculate_away_tour_distance(self, tour, team):
        """
        Calculates the total distance traveled by team in a tour
        Args:
            tour (list): Sequence of matches
            team (str): NBA team
        """
        distance = 0
        if tour[0]['proposed_date'] > self.max_date:
            distance += self.dist_matrix[(team, tour[0]['game'][0])] * ((tour[0]['proposed_date'] - self.max_date).days + 1)
        else:
            distance += self.dist_matrix[(team, tour[0]['game'][0])]
        
        if len(tour) > 1:
            for i in range(len(tour) - 1):
                match_i = tour[i]
                match_j = tour[i + 1]
                if match_i['proposed_date'] > self.max_date:
                    distance += self.dist_matrix[(match_i['game'][0], match_j['game'][0])] * ((match_i['proposed_date'] - self.max_date).days + 1)
                else:
                    distance += self.dist_matrix[(match_i['game'][0], match_j['game'][0])]
            
        if tour[len(tour) - 1]['proposed_date'] > self.max_date:
            distance += self.dist_matrix[(team, tour[len(tour) - 1]['game'][0])] * ((tour[len(tour) - 1]['proposed_date'] - self.max_date).days + 1)
        else:
            distance += self.dist_matrix[(team, tour[len(tour) - 1]['game'][0])]
        return distance
               
        
    def get_tour_shifts(self):
        """
        We calculate, for all the tours, all the combinations that include shifts of the non-disruptions matches and save it 
        in multiple dictionaries that will be used for the model, validating that the tour respects schedule rules
        
        Returns
        -------
        shifts_dict: list
            List with information about all the shifted tours, saved as dicts
        shifts_per_team_and_date_dict: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as dicts
        shifts_per_tour: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples
        shifts_per_team_and_date_dict_tuples: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as tuples
        shifts_per_tour_tuples: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
        """
        print("Creating tour shifts!")
        shifts_dict = []
        shifts_per_team_and_date_dict = {}
        shifts_per_tour = {} 
        
        shifts_dict_tuples = []
        shifts_per_team_and_date_dict_tuples = {}
        shifts_per_tour_tuples = {} 
        
        for team in self.teams:
            shifts_per_team_and_date_dict[team] = {}
            shifts_per_tour[team] = {}
            shifts_per_team_and_date_dict_tuples[team] = {}
            shifts_per_tour_tuples[team] = {}
        
        for team in self.teams:
            
            tours = self.away_future_tours_dict[team]
            
            n = 0            
            for tour in tours:
                # Calculate stats about the existing tour
                shifts_per_tour[team][n] = []
                shifts_per_tour_tuples[team][n] = []
                
                start_date = tour[0]['proposed_date']
                end_date = tour[len(tour) - 1]['proposed_date']
                
                # Calculate distance
                distance = self.calculate_away_tour_distance(tour, team)
                # Construct variable
                tour_variable = {
                    'original_start_date': start_date,
                    'original_end_date': end_date,
                    'original_sequence': tour,
                    'new_start_date': start_date,
                    'new_end_date': end_date,
                    'new_sequence': tour,
                    'n_mods': 0,
                    'distance': distance,
                    'special': 0
                }
                tour_variable_tuple = (
                    start_date, end_date, tour, start_date, end_date, tour, 0, distance, 0
                )
                # Append variable to our dictionaries
                shifts_dict.append(tour_variable)
                shifts_dict_tuples.append(tour_variable_tuple)
                shifts_dict
                
                for match in tour:
                    match_date = match['proposed_date']
                    if match_date in shifts_per_team_and_date_dict[team].keys():
                        shifts_per_team_and_date_dict[team][match_date].append(tour_variable)
                        shifts_per_team_and_date_dict_tuples[team][match_date].append(tour_variable_tuple)
                        
                    else:
                        shifts_per_team_and_date_dict[team][match_date] = [tour_variable]
                        shifts_per_team_and_date_dict_tuples[team][match_date] = [tour_variable_tuple]
                    
                    other_team = match['game'][0]
                    if match_date in shifts_per_team_and_date_dict[other_team].keys():
                        shifts_per_team_and_date_dict[other_team][match_date].append(tour_variable)
                        shifts_per_team_and_date_dict_tuples[other_team][match_date].append(tour_variable_tuple)
                    else:
                        shifts_per_team_and_date_dict[other_team][match_date] = [tour_variable]
                        shifts_per_team_and_date_dict_tuples[other_team][match_date] = [tour_variable_tuple]
                        
                    
                shifts_per_tour[team][n].append(tour_variable)
                shifts_per_tour_tuples[team][n].append(tour_variable_tuple)
                
                # Now, we apply shifts
                buffer = [tour_variable]
                for i in range(len(tour)):
                    # We will apply modifications in order
                    for b in buffer:
                        tour_to_modified_unchanged = b['new_sequence']
                        tour_to_modify = b['new_sequence']
                        days_to_modify = list(range(-self.max_adj_days, self.max_adj_days + 1))
                        date_to_modify = b['new_sequence'][i]['proposed_date']
                        # We will add modifications, substracting or adding up to max_adj_days days
                        add_to_buffer = []
                        for d in days_to_modify:
                            if d != 0:
                                tour_to_modify[i]['proposed_date'] = date_to_modify + datetime.timedelta(days=d)
                                
                                feas_ok = self.validate_tour_feasibility(tour_to_modify)
                                
                                # If we have checked the feasibility of the tour, we order it and calculate distance
                                if feas_ok:
                                    # We sort it
                                    tour_to_modify = sorted(tour_to_modify, key=lambda d: d['proposed_date']) 
                                    distance = self.calculate_away_tour_distance(tour_to_modify, team)
                                    
                                    if b['n_mods'] + 1 <= self.max_mods_per_tour:
                                        # We create the tour variable
                                        tour_variable = {
                                            'original_start_date': b['original_start_date'],
                                            'original_end_date': b['original_end_date'],
                                            'original_sequence': b['original_sequence'],
                                            'new_start_date': tour_to_modify[0]['proposed_date'],
                                            'new_end_date': tour_to_modify[len(tour_to_modify) - 1]['proposed_date'],
                                            'new_sequence': tour_to_modify,
                                            'n_mods': b['n_mods'] + 1,
                                            'distance': distance,
                                            'special': b['special']
                                        }
                                        
                                        tour_variable_tuple = (
                                            b['original_start_date'],
                                            b['original_end_date'],
                                            b['original_sequence'],
                                            tour_to_modify[0]['proposed_date'],
                                            tour_to_modify[len(tour_to_modify) - 1]['proposed_date'],
                                            tour_to_modify,
                                            b['n_mods'] + 1,
                                            distance,
                                            b['special']
                                        )
                                        
                                        shifts_dict.append(tour_variable)
                                        shifts_per_tour[team][n].append(tour_variable)
                                        
                                        shifts_dict_tuples.append(tour_variable_tuple)
                                        shifts_per_tour_tuples[team][n].append(tour_variable_tuple)
                                        
                                        for match in tour:
                                            match_date = match['proposed_date']
                                            if match_date in shifts_per_team_and_date_dict[team].keys():
                                                shifts_per_team_and_date_dict[team][match_date].append(tour_variable)
                                                shifts_per_team_and_date_dict_tuples[team][match_date].append(tour_variable_tuple)
                                            else:
                                                shifts_per_team_and_date_dict[team][match_date] = [tour_variable]
                                                shifts_per_team_and_date_dict_tuples[team][match_date] = [tour_variable_tuple]
                                                
                                            other_team = match['game'][0]
                                            if match_date in shifts_per_team_and_date_dict[other_team].keys():
                                                shifts_per_team_and_date_dict[other_team][match_date].append(tour_variable)
                                                shifts_per_team_and_date_dict_tuples[other_team][match_date].append(tour_variable_tuple)
                                            else:
                                                shifts_per_team_and_date_dict[other_team][match_date] = [tour_variable]
                                                shifts_per_team_and_date_dict_tuples[other_team][match_date] = [tour_variable_tuple]
                                            
                                        add_to_buffer.append(tour_variable)
                    # Add new variants of our tour to this buffer
                    buffer = buffer + add_to_buffer                                  


        return shifts_dict, shifts_per_team_and_date_dict, shifts_per_tour, shifts_dict_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_tour_tuples
        

    def get_tour_shifts_v2(self):
        """
        We calculate, for all the tours, all the combinations that include shifts of the non-disruptions matches and save it 
        in multiple dictionaries that will be used for the model, validating that the tour respects schedule rules
        
        Returns
        -------
        shifts_dict: list
            List with information about all the shifted tours, saved as dicts
        shifts_per_team_and_date_dict: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as dicts
        shifts_per_tour: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples
        shifts_per_team_and_date_dict_tuples: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as tuples
        shifts_per_tour_tuples: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
        """
        print("Creating tour shifts!")
        shifts_dict = []
        shifts_per_team_and_date_dict = {}
        shifts_per_tour = {} 
        
        shifts_dict_tuples = []
        shifts_per_team_and_date_dict_tuples = {}
        shifts_per_tour_tuples = {} 
        
        for team in self.teams:
            shifts_per_team_and_date_dict[team] = {}
            shifts_per_tour[team] = {}
            shifts_per_team_and_date_dict_tuples[team] = {}
            shifts_per_tour_tuples[team] = {}
        
        for team in self.teams:
            
            tours = self.away_future_tours_dict[team]
            
            n = 0            
            for tour in tours:
                # Calculate stats about the existing tour
                shifts_per_tour[team][n] = []
                shifts_per_tour_tuples[team][n] = []
                
                start_date = tour[0]['proposed_date']
                end_date = tour[len(tour) - 1]['proposed_date']
                
                # Calculate distance
                distance = self.calculate_away_tour_distance(tour, team)
                # Construct variable
                tour_variable = {
                    'original_start_date': start_date,
                    'original_end_date': end_date,
                    'original_sequence': tour,
                    'new_start_date': start_date,
                    'new_end_date': end_date,
                    'new_sequence': tour,
                    'n_mods': 0,
                    'distance': distance,
                    'special': 0
                }
                tour_variable_tuple = (
                    start_date, end_date, tour, start_date, end_date, tour, 0, distance, 0
                )
                # Append variable to our dictionaries
                shifts_dict.append(tour_variable)
                shifts_dict_tuples.append(tour_variable_tuple)
                
                for match in tour:
                    match_date = match['proposed_date']
                    if match_date in shifts_per_team_and_date_dict[team].keys():
                        shifts_per_team_and_date_dict[team][match_date].append(tour_variable)
                        shifts_per_team_and_date_dict_tuples[team][match_date].append(tour_variable_tuple)
                        
                    else:
                        shifts_per_team_and_date_dict[team][match_date] = [tour_variable]
                        shifts_per_team_and_date_dict_tuples[team][match_date] = [tour_variable_tuple]
                    
                    other_team = match['game'][0]
                    if match_date in shifts_per_team_and_date_dict[other_team].keys():
                        shifts_per_team_and_date_dict[other_team][match_date].append(tour_variable)
                        shifts_per_team_and_date_dict_tuples[other_team][match_date].append(tour_variable_tuple)
                    else:
                        shifts_per_team_and_date_dict[other_team][match_date] = [tour_variable]
                        shifts_per_team_and_date_dict_tuples[other_team][match_date] = [tour_variable_tuple]
                        
                    
                shifts_per_tour[team][n].append(tour_variable)
                shifts_per_tour_tuples[team][n].append(tour_variable_tuple)
                
                # Now, we calculate the range of days to modify
                days_to_modify = list(range(-self.max_adj_days, self.max_adj_days + 1))
                
                # We calculate all the possible modifications
                mods_masks = list(combinations_with_replacement(days_to_modify, len(tour)))
                
                # We apply the modifications
                for mask in mods_masks:
                    mod_tour = []
                    for i in range(len(tour)):
                        match_i = tour[i] 
                        match_original = match_i
                        new_match = {
                            'game': (match_i['game'][0], match_i['game'][1]),
                            'original_date': match_i['original_date'],
                            'game_date': match_i['game_date'],
                            'proposed_date': match_i['original_date'] + datetime.timedelta(days=mask[i]) 
                        }
                        mod_tour.append(new_match)
                    
                    # Calculate feasibility
                    mod_tour = sorted(mod_tour, key=lambda d: d['proposed_date']) 
                    feas_ok = self.validate_tour_feasibility(mod_tour)
                    if feas_ok:
                        
                        # Calculate distance
                        distance = self.calculate_away_tour_distance(mod_tour, team)
                        n_mods = np.sum([1 if x != 0 else 0 for x in mask])
                
                        tour_variable = {
                            'original_start_date': tour[0]['original_date'],
                            'original_end_date': tour[len(tour) - 1]['original_date'],
                            'original_sequence': tour,
                            'new_start_date': mod_tour[0]['proposed_date'],
                            'new_end_date': mod_tour[len(mod_tour) - 1]['proposed_date'],
                            'new_sequence': mod_tour,
                            'n_mods': n_mods,
                            'distance': distance,
                            'special': 0
                        }
                        
                        tour_variable_tuple = (
                            tour[0]['original_date'],
                            tour[len(tour) - 1]['original_date'],
                            tour,
                            mod_tour[0]['proposed_date'],
                            mod_tour[len(mod_tour) - 1]['proposed_date'],
                            mod_tour,
                            n_mods,
                            distance,
                            0
                        )
                        
                        shifts_dict.append(tour_variable)
                        shifts_per_tour[team][n].append(tour_variable)
                        
                        shifts_dict_tuples.append(tour_variable_tuple)
                        shifts_per_tour_tuples[team][n].append(tour_variable_tuple)
                        
                        for match in mod_tour:
                            match_date = match['proposed_date']
                            if match_date in shifts_per_team_and_date_dict[team].keys():
                                shifts_per_team_and_date_dict[team][match_date].append(tour_variable)
                                shifts_per_team_and_date_dict_tuples[team][match_date].append(tour_variable_tuple)
                            else:
                                shifts_per_team_and_date_dict[team][match_date] = [tour_variable]
                                shifts_per_team_and_date_dict_tuples[team][match_date] = [tour_variable_tuple]
                                
                            other_team = match['game'][0]
                            if match_date in shifts_per_team_and_date_dict[other_team].keys():
                                shifts_per_team_and_date_dict[other_team][match_date].append(tour_variable)
                                shifts_per_team_and_date_dict_tuples[other_team][match_date].append(tour_variable_tuple)
                            else:
                                shifts_per_team_and_date_dict[other_team][match_date] = [tour_variable]
                                shifts_per_team_and_date_dict_tuples[other_team][match_date] = [tour_variable_tuple]
                
                n += 1  
        return shifts_dict, shifts_per_team_and_date_dict, shifts_per_tour, shifts_dict_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_tour_tuples
            
    
    def get_tours_switches(self, shifts_dict, shifts_per_team_and_date_dict, shifts_per_tour, shifts_dict_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_tour_tuples):
        """
        We switch matches within a tour as another alternative to our modifications

        Parameters
        -------
        shifts_dict: list
            List with information about all the shifted tours
        shifts_per_team_and_date_dict: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there
        shifts_per_tour: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it        
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples
        shifts_per_team_and_date_dict_tuples: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as tuples
        shifts_per_tour_tuples: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
            
        Returns
        -------
        shifts_switches_dict: list
            List with information about all the shifted tours
        shifts_switches_per_team_and_date_dict: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there
        shifts_switches_per_tour: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it
            
        """
        shifts_switches_dict = shifts_dict
        shifts_switches_per_team_and_date_dict = shifts_per_team_and_date_dict
        shifts_switches_per_tour = shifts_per_tour
        for team in self.teams:
            for tour in shifts_switches_per_tour[team]:
                # For each sequence, we make the shifts
                for tour_variable in shifts_switches_per_tour[team][tour]:
                    sequence = tour_variable['new_sequence']
                    sequence_unchanged = sequence.copy()
                    # Take two matches, and make the shift
                    for i in range(len(sequence_unchanged)):
                        buffer = [sequence]
                        for seq in buffer:
                            for j in range(i + 1, len(sequence_unchanged)):
                                add_to_buffer = []
                                new_sequence = seq.copy()
                                date_i = seq[i]['original_date']
                                date_j = seq[j]['original_date']
                                new_sequence[i]['original_date'] = date_j
                                new_sequence[j]['original_date'] = date_i
                                new_sequence = sorted(new_sequence, key=lambda d: d['original_date']) 
                                distance = self.calculate_away_tour_distance(new_sequence, team)
                                

        return shifts_switches_dict, shifts_switches_per_team_and_date_dict, shifts_switches_per_tour  
    
    def insert_disruptions(self, shifts_dict, shifts_per_team_and_date_dict, shifts_per_tour,
                           shifts_dict_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_tour_tuples):
        """
        Checks all the disruptions and sees in which potential dates could be inserted

        Parameters
        ----------
        shifts_dict: list
            List with information about all the shifted tours, saved as dicts
        shifts_per_team_and_date_dict: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as dicts
        shifts_per_tour: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples
        shifts_per_team_and_date_dict_tuples: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as tuples
        shifts_per_tour_tuples: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
            
        Returns
        ----------
        shifts_dict: list
            List with information about all the shifted tours, saved as dicts
        shifts_per_team_and_date_dict: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as dicts
        shifts_per_tour: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as dicts
        shifts_per_disruption: dict
            Dictionary that has as keys, each disruption, and by values, all the sequences created to it saved as dicts
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples
        shifts_per_team_and_date_dict_tuples: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as tuples
        shifts_per_tour_tuples: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as tuples
        shifts_per_disruption_tuples: dict
            Dictionary that has as keys, each disruption, and by values, all the sequences created to it saved as tuples
        """     
        print("Inserting Disruptions!")
        shifts_per_disruption = {}   
        shifts_per_disruption_tuples = {}
        for dis in tqdm(self.disruptions):
            shifts_per_disruption[(dis['game'][0], dis['game'][1], dis['original_date'], dis['game_date'])] = []
            shifts_per_disruption_tuples[(dis['game'][0], dis['game'][1], dis['original_date'], dis['game_date'])] = []
            
            away_team = dis['game'][1]
            team_tours = shifts_per_tour[away_team]
            tours_watched_by_disruption = []
            for tour in team_tours:
                if tour not in tours_watched_by_disruption:
                    tours_watched_by_disruption.append(tour)
                    to_add = []
                    to_add_tuples = []
                    for sequence in team_tours[tour]:
                        first_date = sequence['new_sequence'][0]['proposed_date']
                        last_date = sequence['new_sequence'][len(sequence['new_sequence']) - 1]['proposed_date']
                        
                        first_candidate = first_date - datetime.timedelta(days=2)
                        last_candidate = last_date + datetime.timedelta(days=2)
                        
                        seq_dates = []
                        for game in sequence['new_sequence']:
                            seq_dates.append(game['proposed_date'])
                        
                        # Set list of candidates that we will consider for a disruption within a tour
                        candidate_dates = list(pd.date_range(first_candidate, last_date))
                        for d in candidate_dates:
                            if d not in seq_dates:
                                # Setup variable
                                disruption_candidate = {
                                    'game': (dis['game'][0],
                                            dis['game'][1]),
                                'original_date': dis['original_date'],
                                'game_date': dis['game_date'],
                                'proposed_date':  d
                                }
                                # We create the candidate sequence
                                candidate_sequence = sequence['new_sequence'] + [disruption_candidate]
                                # We sort it
                                candidate_sequence = sorted(candidate_sequence, key=lambda d: d['proposed_date']) 
                                # We validate that if follows scheduling rules
                                feas_ok = self.validate_tour_feasibility(candidate_sequence)
                                # If we have checked the feasibility of the tour, we order it and calculate distance
                                if feas_ok:
                                    # If this is a feasible sequence, then we calculate distance
                                    distance = self.calculate_away_tour_distance(candidate_sequence, away_team)
                                    
                                    # We create the variable
                                    tour_variable = {
                                                    'original_start_date': sequence['original_start_date'],
                                                    'original_end_date': sequence['original_end_date'],
                                                    'original_sequence': sequence['original_sequence'],
                                                    'new_start_date': candidate_sequence[0]['proposed_date'],
                                                    'new_end_date': candidate_sequence[len(candidate_sequence) - 1]['proposed_date'],
                                                    'new_sequence': candidate_sequence,
                                                    'n_mods': sequence['n_mods'],
                                                    'distance': distance,
                                                    'special': sequence['special']
                                                }
                                    tour_variable_tuple = (
                                        sequence['original_start_date'],
                                        sequence['original_end_date'],
                                        sequence['original_sequence'],
                                        candidate_sequence[0]['proposed_date'],
                                        candidate_sequence[len(candidate_sequence) - 1]['proposed_date'],
                                        candidate_sequence,
                                        sequence['n_mods'],
                                        distance,
                                        sequence['special']
                                    )
                                    # Add the variable
                                    shifts_dict.append(tour_variable)
                                    shifts_dict_tuples.append(tour_variable_tuple)
                                    
                                    to_add.append(tour_variable)
                                    to_add_tuples.append(tour_variable_tuple)
                                    
                                    shifts_per_disruption[(dis['game'][0], dis['game'][1], dis['original_date'], dis['game_date'])].append(tour_variable)
                                    shifts_per_disruption_tuples[(dis['game'][0], dis['game'][1], dis['original_date'], dis['game_date'])].append(tour_variable_tuple)
                                    
                                    for match in candidate_sequence:
                                        match_date = match['proposed_date']
                                        if match_date in shifts_per_team_and_date_dict[away_team].keys():
                                            shifts_per_team_and_date_dict[away_team][match_date].append(tour_variable)
                                            shifts_per_team_and_date_dict_tuples[away_team][match_date].append(tour_variable_tuple)
                                        else:
                                            shifts_per_team_and_date_dict[away_team][match_date] = [tour_variable]
                                            shifts_per_team_and_date_dict_tuples[away_team][match_date] = [tour_variable_tuple]
                                                        
                                        other_team = match['game'][0]
                                        if match_date in shifts_per_team_and_date_dict[other_team].keys():
                                            shifts_per_team_and_date_dict[other_team][match_date].append(tour_variable)
                                            shifts_per_team_and_date_dict_tuples[other_team][match_date].append(tour_variable_tuple)
                                        else:
                                            shifts_per_team_and_date_dict[other_team][match_date] = [tour_variable]
                                            shifts_per_team_and_date_dict_tuples[other_team][match_date] = [tour_variable_tuple]
                                
                shifts_per_tour[away_team][tour] = shifts_per_tour[away_team][tour] + to_add
                shifts_per_tour_tuples[away_team][tour] = shifts_per_tour_tuples[away_team][tour] + to_add_tuples
            
            # We now create new tours, that will be made up of this only match, for dates from the max date onwards
            first_candidate = self.max_date - datetime.timedelta(days=4)
            last_candidate = self.max_date + datetime.timedelta(days=30)
            candidate_dates = list(pd.date_range(first_candidate, last_candidate))
            team_tours = list(shifts_per_tour[away_team].keys())
            max_team_tour = np.max(team_tours)
            n_tour = max_team_tour + 1
            for d in candidate_dates:
                disruption_candidate = {
                            'game': (dis['game'][0],
                                     dis['game'][1]),
                           'original_date': dis['original_date'],
                           'game_date': dis['game_date'],
                           'proposed_date': d
                        }
                candidate_sequence = [disruption_candidate]
                # Calculate distance
                distance = self.calculate_away_tour_distance(candidate_sequence, away_team)
                tour_variable = {
                            'original_start_date': d,
                            'original_end_date': d,
                            'original_sequence': candidate_sequence,
                            'new_start_date': d,
                            'new_end_date': d,
                            'new_sequence': candidate_sequence,
                            'n_mods': 0,
                            'distance': distance,
                            'special': 1
                            }
                tour_variable_tuple = (
                    d,
                    d,
                    candidate_sequence,
                    d,
                    d,
                    candidate_sequence,
                    0,
                    distance,
                    1
                )
                # Add variables  
                shifts_dict.append(tour_variable)
                shifts_per_tour[away_team][n_tour]= [tour_variable]
                
                shifts_dict_tuples.append(tour_variable_tuple)
                shifts_per_tour_tuples[away_team][n_tour] = [tour_variable_tuple]
                n_tour += 1
                shifts_per_disruption[(dis['game'][0], dis['game'][1], dis['original_date'], dis['game_date'])].append(tour_variable)
                shifts_per_disruption_tuples[(dis['game'][0], dis['game'][1], dis['original_date'], dis['game_date'])].append(tour_variable_tuple)
                for match in candidate_sequence:
                    match_date = match['proposed_date']
                    if match_date in shifts_per_team_and_date_dict[away_team].keys():
                        shifts_per_team_and_date_dict[away_team][match_date].append(tour_variable)
                        shifts_per_team_and_date_dict_tuples[away_team][match_date].append(tour_variable_tuple)
                    else:
                        shifts_per_team_and_date_dict[away_team][match_date] = [tour_variable]
                        shifts_per_team_and_date_dict_tuples[away_team][match_date] = [tour_variable_tuple]
                                    
                    other_team = match['game'][0]
                    if match_date in shifts_per_team_and_date_dict[other_team].keys():
                        shifts_per_team_and_date_dict[other_team][match_date].append(tour_variable)
                        shifts_per_team_and_date_dict_tuples[other_team][match_date].append(tour_variable_tuple)
                    else:
                        shifts_per_team_and_date_dict[other_team][match_date] = [tour_variable]
                        shifts_per_team_and_date_dict_tuples[other_team][match_date] = [tour_variable_tuple]
           
                                
        return shifts_dict, shifts_per_team_and_date_dict, shifts_per_tour, shifts_per_disruption, shifts_dict_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_tour_tuples, shifts_per_disruption_tuples
    
    def create_decision_variables(self, shifts_dict_tuples):
        """
        Creates a dictionary whose key are the tuples that identify each sequence

        Parameters:
        ----------
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples
        
        Returns:
        -------
        x_var_dict: dict
            Dictionary whose key are the tuples that identify the sequence and the value is a number that will identify the order
        """
        n = 0
        x_var_dict = {}
        for var in shifts_dict_tuples:
            x_var_dict[str(var)] = n
            n += 1
        return x_var_dict

    def add_schedule_rules_constraints_all(self, x_var_dict, shifts_per_team_and_date_dict_tuples, prob_lp, n_days):
        """
        Adds a set of constraint that limits the number of games in a particular set of days. For example, for each set
        of consecutive days, we can't have more than two games. A constraint will be created per team, days and number
        of days. For example this constraint

        sum_{i} x_it + sum_{i} x_it+1 <= 2 \foreach t, i \in GamesOfTeamA

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model
        shifts_per_team_and_date_dict_tuples: dict
            Dictionary that has as keys, a tuple of team and date, and as value, all the tours that have games for which a team plays there, saved as tuples            
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
                if initial_day >= (self.end_date - datetime.timedelta(days=7)):

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
                    dates_to_check = list(pd.date_range(start, end))
                    # Check the variables created for each date
                    for d in dates_to_check:
                        if d in shifts_per_team_and_date_dict_tuples[team].keys():
                            vars_to_check = shifts_per_team_and_date_dict_tuples[team][d]
                            for var in vars_to_check:
                                if x_var_dict[str(var)] not in ind:
                                    ind.append(x_var_dict[str(var)])
                                    val.append(1)

                    # We check if we have variables in order to add our constraint
                    if len(ind) > 0:
                        row = [ind, val]

                        # We add the constraint, checking the number of played games and the maximum allowed
                        prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'],
                                                       rhs=[self.max_games_rules[('all', n_days)] - n_games])

        return prob_lp
    
    def assign_all_disruptions(self, x_var_dict, shifts_per_disruption_tuples, prob_lp):
        """
        Constraint that forces that all disruptions should be put at some time
        
        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model
        shifts_per_disruption_tuples: dict
            Dictionary that has as keys, each disruption, and by values, all the sequences created to it saved as tuples
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for dis in self.disruptions:
            ind = []
            val = []
            dis_tuple = (dis['game'][0], dis['game'][1], dis['original_date'], dis['game_date'])
            # Check the variables created
            for var in shifts_per_disruption_tuples[dis_tuple]:
                if x_var_dict[str(var)] not in ind:
                    ind.append(x_var_dict[str(var)])
                    val.append(1)
            
            # We create the constraint
            row = [ind, val]

            # We add the constraint, checking the number of played games and the maximum allowed
            prob_lp.linear_constraints.add(lin_expr=[row], senses=['E'], rhs=[1])
            
        return prob_lp
    
    def assign_one_sequence_per_tour(self, x_var_dict, shifts_per_tour_tuples, prob_lp):
        """
        Checks all sequence within a tour and makes sure that one sequence is selected. If the sequence is all made up of disruptions, we will not add a equal constraint, but a less or equal constraint

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model
        shifts_per_tour_tuples: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as tuples            
        prob_lp: cplex.Cplex
            Cplex problem
        
        Returns
        -------            
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for team in self.teams:
            for tour in shifts_per_tour_tuples[team]:
                # Check if there is one sequence of this tour which is made up of all disruptions
                max_special = -1
                sequences = shifts_per_tour_tuples[team][tour]
                for seq in sequences:
                    special = seq[len(seq) - 1]
                    if special > max_special:
                        max_special = special
                if max_special == 1:
                    sign = 'L'
                else:
                    sign = 'E'
                    
                if max_special == 0:
                    n_seq =0
                    seqs_bad = []
                    for seq in sequences:
                        mods = 0
                        new_seq = seq[5]
                        for match in new_seq:
                            if match['proposed_date'] != match['original_date']:
                                mods += 1
                        if mods == len(new_seq):
                            n_seq += 1
                            seqs_bad.append(seq)
                    if n_seq == len(sequences):
                        seqs_bad
                    
                # Add variables
                ind = []
                val = []
                for seq in sequences:
                    if x_var_dict[str(seq)] not in ind:
                        ind.append(x_var_dict[str(seq)])
                        val.append(1)
                row = [ind, val]
                # Add constraint
                prob_lp.linear_constraints.add(lin_expr=[row], senses=[sign], rhs=[1])
        return prob_lp
                
    def limit_non_disruption_mods(self, x_var_dict, shifts_dict_tuples, prob_lp):
        """
        Limits the number of non disruptions mods allowed
        
        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples            
        prob_lp: cplex.Cplex
            Cplex problem
        
        Returns
        -------            
        prob_lp: cplex.Cplex
            Cplex problem        
        """    
        ind_non = []
        val_non = []

        ind_ok = []
        val_ok = []        
        """
        Remember that this is the structure of the variable
        tour_variable = {
            0: 'original_start_date'
            1: 'original_end_date'
            2: 'original_sequence'
            3: 'new_start_date'
            4: 'new_end_date'
            5: 'new_sequence'
            6: 'n_mods'
            7: 'distance'
            8: 'special'
            }
        """
        
        for var in shifts_dict_tuples:
            mods = var[6]
            mods_cal = 0
            for game in var[5]:
                game_var = {
                'game': (game['game'][0],
                         game['game'][1]),
                'original_date': game['original_date'],
                'game_date': game['game_date']}
                if game_var not in self.disruptions:
                    if game['original_date'] != game['proposed_date']:
                        mods_cal += 1
            
            if mods_cal > self.max_mods_per_tour:
                if x_var_dict[str(var)] not in ind_non:
                    ind_non.append(x_var_dict[str(var)])
                    val_non.append(1)
            else:
                if x_var_dict[str(var)] not in ind_ok:
                    ind_ok.append(x_var_dict[str(var)])
                    val_ok.append(mods_cal)
            
                        
        row = [ind_ok, val_ok]
        prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[self.max_non_dis_mods])
        
        row = [ind_non, val_non]
        prob_lp.linear_constraints.add(lin_expr=[row], senses=['E'], rhs=[0])        
        return prob_lp
    
    def define_a_sequence_for_all_non_disruptions(self, x_var_dict, shifts_dict_tuples, prob_lp):
        """
        Check that every non disruption is scheduled

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model
        shifts_dict_tuples: list
            List with information about all the shifted tours, saved as tuples            
        prob_lp: cplex.Cplex
            Cplex problem
        
        Returns
        -------            
        prob_lp: cplex.Cplex
            Cplex problem        
        """
        non_disruptions_list = []
        non_disruptions_list_dict = {}
        for var in tqdm(shifts_dict_tuples):
            mods = var[6]
            mods_cal = 0
            for game in var[5]:
                game_var = {
                'game': (game['game'][0],
                         game['game'][1]),
                'original_date': game['original_date'],
                'game_date': game['game_date']}
                if game_var not in self.disruptions and game['original_date'] > self.end_date:
                    non_disruptions_list.append(game_var)
                    
                    if str(game_var) not in non_disruptions_list_dict.keys():
                        non_disruptions_list_dict[str(game_var)] = [var]
                    else:
                        non_disruptions_list_dict[str(game_var)].append(var)
        
        # Check the variables and create the constraints        
        for non_dis in tqdm(non_disruptions_list_dict):
            ind = list(set([x_var_dict[str(x)] for x in non_disruptions_list_dict[non_dis]]))
            val = [1]*len(ind)
            row = [ind, val]
            prob_lp.linear_constraints.add(lin_expr=[row], senses=['E'], rhs=[1])
        return prob_lp      
            
            
            
            
    
    def avoid_operlapping_tours(self, x_var_dict, shifts_per_tour_tuples, prob_lp):
        """
        Avoids having a tour that starts before another one has ended

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model
        shifts_per_tour_tuples: dict
            Dictionary that has as keys, each tour, and by values, all the sequences created to it saved as tuples            
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for team in self.teams:
            team_vars = []
            # Check all the variables of a team
            for tour in shifts_per_tour_tuples[team]:
                for var in shifts_per_tour_tuples[team][tour]:
                    team_vars.append(var)
            
            # Once the variables have been saved we create variables that check the finishing and start date of each tour
            for i in range(len(team_vars)):
                for j in range(1 + i, len(team_vars)):
                    var_i = team_vars[i]
                    var_j = team_vars[j]
                    """
                    Remember that this is the structure of the variable
                    tour_variable = {
                        0: 'original_start_date'
                        1: 'original_end_date'
                        2: 'original_sequence'
                        3: 'new_start_date'
                        4: 'new_end_date'
                        5: 'new_sequence'
                        6: 'n_mods'
                        7: 'distance'
                        8: 'special'
                        }
                    """
                    start_i = var_i[3]
                    end_i = var_i[4]
                    start_j = var_j[3]
                    end_j = var_j[4]
                    
                    # We check if tour j starts after the start of i, but before the end of i
                    if start_j > start_i and start_j < end_i:
                        ind = [x_var_dict[str(var_i)], x_var_dict[str(var_j)]]
                        val = [1, 1]
                        row = [ind, val]
                        prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[1])
        return prob_lp


    def no_games_on_prohibited_dates(self, x_var_dict, prob_lp, shifts_per_team_and_date_dict_tuples):
        """
        Constraint that forces to have no games on dates when there are no games

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model
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
        for team in shifts_per_team_and_date_dict_tuples:
            for day in dates_without_matches:
                if day in shifts_per_team_and_date_dict_tuples[team].keys():
                    vars = shifts_per_team_and_date_dict_tuples[team][day]
                    for var in vars:
                        if x_var_dict[str(var)] not in ind:
                            ind.append(x_var_dict[str(var)])
                            val.append(1)
        # We check each variable and see if we should add it
        if len(ind) > 0:
            row = [ind, val]
            prob_lp.linear_constraints.add(lin_expr=[row], senses=['E'], rhs=[0])
        return prob_lp

    def add_constraint_matrix(self, x_var_dict, shifts_per_tour_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_disruption_tuples, shifts_dict_tuples, prob_lp):
        """
        Adds constraint matrix to the problem, calling all the different methods

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
        # Add every created constraint
        if not self.overlap_tours:
            prob_lp = self.avoid_operlapping_tours(x_var_dict, shifts_per_tour_tuples, prob_lp)

        """
        for n_days in list(range(1, 4)):
            #prob_lp = self.add_schedule_rules_constraints_home(x_var_dict, prob_lp, n_days)
            #prob_lp = self.add_schedule_rules_constraints_away(x_var_dict, prob_lp, n_days)
            prob_lp = self.add_schedule_rules_constraints_all(x_var_dict, shifts_per_team_and_date_dict_tuples, prob_lp, n_days)
        """
         
        prob_lp = self.assign_all_disruptions(x_var_dict, shifts_per_disruption_tuples, prob_lp)
        prob_lp = self.assign_one_sequence_per_tour(x_var_dict, shifts_per_tour_tuples, prob_lp)
        prob_lp = self.limit_non_disruption_mods(x_var_dict, shifts_dict_tuples, prob_lp)
        prob_lp = self.no_games_on_prohibited_dates(x_var_dict, prob_lp, shifts_per_team_and_date_dict_tuples)
        prob_lp = self.define_a_sequence_for_all_non_disruptions(x_var_dict, shifts_dict_tuples, prob_lp)

        return prob_lp

    def populate_by_row(self, x_var_dict, shifts_per_tour_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_disruption_tuples, shifts_dict_tuples, prob_lp):
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

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Here the objective funcion will be equal to
        # min sum_{dist_it*x_it}.
        coef = []
        lower_bounds = []
        upper_bounds = []
        types = []
        names = []
        for var in shifts_dict_tuples:
            """
            Remember that this is the structure of the variable
            tour_variable = {
                0: 'original_start_date'
                1: 'original_end_date'
                2: 'original_sequence'
                3: 'new_start_date'
                4: 'new_end_date'
                5: 'new_sequence'
                6: 'n_mods'
                7: 'distance'
                8: 'special'
                }
                """
            coef.append(float(var[7]))
            lower_bounds.append(0)
            upper_bounds.append(1)
            types.append('B')

        prob_lp.variables.add(obj=coef, lb=lower_bounds, ub=upper_bounds, types=types)
        prob_lp.objective.set_sense(prob_lp.objective.sense.minimize)
        prob_lp = self.add_constraint_matrix(x_var_dict, shifts_per_tour_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_disruption_tuples, shifts_dict_tuples, prob_lp)
        prob_lp.write(f"C:/Users/HP/Documents/Sports Analytics/Re Scheduling/code/models/tours sequence model/RescheduleFixture.lp")

        return prob_lp

    def solve_lp(self, x_var_dict, shifts_per_tour_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_disruption_tuples, shifts_dict_tuples, prob_lp):
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
        prob_lp = self.populate_by_row(x_var_dict, shifts_per_tour_tuples, shifts_per_team_and_date_dict_tuples, shifts_per_disruption_tuples, shifts_dict_tuples, prob_lp)

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
        
        homes = []
        visitors = []
        original_dates = []
        game_dates = []
        proposed_dates = []
        reschedule = []

        
        for t in shifts_dict_tuples:
            idx = x_var_dict[str(t)]
            if round(x_variables[idx], 0) == 1:
                selected_sequence = t[5]
                for match in selected_sequence:
                    homes.append(match['game'][0])
                    visitors.append(match['game'][1])
                    original_dates.append(match['original_date'])
                    game_dates.append(match['game_date'])
                    proposed_dates.append(match['proposed_date'])
                    if match['original_date'] != match['proposed_date']:
                        reschedule.append(1)
                    else:
                        reschedule.append(0)

        # Create output dataframe
        output_df = pd.DataFrame({
            'home': homes,
            'visitor': visitors,
            'original_date': original_dates,
            'game_date': game_dates,
            'proposed_date': proposed_dates,
            'model_reschedule': reschedule,
        })
        output_df.to_csv("C:/Users/HP/Documents/Sports Analytics/Re Scheduling/code/models/tours sequence model/test.csv")
        quit(0)
        output_df
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
                    'game_date': row['game_date'],
                    'proposed_date': row['proposed_date']
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
                'game_date': game_date,
                #'proposed_date': original_date
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
