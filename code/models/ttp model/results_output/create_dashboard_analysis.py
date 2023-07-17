import pandas as pd

if __name__ == '__main__':
    names = pd.read_csv('schedule_type_names.csv')
    balance = pd.read_csv('balance_analysis.csv')
    schedules = pd.read_csv('all_schedules.csv')
    balance = pd.merge(balance, names, how='left', on='Schedule Type')
    schedules = pd.merge(schedules, names, how='left', on='Schedule Type')
    balance.to_csv('Balance.csv', index=False, encoding='utf-8 sig')
    schedules.to_csv('Schedules.csv', index=False, encoding='utf-8 sig')
