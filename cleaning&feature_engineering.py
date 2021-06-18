import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools

played = 39

loc = "C:/Users/Enspa/Documents/GitHub/FootballAnalytics/"

epl_1920 = pd.read_csv("https://www.football-data.co.uk/mmz4281/1920/E0.csv")

epl_1920 = epl_1920[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
epl_1920 = epl_1920.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})


data = pd.read_csv("https://www.football-data.co.uk/mmz4281/1920/E0.csv")
data2021 = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/E0.csv")
data1819 = pd.read_csv("https://www.football-data.co.uk/mmz4281/1819/E0.csv")

# Gets the goals scored agg arranged by teams and matchweek
def get_goals_scored(playing_stat):
    # Create a dictionary with team names as keys
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)

    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsScored[0] = 0
    # Aggregate to get uptil that point
    for i in range(2, 39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i - 1]
    return GoalsScored


# Gets the goals conceded agg arranged by teams and matchweek
def get_goals_conceded(playing_stat):
    # Create a dictionary with team names as keys
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)

    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsConceded[0] = 0
    # Aggregate to get uptil that point
    for i in range(2, 39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i - 1]
    return GoalsConceded


def get_gss(playing_stat):
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)

    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1

    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC

    return playing_stat

game_stats1819 = get_gss(data1819)
game_stats1920 = get_gss(data)
game_stats2021 = get_gss(data2021)
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2, 39):
        matchres_points[i] = matchres_points[i] + matchres_points[i - 1]

    matchres_points.insert(column=0, loc=0, value=[0 * i for i in range(20)])
    return matchres_points


def get_matchres(playing_stat):
    # Create a dictionary with team names as keys
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match result
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')

    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T


def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1

    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP

    return playing_stat

game_stats1819 = get_agg_points(game_stats1819)
game_stats1920 = get_agg_points(game_stats1920)
game_stats2021 = get_agg_points(game_stats2021)

games = 6


def get_form(game_stats, num):
    form = get_matchres(game_stats)
    form_final = form.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i - j]
            j += 1
    return form_final


def add_form(game_stats, num):
    form = get_form(game_stats, num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]

    j = num
    for i in range((num * 10), 380):
        ht = game_stats.iloc[i].HomeTeam
        at = game_stats.iloc[i].AwayTeam

        past = form.loc[ht][j]  # get past n results
        h.append(past[num - 1])  # 0 index is most recent

        past = form.loc[at][j]  # get past n results.
        a.append(past[num - 1])  # 0 index is most recent

        if ((i + 1) % 10) == 0:
            j = j + 1

    game_stats['HM' + str(num)] = h
    game_stats['AM' + str(num)] = a

    return game_stats



def add_form_df(neeper):
    neeper = add_form(neeper, 1)
    neeper = add_form(neeper, 2)
    neeper = add_form(neeper, 3)
    neeper = add_form(neeper, 4)
    neeper = add_form(neeper, 5)
    return neeper

playing_statistics_1 = add_form_df(game_stats1920)
playing_statistics_2021 = add_form_df(game_stats2021)
playing_statistics_1819 = add_form_df(game_stats1819)

print("jee", playing_statistics_2021)

def form_points(stats) -> object:
    j = 10
    ht_form_points = []
    at_form_point = []
    for i in range(0, 380):
        ht = stats.iloc[i].HomeTeam
        at = stats.iloc[i].AwayTeam
        ht_form = [stats.iloc[i].HM1, stats.iloc[i].HM2, stats.iloc[i].HM3, stats.iloc[i].HM4, stats.iloc[i].HM5]
        at_form = [stats.iloc[i].AM1, stats.iloc[i].AM2, stats.iloc[i].AM3, stats.iloc[i].AM4, stats.iloc[i].AM5]
        form_point = 0

        for i in range(0, len(ht_form)):
            if ht_form[i] == 'D':
                form_point += 1
            elif ht_form[i] == 'W':
                form_point += 3
            else:
                continue
        ht_form_points.append(form_point)


        ##stats['HTFP'].append(form_point)
        at_form_points = 0
        for i in range(0, len(at_form)):
            if at_form[i] == 'D':
                at_form_points += 1
            elif at_form[i] == 'W':
                at_form_points += 3
            else:
                continue
        at_form_point.append(at_form_points)

    stats['HTFP'] = ht_form_points
    stats['ATFP'] = at_form_point



    return stats

playing_statistics_1819 = form_points(playing_statistics_1819)
stats = form_points(playing_statistics_1)
playing_statistics_2021 = form_points(playing_statistics_2021)
print(playing_statistics_2021)

def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%Y').date()

stats.Date = stats.Date.apply(parse_date)
playing_statistics_2021.Date = playing_statistics_2021.Date.apply(parse_date)
playing_statistics_1819.Date = playing_statistics_1819.Date.apply(parse_date)

print(playing_statistics_1819)

'''
jee = (playing_statistics_1.loc[playing_statistics_1['HomeTeam'] == 'West Ham'])
away = playing_statistics_1.loc[playing_statistics_1['AwayTeam'] == 'West Ham']
frames = [jee, away]
played = pd.concat(frames)
played = played.sort_index()
last = played.tail(6)
print(last[['HomeTeam', 'FTR']])
'''
#def form_points()




cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3',
        'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5', 'HTFP', 'ATFP']
game_stats1920 = stats[cols]
game_stats1819 = playing_statistics_1819[cols]
game_stats2021 = playing_statistics_2021[cols]

def get_mw(stats: object):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    stats['MaW'] = MatchWeek
    return stats

game_stats1819 = get_mw(game_stats1819)
game_stats1920 = get_mw(game_stats1920)
game_stats2021 = get_mw(game_stats2021)



full_stats = pd.concat([game_stats1819, game_stats1920, game_stats2021], ignore_index= True)

def get_winstreak(stats):
    HTStreak = []
    ATStreak = []
    for i in range(len(stats)):
        string = stats.iloc[i].HM1 + stats.iloc[i].HM2 + stats.iloc[i].HM3
        at_string = stats.iloc[i].AM1 + stats.iloc[i].AM2 + stats.iloc[i].AM3
        if string == 'WWW':
            HTStreak.append(1)
        elif string == 'LLL':
            HTStreak.append(-1)
        else:
            HTStreak.append(0)
        if at_string == 'WWW':
            ATStreak.append(1)
        elif at_string == 'LLL':
            ATStreak.append(-1)
        else:
            ATStreak.append(0)

    stats['HTstreak'] = HTStreak
    stats['ATstreak'] = ATStreak
    return stats

playing_stat = get_winstreak(full_stats)


# Get Goal Difference
playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

# Diff in points
playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
playing_stat['DiffFormPts'] = playing_stat['HTFP'] - playing_stat['ATFP']

# Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
cols = ['HTGD','ATGD','DiffPts','DiffFormPts','HTP','ATP']
playing_stat.MaW = playing_stat.MaW.astype(float)

for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MaW

playing_stat_test = playing_stat[1100:]
print(playing_stat_test)

playing_stat.to_csv(loc + "final_dataset.csv")
