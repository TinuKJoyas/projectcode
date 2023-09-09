import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

df_batting=pd.read_csv('D:/dissertation/dupli dataset/all_season_batt_card.csv') #reading the dataset(allseason batting card)
col_remove = df_batting.loc[:,['match_id','match_name','home_team','current_innings','season','away_team','city','innings_id','minutes', 'captain', 'link','minutes','runningScore','commentary','shortText','runningOver','name']] #removing unwanted colums from allseason batting card dataset
print('Before removing unwanted columns: {}'.format(df_batting.shape))
df_batting.drop(labels=col_remove, axis=1, inplace=True)
print('After removing unwanted columns: {}'.format(df_batting.shape))

batting=[]                                           
for x in range(len(df_batting)):
        batsman_a=df_batting.loc[[x]]
        batsman_res=batsman_a['fullName'].values # filtering the batsman based on fullname from df_batting dataframe
        batting.append(batsman_res[0])
print(len(batting))
batsman = []
[batsman.append(x) for x in batting if x not in batsman] # Removing the repeating batsman from the batting list
print(len(batsman))

batsman_ground=[]
for x in range(len(df_batting)):
        batsman_a=df_batting.loc[[x]]
        batsman_res=batsman_a['venue'].values #fetching the venues from df_batting and add it to the batsman_ground list
        batsman_ground.append(batsman_res[0])
print(len(batsman_ground))
batsman_stadium = []
[batsman_stadium.append(y) for y in batsman_ground if y not in batsman_stadium] # Removing the repeating venues from the batsman_ground list
print(len(batsman_stadium))

def averageRuns(runs, matches, notout):  #finding the average runs of a batsman based on stadium
    # Calculate number of
    # dismissals
    out = matches - notout;

    # check for 0 times out
    if (out == 0):
        return -1;

    # Calculate batting average
    avg = runs // out;

    return avg
batsman_data1=[]
for y in range(len(batsman)-1,-1,-1):    #Iterating through the batsman list in descending order to get the latest statistics of the player
    for z in range(len(batsman_stadium)-1,-1,-1): #Iterating through the stadium to find out batsman performance on each stadium
        batsman_data = {}
        count = 0
        matches = 0
        run=0
        sixes=0
        fours=0
        isNotOut=0
        ballsfaced = 0
        strikerate = 0
        player=batsman[y]
        for x in range(len(df_batting)-1,-1,-1): #iterating the df_batting data to get the batsman details
            batsman_a=df_batting.loc[[x]]
            name=batsman_a['fullName'].values[0]
            venue=batsman_a['venue'].values[0]
            if name==batsman[y] and venue == batsman_stadium[z]: # Evaluating the batsman and stadium
                six= batsman_a['sixes'].values[0]
                sixes+= six
                four = batsman_a['fours'].values[0]
                fours += four
                matches+=1
                temp_run=batsman_a['runs'].values[0]             #find the aggregate sum of batsman based on each stadium
                run+= temp_run
                ballsface = batsman_a['ballsFaced'].values[0]
                ballsfaced += ballsface
                if batsman_a['isNotOut'].values[0]==True:
                    isNotOut+=1
                strikerate = (run/ballsfaced)*100
                average = averageRuns(run, matches, isNotOut)
                batsman_data.update({'stadium':venue, 'player':player,'batsman_matches':matches,'isNotOut':isNotOut,'batsman_average':average,'score':run,'fours':fours,'sixes':sixes,'strikerate':strikerate}) #updating the batsman details based on the stadium
        batsman_data1.append(batsman_data)  # appending the batsman details based on stadium in to bastsman_data1 list
batsman_data1 = list(filter(None,batsman_data1)) #removing the null dictionary(all batsman are not played in every stadium so, there are null values possible)
df_bat = pd.DataFrame(batsman_data1) #converting the list of dictionary's to dataframe
df_bat.tail(50)
df_bowling=pd.read_csv('D:/dissertation/dupli dataset/all_season_bowl_card.csv') # reading the data set all season bowling card
col_remove = df_bowling.loc[:,['match_id','match_name','bowling_team','season','home_team','away_team','city','innings_id','foursConceded','sixesConceded', 'wides', 'noballs','captain','href','name']] # removing unwanted columns
print('Before removing unwanted columns: {}'.format(df_bowling.shape))
df_bowling.drop(labels=col_remove, axis=1, inplace=True)
print('After removing unwanted columns: {}'.format(df_bowling.shape))
df_bowling = df_bowling.fillna(0) # filling null values with zero
bowling=[]
for x in range(len(df_bowling)):
        b=df_bowling.loc[[x]]
        res1=b['fullName'].values  # filtering the bowler based on fullname from df_bowling datframe
        bowling.append(res1[0])

bowler = []
[bowler.append(x) for x in bowling if x not in bowler]  # Removing the repeating bowler from the bowling list
ground_bowler=[]
for x in range(len(df_bowling)):
        b=df_bowling.loc[[x]]          
        res1=b['venue'].values         # filtering the bowler based on venue from df_bowling datframe
        ground_bowler.append(res1[0])
print(len(ground_bowler))
bowler_stadium = []
[bowler_stadium.append(y) for y in ground_bowler if y not in bowler_stadium]  # Removing the repeating bowler from the ground_bowler
print(len(bowler_stadium))

def averageBowler(runs_conceded,bowler_wickets):   
    # Calculate batting average
    if bowler_wickets == 0:                                     #calculating the bowler average
        return -1
    avg = runs_conceded // bowler_wickets
    return avg
bowler_data1=[]
for y in range(len(bowler)-1,-1,-1):    #Iterating through the bowler list in descending order to get the latest statistics of the player
    for z in range(len(bowler_stadium)-1,-1,-1):  #Iterating through the stadium to find out bowler performance on each stadium
        bowler_data = {}
        bowler_matches = 0
        runs_conceded = 0
        bowler_wickets = 0
        bowler_economyRate = 0
        bowler_dots = 0
        over = 0
        # bowler_overs = 0
        temp_wicket = 0
        bowler_player=bowler[y]
#         print(player)
#         player = bowling[y]
        for x in range(len(df_bowling)-1,-1,-1):     #iterating the df_bowling data to get the bowler details
            bowler_a = df_bowling.loc[[x]]
            bowler_name = bowler_a['fullName'].values[0]
            bowler_venue = bowler_a['venue'].values[0]
            if bowler_name == bowler[y] and bowler_venue == bowler_stadium[z]:  # Evaluating the bowler and stadium
                bowler_matches += 1
                temp_wicket = bowler_a['wickets'].values[0]
                bowler_wickets += temp_wicket
                bowler_over =  bowler_a['overs'].values[0]
                over+= bowler_over
                runs_conceded += bowler_a['conceded'].values[0]
                bowler_economyRate += bowler_a['economyRate'].values[0] #find the aggregate sum of bowler based on each stadium
                bowler_dots += bowler_a['dots'].values[0]
                bowler_economy = bowler_economyRate / bowler_matches
                bowler_average = averageBowler(runs_conceded, bowler_wickets)
                bowler_data.update({'player': bowler_player, 'stadium':bowler_venue, 'bowler_matches': bowler_matches, 'bowler_average': bowler_average,'overs':over, 'conceded': runs_conceded,'economyRate': bowler_economy, 'dots': bowler_dots, 'wickets': bowler_wickets}) #updating the bowler details based on the stadium
        bowler_data1.append(bowler_data) # appending the bowler details based on stadium in to bowler_data1 list
bowler_data1 = list(filter(None,bowler_data1)) #removing the null dictionary(all bowler are not playing in every stadium so, there are null values possible)
df_bowl = pd.DataFrame(bowler_data1) #converting the list of dictionarys to dataframe

df_merged = df_bat.merge(df_bowl, how = 'outer', on = ['player','stadium']) #merging the dataframe of bowling and batting in to a common data frame called df_merged
df_merged = df_merged.fillna(0) #replacing null values with 0
matches = []
for merge in range(len(df_merged)):
    matche = 0
    match = df_merged.loc[[merge]]
    matche = match['batsman_matches'].values[0] + match['bowler_matches'].values[0]  #calculating the total mtches of a player in each stadium(since the allrounder players have the chance to bat and bowl in a stdium so,there is seperate matches column in batting card and bowling card)
    matches.append(matche) #appending the data in to a list
df = df_merged.assign(total_matches = matches) # insert this list matches in to df_merged dataframe
col_remove = df.loc[:,['bowler_matches','batsman_matches']] # remove the bowler_matches and batsman_matches from df dataframe
print('Before removing unwanted columns: {}'.format(df.shape))
df.drop(labels=col_remove, axis=1, inplace=True)
print('After removing unwanted columns: {}'.format(df.shape))

top_players = df.head(5)
scores = top_players['score']
players = top_players['player']
stadiums = top_players['stadium']
labels = [f'{player} ({stadium})' for player, stadium in zip(players, stadiums)]
plt.figure(figsize=(8, 8))
plt.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Player Score in Different Stadiums')
plt.show()

top_players = df.tail(5)
wickets = top_players['wickets']
players = top_players['player']
stadiums = top_players['stadium']
labels = [f'{player} ({stadium})' for player, stadium in zip(players, stadiums)]
plt.figure(figsize=(8, 8))
plt.pie(wickets, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Player Wickets in Different Stadiums')
plt.show()

top_players = df.head(30)
strikerate = top_players['strikerate']
player = top_players['player']

plt.figure(figsize=(10,5))
plt.bar(player, strikerate, color='blue')
plt.xlabel('Batsmen')
plt.ylabel('strikerate')
plt.title('Strike Rate')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure labels are not cut off
plt.show()

top_players = df.tail(70)
economyrate = top_players['economyRate']
player = top_players['player']

plt.figure(figsize=(10, 6))
plt.bar(player, economyrate, color='green')
plt.xlabel('Bowler')
plt.ylabel('economyrate')
plt.title('Economy Rate')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure labels are not cut off
plt.show()

df.head(60)
batting_average = 30
bowling_average = 25
df['included_in_squad'] = np.where(
    ((df['batsman_average'] > batting_average)& (df['score'] > 200))   |
    ((df['bowler_average'] < bowling_average) & (df['bowler_average'] > 0) & (df['overs'] >=8)),1,0)
print(df)
df.to_csv("updated_player_performance_data.csv", index=False)
df.head(60)
for feature in df.columns:
    if df[feature].dtype == 'object':
        df[feature] = pd.Categorical(df[feature]).codes
y=df['included_in_squad']
df.drop(columns=['included_in_squad'], axis=1,  inplace=True)
x=df
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=5)
print("Training set: {} and Test set: {}".format(X_train.shape, X_test.shape))

sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

decision_classifier = DecisionTreeClassifier()
decision_classifier.fit(X_train, y_train)
y_pred_decision = decision_classifier.predict(X_test)
accuracy_decision = accuracy_score(y_test, y_pred_decision)
print("Decision Tree Classifier Accuracy: {:.3f}".format(accuracy_decision))
print("Classification Report for Decision Tree Classifier:", classification_report(y_test, y_pred_decision))
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_decision), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {decision_classifier}')
plt.show()

random_classifier = RandomForestClassifier(n_estimators=10)
random_classifier.fit(X_train, y_train)
y_pred_random = random_classifier.predict(X_test)
accuracy_random = accuracy_score(y_test, y_pred_random)
print("accuracy = ",accuracy_random)
print("Random Forest Classifier Accuracy: {:.3f}".format(accuracy_random))
print("\nClassification Report for Random Forest Classifier:\n", classification_report(y_test, y_pred_random))
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_random), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {RandomForestClassifier}')
plt.show()

logistic_classifier = LogisticRegression(solver='liblinear')
logistic_classifier.fit(X_train, y_train)
y_pred_logistic = logistic_classifier.predict(X_test)

accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("accuracy=",accuracy_logistic)
print("Logistic Regression Classifier Accuracy: {:.3f}".format(accuracy_logistic))
print("Classification Report for Logistic Regression Classifier:", classification_report(y_test, y_pred_logistic))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_logistic), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {LogisticRegression}')
plt.show()

models = ['Decision Tree', 'Random Forest', 'Logistic Regression']
accuracies = [accuracy_decision, accuracy_random, accuracy_logistic]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1.0)  # Set the y-axis limit
plt.show()
cm = confusion_matrix(y_test, y_pred_decision)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree Classifier')
plt.show()





