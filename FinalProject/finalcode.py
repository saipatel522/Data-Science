import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as se
import statsmodels
import statsmodels.api as sm
data2 = pd.read_csv('pah_wikp_combo.csv', dtype = object)
data2.head()
data2['Year'] = pd.DatetimeIndex(data2['Date']).year
tidy = pd.DataFrame(data2[['Year', 'State', 'Fatalities', 'Wounded', 'School']])
tidy['State'] = tidy['State'].replace('District of Columbia', 'D.C.')
tidy['State'] = tidy['State'].replace('District Of Columbia', 'D.C.')
tidy['State'] = tidy['State'].replace('IA', 'Iowa')
tidy['Wounded'] = tidy['Wounded'].fillna(0)
tidy['Fatalities'] = tidy['Fatalities'].fillna(0)
tidy.head()
year_arr = np.unique(tidy['Year'])
year = pd.DataFrame(columns = ['Year', 'Number_of_Shootings'], index = np.arange(0, 34))
year['Year'] = year_arr
i = 0
curr = 0
for y in tidy['Year']:
    if y == year_arr[i]:
        curr += 1
    else:
        year.at[i, 'Number_of_Shootings'] = curr
        curr = 0
        i += 1
year.at[33, 'Number_of_Shootings'] = curr
year.head()
plt.plot(year['Year'], year['Number_of_Shootings'], color = 'black')
plt.title('Number of School Shootings in the US from 1990 - 2023')
plt.xlabel('Years')
plt.ylabel('Number_of_Shootings')
year.head()
print('Each year the average amount of school shootings is ' + str(year['Number_of_Shootings'].mean()) + 'school shootings per year.')
state_map = dict()
for y in tidy['State']:
    if y in state_map:
        state_map[y] = state_map[y] + 1
    else:
        state_map[y] = 1
state_map
state = pd.DataFrame(list(state_map.items()), columns = ['State', 'Number_of_Shootings'])
state.head()
state[state['State'] == 'California']
plt.figure(figsize = (25, 10))
plt.barh(state['State'], state['Number_of_Shootings'], color = 'blue')
plt.title('Number of School Shootings in Each US State (Including Virgin Islands)')
plt.ylabel('US States')
plt.xlabel('Number_of_Shootings')
wound = pd.DataFrame(columns = ['Year', 'Wounded', 'Fatalities'], index = np.arange(0, 34))
i, j = 0, 0
wound['Year'] = year_arr
curr1 = 0
curr2 = 0
for y in tidy['Year']:
    if y == year_arr[i]:
        curr1 += int(tidy.at[j, 'Wounded'])
        curr2 += int(tidy.at[j, 'Fatalities'])
        j += 1
    else:
        wound.at[i, 'Wounded'] = curr1
        wound.at[i, 'Fatalities'] = curr2
        curr1 = 0
        curr2 = 0
        i += 1
wound.at[i, 'Wounded'] = curr1
wound.at[i, 'Fatalities'] = curr2
wound.head()
plt.plot(wound['Year'], wound['Wounded'], color = 'black')
plt.title('Number of Wounded From School Shootings in the US from 1990 - 2023')
plt.xlabel('Years')
plt.ylabel('Number of People Wounded')
plt.plot(wound['Year'], wound['Fatalities'], color = 'black')
plt.title('Number of Fatalities From School Shootings in the US from 1990 - 2023')
plt.xlabel('Years')
plt.ylabel('Number of Fatalities')
reg = linear_model.LinearRegression()
x_vals = [[x] for x in year['Year'].values]
y_vals = [[y] for y in year['Number_of_Shootings'].values]
lin_fit = reg.fit(x_vals, y_vals)
new_data = []
for x in year['Year'].values:
    new_data.append(lin_fit.predict(x.reshape(-1,1))[0][0])
year['Predicted Shootings'] = pd.Series(new_data, index = year.index)
plt.plot(year['Year'], year['Number_of_Shootings'], color='black')
plt.plot(year['Year'], year['Predicted Shootings'], color='red')
plt.xlabel("Year")
plt.ylabel("Number of School Shootings")
plt.title("Number of School Shootings from 1990-2023 with Linear Regression")
num = year['Number_of_Shootings'].to_numpy()
num = num.astype('float64')
year['Number_of_Shootings'] = num
stat1 = sm.formula.ols(formula= 'Number_of_Shootings ~ Year', data = year).fit()
stat1.summary()
tidy = tidy[['Year', 'State']]
tidy = tidy.groupby(['Year', 'State']).size()
tidy = tidy.reset_index()
tidy['Number_of_Shootings'] = tidy[0]
tidy = tidy.drop(0, 1)
stat2 = sm.formula.ols(formula = 'Number_of_Shootings ~ Year * State', data = tidy).fit()
stat2.summary()
print("F test value for Model 1:", stat1.fvalue)
print("F test value for Model 2:", stat2.fvalue)
mean = wound['Wounded'].mean()
mean = round(mean, 2)
print("The average amount of people wounded from a school shooting within the United States is approximately " + str(mean) + 
      " wounded per year.")
mean = wound['Fatalities'].mean()
mean = round(mean, 2)
print("The average amount of people killed from a school shooting within the United States is approximately " + str(mean) + 
      " killed per year.")
school_map = dict()
for s in tidy['School']:
    if s in school_map:
        school_map[s] = school_map[s] + 1
    else:
        school_map[s] = 1
school = pd.DataFrame(list(school_map.items()), columns = ['School', 'Number_of_Shootings'])
school = school.drop([4, 5])
school
plt.bar(school['School'], school['Number_of_Shootings'])
plt.title('Number of Shootings in the Different Types of School in the US')
plt.xlabel('Type of School')
plt.ylabel('Number of Shootings')