'''
Reffer to the README.txt file, on how to run this file.

Some Imporant Points - 
[I]
files Written
-------------
1. objectunique.csv
2. features.csv - with tot time and sessions (All 14 features and enrollment)
3. features - with result (All 14 features, result and enrollment)
4. nordata - normalised Data 16 columns
5. train.csv
6. test.csv 
7. pickle file, for the enrollment ID to per week click dictionary - perweek.txt
8. perweekframe.csv - csv file of enrollment ID and per week click record

[II]
Any refference to 'subset_train' after the preprocessing step in part 1 - subset_train = pd.read_csv('train.csv')
Any refference to 'subset_test'  after the preprocessing step in part 1 - subset_test = pd.read_csv('test.csv')
'''

import pandas as pd
import numpy as np

################################## Reading in the file, and Basic Processing

train_log= pd.read_csv('log_train.csv')

z0 = list(map(lambda x : x.split('T')[0], train_log['time'].tolist())) ## Splitting the 'time' which is a time stamp into 'date' and 'time' 
train_log['date'] = z0

z1 = list(map(lambda x : x.split('T')[1], train_log['time'].tolist()))
train_log['time'] = z1

################################# PART 1 ###################################
################### Developing the Feature Set - Part 1 ####################
################################## getting a count of the type 'event' type - 'access', 'navigate', 'page close' etc.

fil = train_log[['enrollment_id', 'event', 'time']].groupby(['enrollment_id',  'event']).agg('count') 
fil = fil['time']
fil = fil.unstack()


################################## Adding server and browser to the feature data set which includes the event types as seperate columns

feature1 =  train_log[['enrollment_id', 'source', 'time']].groupby(['enrollment_id',  'source']).agg('count')
feature1 = feature1['time']
feature1 = feature1.unstack()
feature1 = feature1.as_matrix()
feature1 = pd.DataFrame(feature1)
fil['server']=feature1[0]
fil['browser']=feature1[1]
fil[np.isnan(fil)]=0

################################## Adding chapter, sequential and unknown to the feature set

obj_data= pd.read_csv("object.csv")

## Obtaning a unique tuple of <module_id, course_id>, enabling a clean left outer join in this step.
## Repeating <module_id, course_id> in the object.csv file, causes a wrong join on the module_id, which is needed to find the type of module accessed - 'chapter', 'sequential'
x = obj_data.as_matrix()
a = []
for i in x:
    p = (i[0], i[1], i[2])
    a.append(p)
ans = set(a)

j = list(ans) 

g = pd.DataFrame(j)
g.columns = ['course_id', 'object', 'category']
g.to_csv('objectunique.csv', index = False) 
l = g[[1, 2]]

y = train_log.merge(l,on=['object'],how='left',indicator=True)
y = y.replace(np.nan, 'unknown')
z=y[['enrollment_id','category','time']].groupby(['enrollment_id','category']).agg('count')
z = z.unstack()
z=z['time']
z = z[['chapter', 'sequential', 'unknown']]
z = z.replace(np.nan, 0)
fil['chapter'] = z['chapter']
fil['unknown'] = z['unknown'] ## There are several modules accessed in the log data, with no record in the object.csv file, such modules have been classified together into a category called - 'unknown'
fil['sequential'] = z['sequential']

################################## Adding tot_time and session

'''
A session is the period of time when the user is active. 
We look at fifty minute time windows, from the start of 
a user's log record. Any click activity that the user 
performs is considered to be part of one session. If a 
user is inactive for fifty minutes, the session expires,
and any further logs for the user are part of the next 
session. The total time is the total length of all the 
sessions put together. It needs to be understood that a 
session may be less than or more than fifty minutes, but
a session expires when the gap between two consecutive 
click records for the user is greater than fifty minutes.
'''

from datetime import datetime
tt = pd.read_csv('log_train.csv')
tt.drop(tt.columns[[2,3,4]],axis=1,inplace=True)
train_mat = tt.as_matrix()


## Forming a dictionary, with the enrollment ID as the key, and the list of all click timestamps as values.
dict_main = {}
list_temp = []
i=1
prev=1L
for each in train_mat:
    i = each[0]
    if i == prev:
        x = (each[1])
        list_temp.append(x)
    else:
        dict_main[prev] = list_temp
        list_temp = []
        prev = i
        list_temp.append((each[1]))
dict_main[prev] = list_temp

## Checking for each user, for each click, if the click is within a fifty minute window, 
## and qualifies as within the current session. Or is aprt of the next session.
time_mat = []
for key in dict_main.keys():
    tot_time = 0
    session = 1
    start = dict_main[key][0]
    end = start
    for i in range(1,len(dict_main[key])):
        t2 = datetime.strptime(dict_main[key][i], "%Y-%m-%dT%H:%M:%S")
        t1 = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S")
        t0 = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S")
        diff = (t2-t1).seconds
        if diff < 3000:
            end = dict_main[key][i]
        else:
            session+=1
            tot_time+=(t1-t0).seconds
            start = dict_main[key][i]
            end = start
    tot_time+=(t1-t0).seconds
    tot_time = tot_time/60
    abc = (key,session,tot_time)
    #print("In",abc)
    time_mat.append(abc)

time_df = pd.DataFrame(time_mat)
time_df.columns = ["enrollment_id","session","tot_time"]

fil['tot_time'] = time_df['tot_time']
fil['session'] = time_df['session']
fil.to_csv('features.csv')

################################## writing result - 1 - dropout, 0 - completed


## From the truth_train.csv files - addinf the result coulmn
data = pd.read_csv('features.csv')
result = pd.read_csv('truth_train.csv', header = None)
result.columns = ['enrollment_id', 'result']
data['result'] = result['result']
data.to_csv('features.csv', index = False) ## Writing the 14 features with enrollment ID, and result to a file.


################################## Normalisation


## All the data is brought into a range of 0 - 1
from sklearn import preprocessing
norData=pd.DataFrame(preprocessing.normalize(features.iloc[:,1:15]))
colName = features.columns
colName=colName[1:15]
norData.columns=colName
norData['enrollment_id']=features['enrollment_id']
norData['result']=features['result']
norData.to_csv('nordata.csv', index = False) ## Writing the normalised data to a file.

################################## Subsetting the data

## Dividing the data into train and test sets
## The train dataset contains 80% of the total number of students whos data is made avaliable
## The remaning 20% of the data is in the test set.
subset_train = norData.sample(frac = 0.8)
subset_test = norData.loc[~norData.enrollment_id.isin(subset_train.enrollment_id)]

subset_train.to_csv('train.csv', index = False)
subset_test.to_csv('test.csv', index = False)

################### PreProcessing and Visualisation - Part 1 ####################

norData.corr() ## pair wise correlation between the features.

## Heat map for the pairwise correlation - plot inclded in report
import matplotlib.pyplot as plt
data = pd.read_csv('train.csv') 
data = data[range(0, 14)] # removing enrollment_id and result
names = data.columns.tolist()[:14]
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


## Performing PCA - reducing to 2 most significant eigen vectors
## The Graph for PCA is included in the report
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca = pca.fit_transform(norData)
e = pd.DataFrame(pca)
plt.scatter(e[0], e[1])

## Correlation graph for the pair os page_close and access - which are fairly highly correlated
## Graph did not give very useful insights, not included in the report
import matplotlib.pyplot as plt
vis1 = train[['page_close','access','result']]
plt1.scatter(vis1['page_close'],vis1['access'],c=vis1['result'],marker='s')
plt.xlabel("page close")
plt.xlabel("access")
plt.show()



######################## Building Models - Part 1 ########################
################################## naive bayse with all fields
noOfEnrollments = 24108
from __future__ import division
subset_train = pd.read_csv('train.csv')
subset_test = pd.read_csv('test.csv')
x = subset_train[range(0 , 14)]
y = subset_train['result']
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x,y) ## fitting the model on the train data
xtest = subset_test[range(0, 14)]
ytest = subset_test[range(15,16)]
ytest = ytest.as_matrix()
ytest = ytest.tolist()
pred = (model.predict(xtest)).tolist() ## predicting the output on the test data

## Finding the percentage accuracy of the model
comp = [1 if pred[i] == int(ytest[i][0]) else 0 for i in range(noOfEnrollments)]
print sum(comp)/len(ytest) * 100 # 67.61241081798574 - percentage accuracy

################################## decision trees with all fields
from sklearn import tree
noOfEnrollments = 24108
from __future__ import division
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data,train['result']) ## fitting the model on the train data
test=pd.read_csv('R1/test.csv')
test = test[range(0,14)]
ans = clf.predict(test) ## predicting the output on the test data
s = pd.Series(ans)
s = s.tolist()
o = subset_test['result'].tolist()

## Finding the percentage accuracy of the model
res = [1 if o[i] == s[i] else 0 for i in range(noOfEnrollments)]
tot = sum(res)
percAccuLogit = tot / noOfEnrollments * 100
print percAccuLogit # 77.53028040484486 - percentage accuracy

################################## logistic regresssion with all fields
noOfEnrollments = 24108
from sklearn.linear_model import LogisticRegression
from __future__ import division
model = LogisticRegression()
model.fit(subset_train[range(0, 14)], subset_train['result']) ## fitting the model on the train data
ans = model.predict(subset_test[range(0, 14)]) ## predicting the output on the test data
s = pd.Series(ans)
s = s.tolist()
o = subset_test['result'].tolist()

## Finding the percentage accuracy of the model
res = [1 if o[i] == s[i] else 0 for i in range(noOfEnrollments)]
tot = sum(res)
percAccuLogit = tot / noOfEnrollments * 100
print percAccuLogit # 79.7536087605774 - percentage accuracy

################################## KNN with all fields
noOfEnrollments = 24108
from sklearn.neighbors import KNeighborsClassifier
from __future__ import division
model = KNeighborsClassifier()
model.fit(subset_train[range(0, 14)], subset_train['result']) ## fitting the model on the train data
predicted = model.predict(subset_test[range(0, 14)]) ## predicting the output on the test data
s = predicted.tolist()

## Finding the percentage accuracy of the model
res = [1 if o[i] == s[i] else 0 for i in range(noOfEnrollments)]
tot = sum(res)
percAccuKnn = tot / noOfEnrollments * 100
print percAccuKnn # 80.55417288866767 - percentage accuracy

############################## Part 2 ###################################
################## Feature Extraction - Part 2 ##########################

################################## Forming the per week total click data frame for each user
from datetime import datetime
import pandas as pd
from __future__ import division

tt = pd.read_csv('log_train.csv')
tt.drop(tt.columns[[2,3,4]],axis=1,inplace=True)
train_mat = tt.as_matrix()

## Same as done in the eariler step to find session and total count
## dictionary with key as enrollment ID and value a list of timestamps of all clicks by the user
dict_main = {}
list_temp = []
i=1
prev=1L
for each in train_mat:
    i = each[0]
    if i == prev:
        x = (each[1])
        list_temp.append(x)
    else:
        dict_main[prev] = list_temp
        list_temp = []
        prev = i
        list_temp.append((each[1]))
dict_main[prev] = list_temp



idcourse = pd.read_csv('enrollment_train.csv')
date = pd.read_csv('date.csv')
idcourse = idcourse[['enrollment_id', 'course_id']]
y = idcourse.merge(date,on=['course_id'],how='left') ## joining for each enrollment ID the start and end date of the course they enrolled in 
start_matrix = y.as_matrix()

for i in start_matrix:
    i[2] = i[2] + 'T00:00:00'
    i[3] = i[3] + 'T00:00:00'

start_dict = {} 
for i in start_matrix:
    start_dict[i[0]] = list(i[1:])
    

## Finding the total number of weeks in all courses
for i in start_dict:
    st = start_dict[i][1]
    en = start_dict[i][2]
    st = datetime.strptime(st, "%Y-%m-%dT%H:%M:%S")
    en = datetime.strptime(en, "%Y-%m-%dT%H:%M:%S")
    diff = (en-st).total_seconds()
    weeks = diff / 604800
    weeks = int(round(weeks, 0))
    start_dict[i].append(weeks)

ans = {} # A dictionary with a key as enrollment ID, and a list as value, with the number of clicks in each of the week of the course.
## The list of values, is infact a dictionary with key as week number and value as number of clicks in that week
for i in dict_main:
    all_stamps = dict_main[i]
    x = start_dict[i]
    weeks = x[3]
    startday = datetime.strptime(x[1], "%Y-%m-%dT%H:%M:%S")
    perweek = {i+1 : 0 for i in range(weeks)}
    for z in all_stamps:
        z = datetime.strptime(z, "%Y-%m-%dT%H:%M:%S")
        diff = (z - startday).total_seconds()
        week = diff / 604800
        if week >= weeks:
            week = weeks
        else:
            week = int(week) + 1
        perweek[week] = perweek[week] + 1
    ans[i] = perweek


## A pickle file is used to store a data structure as it is in pyhton
## Storing the dictionary 'ans'
import pickle
with open('perweek.txt', 'wb') as handle:
  pickle.dump(ans, handle)
with open('perweek.txt', 'rb') as handle:
  bns = pickle.loads(handle.read())

ans = bns

## forming a data frame from the dictionary 'ans'
## each row contains, the enrollment Id, Week 1 clciks, Week 2 clicks, ..., Week n clicks
import numpy as np
ansmat = np.array([0, 0, 0, 0, 0])
for i in ans:
    j = ans[i]
    a = [i, j[1], j[2], j[3], j[4]]
    ansmat = np.vstack((ansmat, a))

perweekframe = pd.DataFrame()
perweekframe = pd.DataFrame(ansmat)
perweekframe.columns = [['enrollment_id','Week 1', 'Week 2', 'Week 3', 'Week 4']]
perweekframe = perweekframe[1:]

perweekframe.to_csv('perweekframe.csv', index = False)


############### PreProcessing and Visualisation - Part 2 #################
################################## Plotting graph for a particular user
import matplotlib.pyplot as plt

def plotforuser(enrollment_id):
    k = perweekframe[perweekframe['enrollment_id'] == enrollment_id][['Week 1', 'Week 2', 'Week 3', 'Week 4']]
    plot = k.plot(kind = 'bar')
    plot.set_xlabel('Weeks', fontsize=12)
    plot.set_ylabel('Total Clicks', fontsize=12)
    plt.show() ## bar graph of weekly number of interactions of the user

plotforuser(1) ## a non drop out user, plot in report
plotforuser(1579) ## a dropout user, plot in report


############### Building Models - Part 2 #################
################################## Modeling Week wise click data

from datetime import datetime
import pandas as pd
from __future__ import division

perweekframe = pd.read_csv('perweekframe.csv')

tt = pd.read_csv('train.csv')
train = perweekframe.loc[perweekframe.enrollment_id.isin(tt.enrollment_id)]
test = perweekframe.loc[~perweekframe.enrollment_id.isin(tt.enrollment_id)]

################################## naive bayse with four weeks
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
x = train[range(1 , 5)]
y = train['result']
model.fit(x,y) ## fitting the model
xtest = test[range(1,5)]
ytest = test['result']
ytest = ytest.as_matrix()
ytest = ytest.tolist()
pred = (model.predict(xtest)).tolist() ## Predicting the dropouts on the test set.

## Finding accuracy of the model
comp = [1 if pred[i] == int(ytest[i]) else 0 for i in range(0,len(ytest))]
print sum(comp)/len(ytest) * 100 # 84.6026215364 - percentage accuracy

################################## decision trees with four weeks
from sklearn import tree
from __future__ import division
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train[range(1,5)],train['result']) ## fitting the model
ans = clf.predict(test[range(1,5)]) ## Predicting the dropouts on the test set.
s = pd.Series(ans)
s = s.tolist()
o = test['result'].tolist()
res = [1 if o[i] == s[i] else 0 for i in range(len(ytest))]
tot = sum(res)

## Finding accuracy of the model
percAccuLogit = tot / len(ytest) * 100
print percAccuLogit # 83.7979094077 - percentage accuracy


############### Experiments and Results #################
############### From Part 1 Feature set #################
################################## Finding the coefficients of logitstic regression
subset_train = pd.read_csv('train.csv')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(subset_train[range(0, 14)], subset_train['result'])
ls = model.coef_.tolist() ## computing the coefficients of the logistic regression
import math
ms = list(map(lambda x: (math.exp(x) / (1 + math.exp(x))), ls[0])) ## finding the log odds of success
## It is found from the above analysis that video and navigate have a higher odds and chapter access has minimum odds

################################## Determining the chance of dropout for a user with and wihout a chapter access
log = pd.read_csv('/home/pratul/Documents/MOOCS/KDD data/nordata.csv')
zchap = log[log.chapter != 0]
q = zchap['result'].sum()
print q / zchap.shape[0] ## 61.30118430219057
nzchap = log[log.chapter == 0]
w = nzchap['result'].sum()
print w / nzchap.shape[0] ## 92.7932506607034

############### From Part 2 Feature set #################
################################## Finding the average number of clicks for dropouts and nondropouts across all four weeks

drop = pd.read_csv('/home/pratul/Documents/MOOCS/KDD data/train/truth_train.csv', header = None)
perweekframe = pd.read_csv('/home/pratul/Documents/MOOCS/KDD data/perweekframe.csv')
res = perweekframe.merge(drop, on = ['enrollment_id'], how = 'left') ## Merging the weekly clicks and enrollment ID with the final result

dr = res[res['result'] == 1] ### for dropuouts
p = []
x = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
tot = dr.shape[0]
print tot
for i in x:
    s = sum(dr[i].tolist())
    s = s / tot
    p.append(s)

ndr = res[res['result'] == 0] ### for non-droupouts
p1 = []
tot = ndr.shape[0]
for i in x:
    s = sum(ndr[i].tolist())
    print s
    s = s / tot
    p1.append(s)

droupout_avg_per_week = p # [12, 8, 6, 5] - average number of clicks for a dropout user across the weeks
nondroupout_avg_per_week = p1 # [43, 40, 46, 66] - the average number of clicks for a non dropout user across the weeks

x = pd.DataFrame({'Week 1' : [43], 'Week 2' : [40], 'Week 3' : [46], 'Week 4': [66]})
plot = x.plot(kind = 'bar')
plot.set_xlabel('Weeks', fontsize=12)
plot.set_ylabel('Total Clicks', fontsize=12)
plt.show() ## A bar plot which shows the average number of clicks for a non dropout user

labels = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
colors = ['yellow', 'purple', 'blue', 'red'] 
explode = (0, 0, 0, 0.05)
plt.pie(nondroupout_avg_per_week, # data
        explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        startangle=70       # starting angle
        )

plt.show() ## A pie chart showing the average number od clicks for a non dropout user

################################## Finding the coefficients of logitstic regression 
from sklearn.linear_model import LogisticRegression
tt = pd.read_csv('/home/pratul/Documents/MOOCS/KDD data/train.csv')
model = LogisticRegression()
train = perweekframe.loc[perweekframe.enrollment_id.isin(tt.enrollment_id)]
model.fit(train[range(1, 5)], tt['result'])
ls = model.coef_.tolist() ## computing the coefficients of the logistic regression
import math
ms = list(map(lambda x: (math.exp(x) / (1 + math.exp(x))), ls[0])) ## finding the log odds of success
## Week 2 seems to have a considerable effect on the result fllowed by Week 4.