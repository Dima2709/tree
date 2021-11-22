import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
from sklearn import tree

df = pd.read_csv('StudentsPerformance.csv')

le_gender = le()
le_race_ethnicity = le()
le_parental_level_of_education = le()
le_test_preparation_course = le()

df['le_gender'] = le_gender.fit_transform(df['gender'])
df['le_race_ethnicity'] = le_race_ethnicity.fit_transform(df['race/ethnicity'])
df['le_parental_level_of_education'] = le_parental_level_of_education.fit_transform(df['parental level of education'])
df['le_test_preparation_course'] = le_test_preparation_course.fit_transform(df['test preparation course'])

df = df.drop('gender',1)
df = df.drop('race/ethnicity',1)
df = df.drop('parental level of education',1)
df = df.drop('lunch',1)
df = df.drop('test preparation course',1)

mass = []
for i in range(len(df)):
    mass1 = []
    mass1.append(df['math score'][i])
    mass1.append(df['reading score'][i])
    mass1.append(df['writing score'][i])
    mass.append(int(sum(mass1)/len(mass1)))

df['score'] = mass

df1 = df.drop('math score',1)
df1 = df1.drop('reading score',1)
df1 = df1.drop('writing score',1)

model = tree.DecisionTreeClassifier()
model.fit(df1.drop('score',1), df1['score'])

print(model.predict([[0,1,2,0]]))