import pandas as pd
from sklearn import tree

# Read the CSV file into a pandas dataframe

df = pd.read_csv("PastHires.csv")

d1 = {"Y": 1, "N": 0}

df["Hired"] = df["Hired"].map(d1)
df["Employed?"] = df["Employed?"].map(d1)
df["Top-tier school"] = df["Top-tier school"].map(d1)
df["Interned"] = df["Interned"].map(d1)

d2 = {"BS": 0, "MS": 1, "PhD": 2}
df["Level of Education"] = df["Level of Education"].map(d2)

print(df)

# Create a decision tree classifier and fit it to the data

features = list(df.columns[:6])

X = df[features]
y = df["Hired"]

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

# Interact with the user to get their information

job1 = input("Are you employed? (Y/N): ").upper()
job2 = input("Did you graduate from a top-tier school? (Y/N): ").upper()
job3 = input("Did you intern during college? (Y/N): ").upper()
job4 = input("What is your level of education? (BS/MS/PhD): ")
job5 = input("How many years of experience do you have? (0-20): ").upper()
job6 = input("What is your number of previous employers? (0-20): ").upper()

# convert the user's input into the appropriate format for prediction

job1 = d1[job1]
job2 = d1[job2]
job3 = d1[job3]
job4 = d2[job4]
job5 = int(job5)
job6 = int(job6)

# create a new dataframe with the user's information

new_person = pd.DataFrame({
    "Years Experience": [job5],
    "Employed?": [job1],
    "Previous employers": [job6],
    "Level of Education": [job4],
    "Top-tier school": [job2],
    "Interned": [job3]
})

# create a list to store the user's information



print(new_person)

# Make a prediction using the decision tree classifier

prediction = clf.predict(new_person)

print(prediction)

if prediction[0] == 1:
    print("Congratulations! You are likely to be hired.")
else:
    print("Sorry, you are not likely to be hired.")




