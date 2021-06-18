#data preprocessing
import pandas as pd
#produces a prediction model in the form of an ensemble of weak prediction models, typically decisioni
import matplotlib.pyplot as plt
# produces a prediction model in the form of an ensemble of weak prediction models, typically decisioni
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

loc = "C:/Users/Enspa/Documents/GitHub/FootballAnalytics/"
data = pd.read_csv(loc + "final_dataset.csv")

data_before = data[:667]
data_covid = data[668:]

n_matches = data_covid.shape[0]
n_features = data.shape[1] - 1
n_homewins = len(data_covid[data_covid.FTR == 'H'])

before_n_matches = data_before.shape[0]
bef_n_homewins = len(data_before[data_before.FTR == 'H'])

win_rate = (float(n_homewins) / (n_matches)) * 100

bef_win_rate = (float(bef_n_homewins)/ (before_n_matches) * 100)

print("Covid time home winrate: ", win_rate)
print("Before covid home winrate: ", bef_win_rate)

df = data[['HTGD','ATGD','HTP','ATP','DiffFormPts']]
ts = pd.plotting.scatter_matrix(df, figsize=(10,10))
##plt.show()


print(win_rate/bef_win_rate)

train_data = data[:1000]
test_data = data[1001:]

x_all = data.drop(['FTR'],1)
y_all = data['FTR']

print(list(x_all.columns))


cols = [['HTGD','ATGD','HTP','ATP']]

for col in cols:
    x_all[col] = scale(x_all[col])

x_all.HM1 = x_all.HM1.astype('str')
x_all.HM2 = x_all.HM2.astype('str')
x_all.HM3 = x_all.HM3.astype('str')
x_all.AM1 = x_all.AM1.astype('str')
x_all.AM2 = x_all.AM2.astype('str')
x_all.AM3 = x_all.AM3.astype('str')


def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)

        # Collect the revised columns
        output = output.join(col_data)

    return output

x_all = preprocess_features(x_all)

print(display(x_all.tail()))

X_train, X_test, y_train, y_test = train_test_split(x_all, y_all,
                                                    test_size = 100,
                                                    shuffle = False)


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)

    end = time()
    # Print and return results
    print
    "Made predictions in {:.4f} seconds.".format(end - start)

    return f1_score(target, y_pred, pos_label='H', average='micro'), sum(target == y_pred) / float(len(y_pred))

def predict(clf, X_train, y_train, X_test, y_test):

    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("Trained model in {:.4f} seconds".format(end-start))

    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, " - ", acc)

    f1, acc = predict_labels(clf, X_test, y_test)
    print(f1, "test ", acc)



clf_gnb = GaussianNB()
clf_A = LogisticRegression(random_state=42)
predict(clf_gnb, X_train, y_train, X_test, y_test)
print('')
predict(clf_A, X_train, y_train, X_test, y_test)

