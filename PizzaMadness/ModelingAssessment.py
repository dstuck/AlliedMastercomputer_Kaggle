from PizzaManager import *

X_train = None
y_train = None
X_test  = None
scaling = None
cheat   = None

word = WordVecObj()

def initWords(df):
    print "Loading word2vec model. Could take a while."
    loadModel()
    print "Finished loading. Now loading EVecs."
    loadEVecs(df,numVecs=10)
    print "Finished loading EVecs."

def addFeatures(df):
    addTextLen(df)
    addPostHour(df)
    addWordVecFeatures(df)

def initModeling():
# Set up training data
    df = getTrainingData()
    addFeatures(df)
    X_train,y_train = getNumericTraining(df)
    scaling = sk.preprocessing.StandardScaler()
    X_train = scaling.fit_transform(X_train)
# Set up test data
    testdf = getTestData()
    addFeatures(testdf)
    X_test = getNumericTest(testdf)
    X_test = scaling.transform(X_test)
    cheat = getTestCheat()

def modelSVM(cval = 0.01):
    pizzaSVM = svm.SVC(kernel='linear', C=cval)
    scores = cross_validation.cross_val_score(pizzaSVM, X_train, y_train, cv=10,scoring='f1')
    pizzaSVM.fit(X_train,y_train)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (pizzaSVM2,scores)

def modelLog(cval = 0.01):
    pizzaLog = sk.linear_model.LogisticRegression(penalty='l2',C=cval)
    scores = cross_validation.cross_val_score(pizzaLog, X_train, y_train, cv=10,scoring='f1')
    pizzaLog.fit(X_train,y_train)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (pizzaLog,score)

def modelRandomTree(ntrees = 10):
    pizzaTree = sk.ensemble.RandomForestClassifier(n_estimators = ntrees)
    scores = cross_validation.cross_val_score(pizzaTree, X_train, y_train, cv=10,scoring='f1')
    pizzaTree.fit(X_train,y_train)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return (pizzaTree,scores)

def getPrediction(mlmodel,metrics=True):
    predicted = mlmodel.predict(X_train)
    if metrics:
        print(sk.metrics.confusion_matrix(y_train, predicted))
        print(sk.metrics.classification_report(y_train, predicted))
    return predicted



