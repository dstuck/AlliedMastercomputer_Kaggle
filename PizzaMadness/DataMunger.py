from PizzaManager import *

def getTrainingData():
    df = pd.read_json('train.json')
    removeList = list(df.columns[map((lambda x:'at_retrieval' in x),df.columns)])
    removeList.append(u'giver_username_if_known')
    removeList.append(u'post_was_edited')
    removeList.append(u'requester_user_flair')
    df.drop(removeList,axis=1,inplace=True)
    return df


if __name__ == '__main__':
    print "hello"
    getTrainingData
