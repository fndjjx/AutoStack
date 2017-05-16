import sys
import traceback
from functools import reduce


import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score, make_scorer, accuracy_score,mutual_info_score,auc, roc_auc_score
from sklearn.feature_selection import RFECV

import autotune

class AutoStack():
    def __init__(self, df, target_label):
        self.y = df[target_label]
        self.x = df.drop(target_label, axis=1)
        self.x_col = self.x.columns
        self.base_clfs = [XGBClassifier, LogisticRegression, RandomForestClassifier]#, AdaBoostClassifier, ExtraTreesClassifier, GaussianProcessClassifier]
        self.trained_base_clfs = []
        self.stack_clf = XGBClassifier()
        self.stack_clf = SVC()
        self.masks = []

        kf = KFold(len(self.x.values), n_folds=3)
        
        self.x_train_list = []
        self.x_test_list = []
        self.y_train_list = []
        self.y_test_list = []
        for train_index, test_index in kf:
            self.x_train_list.append(self.x.values[train_index])
            self.x_test_list.append(self.x.values[test_index])
            self.y_train_list.append(self.y.values[train_index])
            self.y_test_list.append(self.y.values[test_index])

    def select_feature(self, x, y, df, clf):
        try:
            scorer = make_scorer(accuracy_score)
            selector = RFECV(clf, step=1, cv=10, scoring=scorer)
            selector = selector.fit(x, y)
            mask = selector.support_
            print(mask)
            return mask
        except Exception as err:
            ty, tv, tb = sys.exc_info()
            print("object info: {}".format(err))
            print('Error type: {}, Error information: {}'.format(ty, tv))
            print(''.join(traceback.format_tb(tb)))
            return len(df.columns)*[True]

    def get_best_para(self,clf_obj, x,y):
        at=autotune.ParameterTune(clf_obj,x,y)
        at.run(pop_num=100, cxpb=0.5, mutpb=0.3, gen_num=10)
        r=at.get_best(1)
        print(r)
        return r[0][1]



    def build_stack(self):
        all_clfs_result = []
        for clf_obj in self.base_clfs:
            single_clf_result = []
            clf = clf_obj()
            mask = self.select_feature(self.x.values, self.y.values, self.x, clf)
            #mask = len(self.x.columns)*[True]
            self.masks.append(mask)
            single_clfs = []
            for data_index in range(len(self.x_train_list)):
                x_df = pd.DataFrame(self.x_train_list[data_index], columns=self.x_col).loc[:, mask]
                x = x_df.values
                y = self.y_train_list[data_index]
                best_para = self.get_best_para(clf_obj, x,y)
                print("best")
                print(best_para)
                clf = clf_obj(**best_para)
                clf.fit(x, y)
                single_clfs.append(clf)
                x_test_df = pd.DataFrame(self.x_test_list[data_index], columns=self.x_col).loc[:, mask]
                x_test = x_test_df.values
                result = clf.predict_proba(x_test)[:,1]
                #result = clf.predict(x_test)
                single_clf_result.extend(result)
            all_clfs_result.append(single_clf_result) 
            self.trained_base_clfs.append(single_clfs)

        x_stack = np.array(all_clfs_result).T 
        print(x_stack.shape)
        print(x_stack)
        y_stack = list(self.y_test_list[0])
        for i in range(1, len(self.y_test_list)):
            y_stack.extend(self.y_test_list[i])
        print(y_stack)
        
        df1 = pd.DataFrame(x_stack, columns=["1","2", "3"])
        df2 = pd.DataFrame(y_stack, columns=["target"])
        #print(df1.shape)
        #print(df2.shape)
        df3 = pd.concat([df1,df2],axis=1)
        df3.to_csv("/tmp/123.csv",index=False)
        return self.stack_clf.fit(x_stack, y_stack), self.masks

#    def predict(self, test_df):
#        print(self.masks)
#        single_prediction = []
#        for clf_index in range(len(self.base_clfs)):
#            clf = self.base_clfs[clf_index]()
#            mask = self.masks[clf_index]
#            x_test = test_df.loc[:, mask].values
#            x_train = self.x.loc[:, mask].values
#            y_train = self.y.values
#            clf.fit(x_train, y_train)
#            single_prediction.append(clf.predict(x_test))
#        x_test_stack = np.array(single_prediction).T
#        return self.stack_clf.predict(x_test_stack)

    def predict(self, test_df):
        single_prediction = []

        for clf_index in range(len(self.trained_base_clfs)):
            mask = self.masks[clf_index]
            x_test = test_df.loc[:, mask].values
            x_train = self.x.loc[:, mask].values
            y_train = self.y.values
            single_clf_result = []
            for clf in self.trained_base_clfs[clf_index]:
                #lf.fit(x_train, y_train)
                single_clf_result.append(clf.predict_proba(x_test)[:,1])
                #single_clf_result.append(clf.predict(x_test))
            print("aa")
            print(single_clf_result)
            rr = reduce(lambda x,y:x+y,single_clf_result)
            rr = [i/len(single_clf_result) for i in rr]
            print(rr)
#            rr = list(map(f, rr))
            single_prediction.append(rr)
        x_test_stack = np.array(single_prediction).T
        return self.stack_clf.predict(x_test_stack)


def f(i):
    if i>=0.5:
        return 1
    else:
        return 0
        
        
        

if __name__ == "__main__":
    target_label = "Survived"
    df = pd.read_csv("/tmp/middle.csv")
    myas = AutoStack(df, target_label)
    clf, masks = myas.build_stack()
    
    test_df = pd.read_csv("/tmp/middle2.csv")
    r = myas.predict(test_df)
    print(r)
    df = pd.read_csv("/tmp/test_after_etl.csv")
    id = df["PassengerId"].values

    df = pd.DataFrame({"PassengerId":id, "Survived":r})
    df.to_csv("/tmp/submission.csv",index=False)

