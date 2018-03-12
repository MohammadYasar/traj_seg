import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import pickle
import csv
from sklearn.externals import joblib
import math
from sklearn.metrics import r2_score
import sys
from sys import argv


global X
global y
global z
global state_indices


def loadData(sheet = None):
    global X
    global y
    global z
    base_dir = '/Users/mohammadsaminyasar/Documents/Lfd_data'
    df = pd.read_excel('/Users/mohammadsaminyasar/Box Sync/Research Papers/imitation:reinforcement learning/Lfd_data/synthesized_data.xlsx', sheet)
    X =  df[['field.pos0', 'field.pos1', 'field.pos2', 'field.ori0', 'field.ori1', 'field.ori2', 'field.ori3', 'field.ori4', 'field.ori5','field.ori6', 'field.ori7', 'field.ori8']]#
    # 'field.ori9'
    #, 'field.ori10', 'field.ori11', 'field.ori12', 'field.ori13', 'field.ori14', 'field.ori15', 'field.ori16', 'field.ori17']]

    y = df[['field.jpos_d0','field.jpos_d1','field.jpos_d2','field.jpos_d3','field.jpos_d4','field.jpos_d5',]]


    #'field.jpos_d8', 'field.jpos_d9', 'field.jpos_d10', 'field.jpos_d11','field.jpos_d12', 'field.jpos_d13','field.jpos_d14', 'field.jpos_d15',]]
    #y = df[['field.pos_d0', 'field.pos_d1', 'field.pos_d2', 'field.pos_d3', 'field.pos_d4', 'field.pos_d5']]
    z = df[['field.pos_d0', 'field.pos_d1',	'field.pos_d2']]

    #state = df[['State']]
    #y = np.array(state)
    filename = 'traing_data.p'
    joblib.dump(X, filename)
    filename = 'testing_data.p'
    joblib.dump(y, filename)
    filename = 'desired_cartesians.p'
    joblib.dump(z, filename)
    y = np.array(y)
    X = np.array(X)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def cleanData():
    global X
    global y
    global z
    global state_indices
    data = False
    X = joblib.load('traing_data.p')
    y = joblib.load('testing_data.p')
    z = joblib.load('desired_cartesians.p')
    X=np.array(X)
    y=np.array(y)
    z=np.array(z)
    temp_X = []
    temp_y = []
    temp_z = []
    zero_count  = 0


    state_indices = []
    for i in range(X.shape[0] + 1):
        if (i%20==0):
            state_indices.append(i)

    temp_z = []
    temp_X = []
    temp_y = []
    for j in range(len(state_indices) - 1):
        #print "state_indices[j] :{} , state_indices[j+1] : {}".format(state_indices[j], state_indices[j+1])
        for i in range(state_indices[j], state_indices[j+1]):
            temp_z.append(z[state_indices[j+1]-1] - z[i])

            temp_y.append(y[state_indices[j+1]-1] - y[i])
            temp_X.append(X[i])
    z = np.array(z)
    X = np.array(temp_X)
    '''X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std'''
    #print z.shape
def invertData():
    global X, y, z
    temp_X = []
    temp_y = []
    temp_z = []
    for j in range(len(state_indices) - 1):
        for i in range(state_indices[j], state_indices[j+1]):
            temp_X.append(X[i, 3:])
            temp_z.append(z[state_indices[j+1]-1] - z[i])
    temp_X = np.array(temp_X)

    temp_z = np.array(temp_z)
    temp_X = np.concatenate([temp_z, temp_X], axis = 1)
    X = temp_X
    z = np.array(z)
    print X[19]
    #for i, value in enumerate (X):
    #    print "i : {} X[i] :{}, y[i] :{}".format(i, value, z[i])
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    checkData()
def checkData():
    global X, y, z
    for i, value in enumerate(X):
        for j in range(len(value)):

            if np.isnan(value[j]):
                print "isNan :{} :{} :{}" .format(i, j, value[j])

def trainMLP():
    global X
    X = np.asarray(X)
    global y
    global z
    global state_indices

    y = np.asarray(y)
    kf = KFold(n_splits=19)
    kf.get_n_splits(X)
    temp_x = []
    temp_y = []
    sweeping_layers = [1,1,3,3]#,1,2,2,2]
    sweeping_units = [30,300,30,300]#,300,300,300,300]

    swept = []
    max_acc = -1000
    fold_count = 0
    outer_fold_scores = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        fold_count = fold_count + 1
        avg_scores = []
        for i in range(len(sweeping_layers)):
            MLP = MLPRegressor(solver = 'lbfgs',hidden_layer_sizes = (sweeping_units[i], sweeping_layers[i]), activation='tanh',  max_iter = 1000, tol=0.00000001)
            inner_cv = KFold(n_splits=5, random_state=None, shuffle=True)
            scores = []
            best_params = []
            for t_train_index, t_test_index in inner_cv.split(X_train):
                print "inner_cv : {},  : {}".format(sweeping_layers[i], sweeping_units[i])
                X_train_train, y_train_train = X[t_train_index], y[t_train_index]
                X_test_test, y_test_test = X[t_test_index], y[t_test_index]
                MLP.fit(X_train_train, y_train_train)
                preds = MLP.predict(X_test_test)

                abs_errors = np.abs(y_test_test-preds)
                abs_mean_err = np.mean(abs_errors, axis=0)

                #print abs_mean_err
                print MLP.score(X_test_test, y_test_test)
                scores.append(MLP.score(X_test_test, y_test_test))
            scores = np.array(scores)
            #print np.mean(scores)
            avg_scores.append(np.mean(scores))
            swept.append([sweeping_layers[i], sweeping_units[i]])
            print "Average mean for fold: {} for hidden Layers : {} and hidden units : {} is : {}" .format(fold_count, sweeping_layers[i], sweeping_units[i], np.mean(scores))
            if np.mean(scores)>max_acc:
                max_acc = np.mean(scores)
                filename = 'best_MLP.sav'
                joblib.dump(MLP, filename)
                print "best model updated "
                best_params = [sweeping_units[i], sweeping_layers[i]]
        avg_scores = np.array(avg_scores)
        max_index = np.argmax(avg_scores)
        print "loading best model parameters : {}".format(best_params)
        MLP = joblib.load(filename)
        out_preds = MLP.predict(X_test)
        abs_errors = np.abs(y_test-out_preds)
        abs_mean_err = np.mean(abs_errors, axis=0)
        print abs_mean_err
        print MLP.score(X_test, y_test)
        outer_fold_scores.append(MLP.score(X_test, y_test))
    for i, value in enumerate(outer_fold_scores):
        print "fold : {} score : {}" .format(i, value)
        #print "mean_square_error for trees: {} depth : {} is error: {}".format(sweeping_trees[i], sweeping_depth[i], err)



def trainRF():
    global X
    X = np.asarray(X)
    global y
    global z
    global state_indices
    y = np.asarray(y)
    kf = KFold(n_splits=19)
    kf.get_n_splits(X)
    temp_x = []
    temp_y = []
    sweeping_trees = [100, 1000, 1000, 100, 100, 10]
    sweeping_depth = [ None, None ,10, 100, 10, None]
    fold_count = 0
    scores = []
    swept = []
    max_acc = 0

    outer_fold_scores = []
    print y.shape
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        avg_scores = []
        inner_fold_scores = []
        fold_count = fold_count + 1
        #print ("X_train_index : {} : {}".format(train_index, test_index))
        best_params = []
        for i in range(len(sweeping_trees)):
            regr = RandomForestRegressor(n_estimators=sweeping_trees[i], max_depth=sweeping_depth[i], random_state=0)
            inner_cv = KFold(n_splits=5, random_state=None, shuffle=True)
            scores = []

            for t_train_index, t_test_index in inner_cv.split(X_train):
                #print ("X_train_index : {} : {}".format(t_train_index, t_test_index))
                #print "inner_cv : {},  : {}".format(sweeping_trees[i], sweeping_depth[i])
                X_train_train, y_train_train = X[t_train_index], y[t_train_index]
                X_test_test, y_test_test = X[t_test_index], y[t_test_index]
                regr.fit(X_train_train, y_train_train)
                preds = regr.predict(X_test_test)
                abs_errors = np.abs(y_test_test-preds)
                abs_mean_err = np.mean(abs_errors, axis=0)
                #print abs_mean_err
                #print regr.score(X_test_test, y_test_test)
                scores.append(regr.score(X_test_test, y_test_test))
            scores = np.array(scores)
            #print np.mean(scores)
            avg_scores.append(np.mean(scores))
            inner_fold_scores.append(np.mean(scores))
            swept.append([sweeping_trees[i], sweeping_depth[i]])
            print "Average mean for fold: {} for sweeping_trees : {} and sweeping_depth : {} is : {}" .format(fold_count, sweeping_trees[i], sweeping_depth[i], np.mean(scores))
            if np.mean(scores)>max_acc:
                max_acc = np.mean(scores)
                filename = 'best_RF.sav'
                joblib.dump(regr, filename)
                best_params = [sweeping_trees[i], sweeping_depth[i]]
        avg_scores = np.array(avg_scores)
        max_index = np.argmax(avg_scores)
        print "loading best model parameters : {}".format(best_params)
        regr = joblib.load(filename)
        out_preds = regr.predict(X_test)
        print out_preds.shape
        #print r2_score(y_test, out_preds)
        abs_errors = np.abs(y_test-out_preds)
        abs_mean_err = np.mean(abs_errors, axis=0)
        print abs_mean_err
        print "outer fold score: {}".format(regr.score(X_test, y_test))
        outer_fold_scores.append(regr.score(X_test, y_test))
    for i, value in enumerate(outer_fold_scores):
        print "i :{} outer_fold_scores : {}" .format(i, outer_fold_scores[i])
        #print "mean_square_error for trees: {} depth : {} is error: {}".format(sweeping_trees[i], sweeping_depth[i], err)
        #print "mean_square_error for trees: {} depth : {} is error: {}".format(sweeping_trees[i], sweeping_depth[i], err)

        #print mean_absolute_percentage_error(y_test, preds)
        #graph_plot(y_test, preds)


    with open ('results.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(preds.shape[0]):
            preds[i]=preds[i]*3.142/180

            spamwriter.writerow(preds[i])

    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    #print(result)

def train_GPR():
    global X
    X = np.asarray(X)
    global y
    global z
    global state_indices
    y = np.asarray(y)
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    temp_x = []
    temp_y = []
    sweeping_trees = [1, 5, 10, 20, 50, 100]
    sweeping_depth = [ None, None ,10, 100, 10, None]
    fold_count = 0
    scores = []
    swept = []


    outer_fold_scores = []
    print y.shape
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        avg_scores = []
        inner_fold_scores = []
        fold_count = fold_count + 1
        max_acc = -1000
        #print ("X_train_index : {} : {}".format(train_index, test_index))
        best_params = []
        for i in range(len(sweeping_trees)):
            regr = GaussianProcessRegressor(n_restarts_optimizer = sweeping_trees[i])
            inner_cv = KFold(n_splits=5, random_state=None, shuffle=True)
            scores = []

            for t_train_index, t_test_index in inner_cv.split(X_train):
                #print ("X_train_index : {} : {}".format(t_train_index, t_test_index))
                #print "inner_cv : {},  : {}".format(sweeping_trees[i], sweeping_depth[i])
                X_train_train, y_train_train = X[t_train_index], y[t_train_index]
                X_test_test, y_test_test = X[t_test_index], y[t_test_index]
                regr.fit(X_train_train, y_train_train)
                preds = regr.predict(X_test_test)

                #abs_errors = np.abs(y_test_test-preds)
                #abs_mean_err = np.mean(abs_errors, axis=0)
                #print abs_mean_err
                #print regr.score(X_test_test, y_test_test)
                scores.append(regr.score(X_test_test, y_test_test))
            scores = np.array(scores)
            #print np.mean(scores)
            avg_scores.append(np.mean(scores))
            inner_fold_scores.append(np.mean(scores))
            swept.append([sweeping_trees[i], sweeping_depth[i]])
            print "Average mean for fold: {} for sweeping_trees : {} and sweeping_depth : {} is : {}" .format(fold_count, sweeping_trees[i], sweeping_depth[i], np.mean(scores))
            if np.mean(scores)>max_acc:
                max_acc = np.mean(scores)
                filename = 'best_GPR.sav'
                joblib.dump(regr, filename)
                best_params = [sweeping_trees[i], sweeping_depth[i]]
        avg_scores = np.array(avg_scores)
        max_index = np.argmax(avg_scores)
        print "loading best model parameters : {}".format(best_params)
        regr = joblib.load(filename)
        out_preds = regr.predict(X_test, return_cov=True)
        print out_preds
        out_preds = regr.predict(X_test)
        #print r2_score(y_test, out_preds)
        abs_errors = np.abs(y_test-out_preds)
        abs_mean_err = np.mean(abs_errors, axis=0)
        print abs_mean_err
        print "outer fold score: {}".format(regr.score(X_test, y_test))
        outer_fold_scores.append(regr.score(X_test, y_test))
    for i, value in enumerate(outer_fold_scores):
        print "i :{} outer_fold_scores : {}" .format(i, outer_fold_scores[i])

def myBoxPlot():
    sweeping_trees = [100, 1000, 1000]#, 100, 100, 10]
    sweeping_depth = [ None, None ,10]#, 100, 10, None]
    objects = ['100, None', '1000, None', '1000, 10', '100, 100', '100, 10', '10, None']

    y_pos = np.arange(len(objects))

    data = [5.26, 5.04, 62.21, 5.26, 65.96, 8.98]
    plt.bar(y_pos, data, 0.5, alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Average Squared Loss')
    plt.title('Perfromance of random_forest')

    plt.show()
def mean_absolute_percentage_error(y_true, y_pred):


    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    percentage_error = []
    per = []
    for i in range(len(y_true)):
        percentage_error.append(np.abs(100*(y_true[i] - y_pred[i]) / y_true[i]))
    for i in range(y_true.shape[1]):
        per.append(np.mean(percentage_error[:i]))
    for i in range(y_true.shape[0]):
        print "y_true: {}, y_test: {}, error: {}".format(y_true[i], y_pred[i], y_pred[i]-y_true[i])
    return per

def graph_plot(y_true, y_pred):
    y_true = np.transpose(y_true)
    y_pred = np.transpose(y_pred)
    new_y_true= []
    new_y_pred= []
    plt.xlabel('iteration')
    plt.ylabel('x_position')
    plt.title('x_position with time')
    plt.plot(y_true[0][0:1000],'r')
    plt.plot(y_pred[0][0:1000],'b')
    plt.grid(True)

    plt.show()
def loadModel(model = None):
    filename = 'best_MLP.sav'
    if model == "rf":
        filename = 'best_rf.sav'

    loadData("lfd_gen")
    cleanData()
    invertData()
    global X
    X = np.asarray(X)
    global y
    global z
    global state_indices

    regr = joblib.load(filename)
    preds = regr.predict(X)
    y = np.asarray(y)
    #for i, value in enumerate(preds):
    #    print "i :{} predicted :{} actual :{} difference : {}" .format(i, value, y[i], value - y[i])
    print (regr.score(X,y))
    print mean_squared_error(y, preds)

def main():
    usage = "Usage: python attempt.py <mlp|rf|gpr> <1:loadModel|0:trainModel>"
    try:
        script, model, mode = argv
    except:
        print "Error: missing parameters"
        print usage
        sys.exit(0)

    loadData("lfd_init")
    cleanData()
    invertData()

    if mode == "1":
        print "loading model"
        loadModel(model)
    else:
        if model.lower() == "mlp":
            trainMLP()
        elif model.lower() == "rf":
            trainRF()
        elif model.lower() == "gpr":
            train_GPR()


if __name__ == '__main__':
    main()
