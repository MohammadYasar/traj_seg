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
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
global state_indices
from pylab import savefig
import random

global X
global y
global z
global traj_check
def loadData(sheet = None):
    global X
    global y
    global z
    base_dir = '/Users/mohammadsaminyasar/Documents/Lfd_data'
    #df = pd.read_excel('/Users/mohammadsaminyasar/Box Sync/Research Papers/imitation:reinforcement learning/Lfd_data/synthesized_data.xlsx', sheet)
    df = pd.read_csv('/Users/mohammadsaminyasar/Downloads/test1_andy.csv')
    X =  df[['field.pos0', 'field.pos1', 'field.pos2', 'field.ori0', 'field.ori1', 'field.ori2', 'field.ori3', 'field.ori4', 'field.ori5','field.ori6', 'field.ori7', 'field.ori8']]#
    # 'field.ori9'
    #, 'field.ori10', 'field.ori11', 'field.ori12', 'field.ori13', 'field.ori14', 'field.ori15', 'field.ori16', 'field.ori17']]
    y = df[['field.jpos_d0','field.jpos_d1','field.jpos_d2','field.jpos_d3','field.jpos_d4','field.jpos_d5','field.jpos_d6','field.jpos_d7']]
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
    #print X[19]

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
    #state_indices = [700, 1000, 2500, 6600, 10000, 10700]
    for j in range(len(state_indices) - 1):
        #print "state_indices[j] :{} , state_indices[j+1] : {}".format(state_indices[j], state_indices[j+1])
        #print "y_dif_1 :{} , y_diff_2 : {}" .format(y[state_indices[j]][2],y[state_indices[j+1]-1][2])
        for i in range(state_indices[j], state_indices[j+1]):
            temp_z.append(z[state_indices[j+1]-1] - z[i])

            temp_y.append(y[state_indices[j+1]-1] - y[i])
            temp_X.append(X[i])

    z = np.array(z)
    X = np.array(temp_X)

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
    #X_mean = np.mean(X, axis=0)
    #X_std = np.std(X, axis=0)
    #X = (X - X_mean) / X_std
    scaler = StandardScaler()
    scaler.fit(X)
    scaler = joblib.load('scaler.p')
    X = scaler.transform(X)
    checkData()

def checkData():
    global X, y, z
    for i, value in enumerate(X):
        for j in range(len(value)):

            if np.isnan(value[j]):
                print "isNan :{} :{} :{}" .format(i, j, value[j])
    computeGradient(y)

def resizeArray():
    global X, y, z
    temp_X, temp_y = X, y
    rf_min_mse = 1
    mlp_min_mse = 1
    rf_filename = 'best_RF.sav'
    mlp_filename = 'best_MLP.sav'
    rf_regr = RandomForestRegressor(n_estimators = 1000, max_depth = 10, random_state = 0)
    mlp_regr = MLPRegressor(solver = 'lbfgs',hidden_layer_sizes = (30, 3), activation='tanh',  max_iter = 1000, tol=0.00000001)
    rf_best_params = []
    mlp_best_params = []


    for i in range(1, 10):
        #print "previous shapes: {} : {}".format(X.shape, y.shape)
        X = np.resize(X,(X.shape[0]/i, X.shape[1]))
        y = np.resize(y, (y.shape[0]/i, y.shape[1]))
        scaler = joblib.load('scaler.p')
        X = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_regr.fit(X_train, y_train)
        mlp_regr.fit(X_train, y_train)
        rf_preds = rf_regr.predict(X_test)
        mlp_preds = mlp_regr.predict(X_test)
        rf_mse = mean_squared_error(rf_preds, y_test)
        mlp_mse = mean_squared_error(mlp_preds, y_test)
        if rf_mse<rf_min_mse:
            rf_min_mse = rf_mse
            joblib.dump(rf_regr, rf_filename)
            rf_best_params = X.shape
        if mlp_mse<mlp_min_mse:
            mlp_min_mse = mlp_mse
            joblib.dump(mlp_regr, mlp_filename)
            mlp_best_params= X.shape
        print "mse results for RF : {} : {} MLP : {} : {} for array shape : {}".format(rf_mse, rf_regr.score(X_test, y_test), mlp_mse, mlp_regr.score(X_test, y_test),X.shape)
        X, y = temp_X, temp_y
    print "rf_best_params : {} mlp_best_params : {}".format(rf_best_params, mlp_best_params)

def newSkill(num=0):
    global X
    global y
    global z
    X = np.array(X)
    y = np.array(y)

    temp_X = []
    temp_y = []
    for i in range(X.shape[0]):
        if i%num == 0:
            temp_X.append(X[i])
            temp_y.append(y[i])
    temp_X = np.array(temp_X)
    temp_y = np.array(temp_y)
    X = temp_X
    y = temp_y


def trainMLP():
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
    sweeping_layers = [1,1,3,3]
    sweeping_units = [300,30,30,300]

    swept = []
    min_err = +1000
    fold_count = 0
    outer_fold_scores = []
    best_params = []
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]

        X_test, y_test = X[test_index], y[test_index]

        fold_count = fold_count + 1
        avg_scores = []
        for i in range(len(sweeping_layers)):
            MLP = MLPRegressor(solver = 'lbfgs',hidden_layer_sizes = (sweeping_units[i], sweeping_layers[i]), activation='tanh',  max_iter = 1000, tol=0.00000001,)
            inner_cv = KFold(n_splits=5, random_state=None, shuffle=True)
            scores = []

            for t_train_index, t_test_index in inner_cv.split(X_train):
                #print "inner_cv : {},  : {}".format(sweeping_layers[i], sweeping_units[i])
                X_train_train, y_train_train = X[t_train_index], y[t_train_index]
                X_test_test, y_test_test = X[t_test_index], y[t_test_index]
                MLP.fit(X_train_train, y_train_train)
                preds = MLP.predict(X_test_test)
                mse = mean_squared_error(y_test_test, preds)
                abs_errors = np.abs(y_test_test-preds)
                abs_mean_err = np.mean(abs_errors, axis=0)

                #print abs_mean_err
                print mse
                scores.append(mse)


            #print np.mean(scores)
            avg_scores.append(np.mean(scores))
            swept.append([sweeping_layers[i], sweeping_units[i]])

            scores = np.array(scores)
            inner_figName = "MLP/InnerFold/MLP_inner_fold for params : {} in fold : {}".format(swept[i], fold_count)
            graph_plot(y_test_test, preds, inner_figName)
            print "Average mean for fold: {} for hidden Layers : {} and hidden units : {} is : {}" .format(fold_count, sweeping_layers[i], sweeping_units[i], np.mean(scores))
            if np.mean(scores)<min_err:
                min_err = np.mean(scores)
                filename = 'best_MLP.sav'
                joblib.dump(MLP, filename)
                best_params = [sweeping_units[i], sweeping_layers[i]]
                print "best model updated : {} :{}".format(best_params, min_err)
            else:
                print "best params unchanged: {} :{} : {}".format(best_params, min_err, np.mean(scores))
        avg_scores = np.array(avg_scores)
        max_index = np.argmax(avg_scores)
        #print "loading best model parameters : {}".format(best_params)
        MLP = joblib.load(filename)

        out_preds = MLP.predict(X_test)
        abs_errors = np.abs(y_test-out_preds)
        abs_mean_err = np.mean(abs_errors, axis=0)
        mse = mean_squared_error(out_preds, y_test)
        print "outer fold mse :{} ".format(mse)
        abs_errors = np.abs(y_test-out_preds)
        abs_mean_err = np.mean(abs_errors, axis=0)
        print "absolute mean error per joint : {}" .format(abs_mean_err)
        #print MLP.score(X_test, y_test)
        outer_fold_scores.append(mse)
        #graph_plot(y_test, out_preds)
        outer_figName = "MLP/OuterFOld/MLP_outer_fold for params : {} in fold : {}".format(best_params, fold_count)
        graph_plot(y_test, out_preds, outer_figName)
    for i, value in enumerate(outer_fold_scores):
        print "fold : {} score : {}" .format(i, value)
    print (np.mean(outer_fold_scores))
        #print "mean_square_error for trees: {} depth : {} is error: {}".format(sweeping_trees[i], sweeping_depth[i], err)



def trainRF():
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
    sweeping_trees = [100, 1000, 1000, 100, 100, 10]
    sweeping_depth = [ None, None ,10, 100, 10, None]
    fold_count = 0
    scores = []
    swept = []
    max_acc = 0
    min_err = 100
    outer_fold_scores = []
    print y.shape
    best_params = []
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        avg_scores = []
        inner_fold_scores = []
        fold_count = fold_count + 1
        #print ("X_train_index : {} : {}".format(train_index, test_index))

        for i in range(len(sweeping_trees)):
            regr = RandomForestRegressor(n_estimators=sweeping_trees[i], max_depth=sweeping_depth[i], random_state=0)
            inner_cv = KFold(n_splits=5, random_state=None, shuffle=True)
            scores = []

            for t_train_index, t_test_index in inner_cv.split(X_train):

                X_train_train, y_train_train = X[t_train_index], y[t_train_index]
                X_test_test, y_test_test = X[t_test_index], y[t_test_index]
                regr.fit(X_train_train, y_train_train)
                preds = regr.predict(X_test_test)
                err = mean_squared_error(y_test_test, preds)
                abs_errors = np.abs(y_test_test-preds)
                abs_mean_err = np.mean(abs_errors, axis=0)
                #print abs_mean_err
                #print regr.score(X_test_test, y_test_test)

                scores.append(err)
            scores = np.array(scores)
            #print np.mean(scores)
            avg_scores.append(np.mean(scores))
            inner_fold_scores.append(np.mean(scores))
            swept.append([sweeping_trees[i], sweeping_depth[i]])
            inner_figName = "RF/InnerFold/RF_inner_fold for params : {} in fold : {}".format(swept[i], fold_count)
            graph_plot(y_test_test, preds, inner_figName)
            print "Average mean for fold: {} for sweeping_trees : {} and sweeping_depth : {} is : {}" .format(fold_count, sweeping_trees[i], sweeping_depth[i], np.mean(scores))
            if np.mean(scores)<min_err:
                min_err = np.mean(scores)
                filename = 'best_RF.sav'
                joblib.dump(regr, filename)
                best_params = [sweeping_trees[i], sweeping_depth[i]]
                print "best model updated : {} :{}".format(best_params, min_err)
            else:
                print "best params unchanged: {} :{} : {}".format(best_params, min_err, np.mean(scores))
        avg_scores = np.array(avg_scores)
        max_index = np.argmax(avg_scores)
        #print "loading best model parameters : {}".format(best_params)
        regr = joblib.load(filename)
        out_preds = regr.predict(X_test)
        #print out_preds.shape
        #print r2_score(y_test, out_preds)
        abs_errors = np.abs(y_test-out_preds)
        abs_mean_err = np.mean(abs_errors, axis=0)
        print abs_mean_err
        print "outer fold score: {}".format(regr.score(X_test, y_test))
        outer_fold_scores.append(regr.score(X_test, y_test))
        #graph_plot(y_test, out_preds)
        outer_figName = "RF/OuterFold/RF_outer_fold for params : {} in fold : {}".format(best_params, fold_count)
        graph_plot(y_test, out_preds, outer_figName)
    for i, value in enumerate(outer_fold_scores):
        print "i :{} outer_fold_scores : {}" .format(i, outer_fold_scores[i])




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
    sweeping_trees = [100, 1000, 1000, 100, 100, 10]
    sweeping_depth = [ None, None ,10, 100, 10, None]
    rf_objects = ['100, None', '1000, None', '1000, 10', '100, 100', '100,10', '10, None']
    mlp_objects = ['1,300', '1,30', '3,30', '3,300']
    y_pos = np.arange(len(rf_objects))
    mlp_data = [0.0106609708773, 0.0107878968702, 0.000759878333244, 0.000874131521023]
    rf_data = [0.000193185617371, 0.00018660589931,0.000182798124034, 0.000212587769437, 0.000212919171616, 0.000247167028237] #rf
    rf = [0.00190901870561,0.000370657261405,0.000137586088156,0.000227826685005,0.00010133410455,0.000196728311414,0.000111824679016,0.000146051038976,0.00164574642416]
    rf = np.mean(rf)
    print "mean :{}".format(rf)
    plt.bar(y_pos, rf_data, 0.5, alpha=0.5)
    plt.xticks(y_pos, rf_objects)
    plt.ylabel('Average Squared Loss')
    plt.title('Perfromance of MLP')

    #plt.show()
def mean_absolute_percentage_error(y_true, y_pred):
    percentage_error = []
    per = []
    global y
    for i in range(len(y_true)):
        percentage_error.append(np.abs(100*(y_true[i] - y_pred[i]) / y_true[i]))
    for i in range(y_true.shape[1]):
        per.append(np.mean(percentage_error[:i]))
    for i in range(y_true.shape[0]):
        print "y_true: {}, y_test: {}, error: {}".format(y_true[i], y_pred[i], y_pred[i]-y_true[i])
    return per

def graph_plot(y_true, y_pred, figName = None):
    y_true = np.transpose(np.array(y_true))
    y_pred = np.transpose(np.array(y_pred))

    #e_traj = np.transpose(np.array(e_traj))
    no_graphs = y_true.shape[0]

    new_y_true= []
    new_y_pred= []
    #y_real = np.transpose(np.array(y_real))
    plt.grid(True)
    for i in range(no_graphs):
        y_label = "joint :{}" .format(i)
        x_label = "steps in trajectory"
        subplot_num = "81{}" .format(i+1)
        plt.subplot(int(subplot_num))


        plt.plot(y_pred[i][0:y_pred.shape[1]]/10, 'b')
        plt.plot(y_true[i][0:y_true.shape[1]]/10, 'r')
        plt.xlabel(x_label)
        plt.ylabel(y_label,fontsize = 5)
        #if e_traj == '':
        #    plt.plot(e_traj[i][0:e_traj.shape[1]], 'g')

        savefig(figName, dpi = 300)
    plt.close()

def loadModel(model = None):
    global traj_check
    Skill = 1
    figName = "MLP/General/GenTraj"
    filename = 'best_MLP.sav'
    if model == "rf":
        filename = 'best_rf.sav'
        figName = "RF/General/GenTraj"
    elif model == 'gpr':
        filename = 'best_GPR.sav'
    loadData("lfd_gen_alt")
    cleanData()
    invertData()

    if Skill>0:
        newSkill(Skill)
    global X
    X = np.asarray(X)
    global y
    regr = joblib.load(filename)
    preds = regr.predict(X)
    y = np.asarray(y)
    print (regr.score(X,y))
    print mean_squared_error(y, preds)
    abs_errors = np.abs(y-preds)
    abs_mean_err = np.mean(abs_errors, axis=0)
    print abs_mean_err
    traj_check = preds
    #e_traj = err_traj(y, preds)
    graph_plot(y, preds, figName)

def computeGradient(traj):
    old_diff = np.zeros(8)
    new_diff = []


    new_diff = np.gradient(traj)
    signage = np.sign(new_diff)
    signchange = ((np.roll(signage, 1) - signage) != 0).astype(float)

    graph_plot(signchange, signchange, "diff.png")

def err_traj(actual_traj, predicted_value):
    min_random_value = []
    max_random_value = []
    for i in range (actual_traj.shape[1]):
        if i ==2:
            min_random_value.append(0.005*min(actual_traj[:][i]))
            max_random_value.append(0.005*max(actual_traj[:][i]))
        else:
            min_random_value.append(min(actual_traj[:][i]))
            max_random_value.append(max(actual_traj[:][i]))
    temp_y = np.zeros((y.shape[0], y.shape[1]))
    for i in range(actual_traj.shape[0]):
        for j in range(actual_traj.shape[1]):
            l = actual_traj[i][j]
            temp_y[i][j] = l
            inject = random.randint(0,2)
            if inject == 1:
                actual_traj[i][j]= actual_traj[i][j] + 0.05*random.uniform(min_random_value[j], max_random_value[j])
    graph_plot(predicted_value, actual_traj, temp_y, "erroneous_trajectory")

    return actual_traj
def main():
    usage = "Usage: python attempt.py <mlp|rf|gpr> <1:loadModel|0:trainModel>"
    #myBoxPlot()
    try:
        script, model, mode = argv
    except:
        print "Error: missing parameters"
        print usage
        sys.exit(0)

    #loadData("lfd_init_alt")
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
