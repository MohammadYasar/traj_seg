import numpy as np
import pandas as pd
import itertools
from sklearn import mixture, preprocessing, metrics, decomposition, cluster
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from pylab import savefig
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
import sys
from sys import argv
import time
global root
root = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/'
transitions = []
demonstrations = []
changepoints = None

def readData(dataFile, transcriptFile):
    try:
        df = pd.read_csv(dataFile, header = None, dtype = np.float64, delimiter = ',')
        s = df.values.tolist()
        df1 = pd.read_csv(transcriptFile, header = None, delimiter = ' ')
        s1 = df1.values.tolist()
        transcripts = []
        for value in s1:
            transcripts.append([value[0], value[1]])

        s = np.array(s)
        '''new_s = np.zeros((s.shape[0], s.shape[1]/2))
        for i in range (s.shape[0]):
            for j in range (s.shape[1]/2):
                new_s[i][j] = s[i][j+s.shape[1]/2]

        s = np.array(new_s)'''
        new_s = np.zeros((s.shape[0], 6))
        for i, value in enumerate(s):
            for j in range(3):#s.shape[1]/2):#, s.shape[1]):
                new_s[i][j] = value[j+38]
                new_s[i][j+3] = value[j+57]

        s1 = np.array(transcripts)

        s = np.array(s)
        print s.shape
        #s = np.array(s)
        s1 = np.array(transcripts)
        #s = s.reshape(1,-1)

    except IOError:
        print "no file"
        return [], []

    return s,s1

def generate_transition_features(trajectory, temporal_window):
    X_dimension = trajectory.shape[1]
    #print "X dimension", str(X_dimension)
    T = trajectory.shape[0]
    N = None
    for t in range(T - temporal_window):

    	n_t = make_transition_feature(trajectory, temporal_window, t)
    	N = safe_concatenate(N, n_t)
        #print N.shape
    return N

def make_transition_feature(matrix, temporal_window, index):
    result = None
    for i in range(temporal_window + 1):
    	result = safe_concatenate(result, reshape(matrix[index + i]), axis = 1)
    return result

def reshape(data):
    """
    Reshapes any 1-D np array with shape (N,) to (1,N).
    """
    return data.reshape(1, data.shape[0])

def safe_concatenate(X, W, axis = 0):
    if X is None:
    	return W
    else:
    	return np.concatenate((X, W), axis = axis)

def loadDemonstrations(dataFile, transcriptFile, videoFile):
    global root, demonstrations
    demonstrations = []
    transcripts = []
    traj, transcript = readData(dataFile, transcriptFile)
    print dataFile
    if len(traj) > 1:
        scaler = preprocessing.StandardScaler().fit(traj)
        joblib.dump(scaler, 'scaler.p')
        scaler = joblib.load('scaler.p')
        traj = scaler.transform(traj)
        #demonstrations.append(traj)
        #transcripts.append(transcript)
        graph_plot(traj, traj, videoFile)
        clusters(transcriptFile, np.array(traj), np.array(transcript))

def append_cp_array(cp):
    global changepoints
    changepoints = safe_concatenate(changepoints, cp)

def clusters(transcriptFile, demonstrations = None, transcripts = None):
    #for i in range (demonstrations.shape[0]):
    traj = demonstrations
    new_s = np.zeros((traj.shape[0], 6))
    for i, value in enumerate(traj):
        for j in range(3):#s.shape[1]/2):#, s.shape[1]):
            new_s[i][j] = value[j+38]
            new_s[i][j+3] = value[j+57]
    traj = new_s



    temporal_window = 2
    traj = generate_transition_features(traj, temporal_window)
    print traj.shape
    n_components_range = range(5,6)
    cv_types = ['full']

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.BayesianGaussianMixture(n_components = 20, covariance_type='full', max_iter = 10000, tol = 1e-5, random_state = 500)
            #gmm = cluster.AgglomerativeClustering(linkage = 'average', n_clusters = 6)
            start = time.time()
            results = gmm.fit(traj)
            end = time.time()
            #gmm.predict(traj[0].reshape(1,-1))
            print "time taken: {}".format(start-end)
            #gmm = mixture.DPGMM(n_components = 7, covariance_type='diag', n_iter = 10000, tol= 1e-4)
            #gmm = mixture.GaussianMixture(n_components=n_components, max_iter = 10000,covariance_type=cv_type,  tol = 1e-5, random_state = 500)
            results = gmm.predict(traj)
            best_gmm = gmm
    score = 0
    cp_times = []
    prev = 0
    #print "checking results"
    transition_points = []
    for i in range(len(results)-1):
        if (results[i] != results[i+1] and i-prev>=2*temporal_window):
            #print "previous:{} new:{}".format(prev+1, i)
            transition_points.append([prev+1, i])
            prev = i
            change_pt = []
            #change_pt.append(i)
            for i, value in enumerate(traj[i]):
                change_pt.append(value)
            append_cp_array(reshape(traj[i][:traj.shape[1]]))
            cp_times.append(change_pt)

            for trans in transcripts:
                if i == trans[0]  or i == trans[1] :
                    score +=1

    print score
    store_changepoints(transcriptFile, transition_points, cp_times)
    joblib.dump(best_gmm, 'best_gmm.p')
    best_gmm = joblib.load('best_gmm.p')


            #plot_clusters(traj,clusters)


def plot_clusters(traj, clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x,c=clusters,s=50)
    for i,j in centers:
        ax.scatter(i,j,s=50,c='red',marker='+')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(scatter)

    fig.show()
    plt.close
def generateData(rootPath):
    df = []
    s = []
    alpha = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for i in range (len(alpha)):
        for j in range (1,6):
            taskPath = 'Knot_Tying/kinematics/AllGestures/Knot_Tying_{}00{}.csv'.format(alpha[i], j)
            taskFile = rootPath + taskPath
            transcriptFile = rootPath + 'Knot_Tying/transcriptions/Knot_Tying_{}00{}.txt'.format(alpha[i], j)
            videoFile = rootPath + 'Knot_Tying/video/Knot_Tying_{}00{}.avi'.format(alpha[i], j)
            loadDemonstrations(taskFile, transcriptFile, videoFile)
    return j

def generateTranscript(transcriptFile):
    df = pd.read_csv(transcriptFile, header = None, delimiter = ' ')
    s = df.values.tolist()
    first_line = s[0][:]
    last_line = s[1][:]
    gestures = s[2][:]

def graph_plot(y_true, y_pred, figName = None):
    figName = figName.replace(".avi", ".png")
    y_true = np.transpose(np.array(y_true))
    y_pred = np.transpose(np.array(y_pred))
    no_graphs = 2
    plt.grid(True)
    fig = plt.figure()
    for i in range(no_graphs):
        subplot_num = "21{}".format(1+i)
        ax = fig.add_subplot(int(subplot_num), projection = '3d')
        y_label = "xyz : {}".format(1)
        x_label = "steps in trajectory".format(1)
        ax.plot(y_pred[0+3*i], y_pred[1+3*i], y_pred[2+3*i], label = '3d plot')

    savefig(figName, dpi = 300)
    plt.close
def plot_labels(y_true, y_pred, figName = None):
    y_true = np.transpose(np.array(y_true))
    y_pred = np.transpose(np.array(y_pred))
    #print y_true.shape
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


        plt.plot(y_pred[i], 'b')
        plt.plot(y_true[i], 'r')
        plt.xlabel(x_label)
        plt.ylabel(y_label,fontsize = 5)
        #if e_traj == '':
        #    plt.plot(e_traj[i][0:e_traj.shape[1]], 'g')

        savefig(figName, dpi = 300)
    plt.close()

def store_changepoints(filename, cp = None, cp_time = None):
    transitions = []
    filename = filename.replace(".txt", ".csv")
    cp = pd.DataFrame(data = np.array(cp))
    cp.to_csv(filename, sep = ',')
    transitions.append(cp_time)
    joblib.dump(cp_time, 'transitons.p')
    #print transitions
    generateClusters()

def generateClusters():
    super_clusters = []
    transitions = []
    transitions = np.array(joblib.load('transitons.p'))
    transitions = changepoints
    print transitions.shape
    gmm = mixture.GaussianMixture(n_components = 5, covariance_type='full', max_iter = 10000, tol= 1e-5)
    #print transitions[i].reshape(-1,1).shape
    gmm.fit(transitions)
    results = gmm.predict(transitions)
    super_clusters.append(results)
    for i in range(len(results)-1):
        if results[i]!=results[i+1]:
            print transitions[i][0]

    joblib.dump(super_clusters, 'super_clusters.p')

def main():
    global root
    videoFile = root + 'Knot_Tying/video/Knot_Tying_B002_capture1.avi'
    dataFile = root + 'Knot_Tying/kinematics/AllGestures/Knot_Tying_B002.csv'
    transcriptFile = root + 'Knot_Tying/transcriptions/Knot_Tying_B002.txt'
    generateData(root)
    #generateClusters()



if __name__ == '__main__':
    main()
