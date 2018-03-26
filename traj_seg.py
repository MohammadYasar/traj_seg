import numpy as np
import pandas as pd
import itertools
from sklearn import mixture, preprocessing, metrics, decomposition
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
global prev
global root
#'/home/uva-dsa1/Downloads/dVRK videos/Knot_Tying/kinematics/AllGestures'
root = '/home/uva-dsa1/Downloads/dVRK videos/'
transitions = []
demonstrations = []
def readData(dataFile, transcriptFile):
    global prev
    prev = 0
    try:
        df = pd.read_csv(dataFile, header = None, dtype = np.float64, delimiter = '     ')
        s = df.values.tolist()
        df1 = pd.read_csv(transcriptFile, header = None, delimiter = ' ')
        s1 = df1.values.tolist()
        transcripts = []
        for value in s1:
            transcripts.append([value[0], value[1]])
        s = np.array(s)
        '''
        s = np.array(s)
        new_s = np.zeros((s.shape[0], s.shape[1]/2))
        for i in range (s.shape[0]):
            for j in range (s.shape[1]/2):
                new_s[i][j] = s[i][j+s.shape[1]/2]

        s = np.array(new_s)

        '''
        new_s = np.zeros((s.shape[0], 6))
        for i, value in enumerate(s):
            for j in range(3):#s.shape[1]/2):#, s.shape[1]):
                new_s[i][j] = value[j+38]
                new_s[i][j+3] = value[j+57]


        s1 = np.array(transcripts)

        s = np.array(new_s)
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
    print "X dimension", str(X_dimension)
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

        traj = scaler.transform(traj)
        #demonstrations.append(traj)
        #transcripts.append(transcript)
        graph_plot(traj, traj, videoFile)
        clusters(transcriptFile, np.array(traj), np.array(transcript))

def clusters(dataFile, demonstrations = None, transcripts = None):
    """
    First layer of clustering based on cartesian space
    """
    temporal_window = 2
    traj = demonstrations
    transcript = transcripts
    print demonstrations.shape
    traj = generate_transition_features(traj, temporal_window)
    bic =[]
    lowest_bic = 10000000
    cv_types = ['full']
    #print traj.shape
    n_components_range = range(6,7)
    #traj = hist
    n_splits = 2
    traj1, traj2 = np.split(traj,n_splits, axis = 1)

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.BayesianGaussianMixture(n_components = 15, covariance_type='full', max_iter = 10000, tol = 1e-7, random_state = 00)
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
    time_stamp = np.zeros((results.shape[0],1))
    for i in range(len(time_stamp)):
        time_stamp[i][0] = i
    results = results.reshape(-1,1)
    results = np.concatenate((results, time_stamp), axis = 1)
    new_segments = np.concatenate((results, traj), axis = 1)
    new_segments = np.sort(new_segments, axis = 0)
    current_label = new_segments[0][0]
    cluster_array = []
    for i in range(new_segments.shape[0]):
        if (new_segments[i][0]!=current_label):
            current_label = new_segments[i][0]
            subClusters(cluster_array)
            cluster_array = []

        else:
            cluster_array.append(new_segments[i][1:])

    #results = results.reshape(-1,1)
    #print "checking results {} {}" .format(results.shape, traj.shape)
    #traj = np.concatenate((traj, results), axis = 1)
    #gmm = mixture.GaussianMixture(n_components=5, max_iter = 10000,covariance_type='full',  tol = 1e-5, random_state = 00)
    #gmm.fit(traj)
    #results = gmm.predict(traj)
    '''transition_points = []
    for i in range(len(results)-1):
        if (results[i] != results[i+1] and i-prev>=2*temporal_window):
            #print "previous:{} new:{}".format(prev+1, i)
            transition_points.append([prev+1, i])
            prev = i
            change_pt = []
            change_pt.append(i)
            for i, value in enumerate(traj[i]):
                change_pt.append(value)

            cp_times.append(change_pt)

            for trans in transcripts:
                if i == trans[0]  or i == trans[1] :
                    score +=1

    print score
    store_changepoints(transcriptFile, transition_points, cp_times)
    joblib.dump(best_gmm, 'best_gmm.p')
    best_gmm = joblib.load('best_gmm.p')'''


            #plot_clusters(traj,clusters)

def subClusters(data):
    global prev
    gmm = mixture.BayesianGaussianMixture(n_components = 3, covariance_type = 'diag', max_iter = 10000,tol = 1e-7, random_state = 00)
    gmm.fit(data)
    sub_results = gmm.predict(data)

    for i in range(len(sub_results)-1):
        if sub_results[i]!=sub_results[1+i]:
            print "subClusters: {}".format([prev,data[i][0], sub_results[i]])
            prev = data[i][0]
def plot_clusters(traj, clusters):
    """
    Function not currently used, could be used for plotting clusters
    """
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
    """
    Loading the files of the task
    """
    df = []
    s = []
    alpha = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for i in range (len(alpha)):
        for j in range (1,6):
            taskPath = 'Knot_Tying/kinematics/AllGestures/Knot_Tying_{}00{}.txt'.format(alpha[i], j)
            taskFile = rootPath + taskPath
            transcriptFile = rootPath + 'Knot_Tying/transcriptions/Knot_Tying_{}00{}.txt'.format(alpha[i], j)
            videoFile = rootPath + 'Knot_Tying/video/Knot_Tying_{}00{}.avi'.format(alpha[i], j)
            loadDemonstrations(taskFile, transcriptFile, videoFile)
    return j

def generateTranscript(transcriptFile):
    """
    Generates the transcripts for the given task
    """
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
    print y_true.shape
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
    """
    Storing the changepoints of the first layer
    """

    filename = filename.replace(".txt", ".csv")
    filename =filename
    cp = pd.DataFrame(np.array(cp))
    cp.to_csv(filename, sep='\t', encoding='utf-8')
    transitions = []
    transitions.append(cp_time)
    joblib.dump(cp_time, 'transitons.p')
    #print transitions
    generateClusters()

def generateClusters():
    """
    Second layer of clustering
    """
    super_clusters = []
    transitions = []
    transitions = np.array(joblib.load('transitons.p'))
    print transitions.shape
    gmm = mixture.GaussianMixture(n_components = 5, covariance_type='diag', max_iter = 10000)
    gmm.fit(transitions)
    results = gmm.predict(transitions)
    super_clusters.append(results)
    for i in range(len(results)-1):
        if results[i]!=results[i+1]:
            print "{} ".format(transitions[i][0])

    joblib.dump(super_clusters, 'super_clusters.p')


def main():
    global root
    videoFile = root + 'Knot_Tying/video/Knot_Tying_B002_capture1.avi'
    dataFile = root + 'Knot_Tying/kinematics/AllGestures/Knot_Tying_B002.csv'
    transcriptFile = root + 'Knot_Tying/transcriptions/Knot_Tying_B002.txt'
    generateData(root)


if __name__ == '__main__':
    main()
