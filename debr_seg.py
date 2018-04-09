import numpy as np
import pandas as pd
from sklearn import mixture, preprocessing, metrics, decomposition, cluster, externals
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from pylab import savefig
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
import dpcluster
import sys
from sys import argv
import time

global prev

global root
#'/home/uva-dsa1/Downloads/dVRK videos/Knot_Tying/kinematics/AllGestures'
root = '/home/uva-dsa1/Downloads/dVRK videos/'
filename = '/home/uva-dsa1/Downloads/Suturing fails/sut1.csv'
transitions = []
demonstrations = []
changepoints = None

def readData(dataFile):
    global prev
    prev = 0
    traj = []
    try:
        df = pd.read_csv(dataFile, delimiter = ',', low_memory = 'False')
        s = df[['field.runlevel', 'field.pos_d0', 'field.pos_d1', 'field.pos_d2', 'field.pos_d3', 'field.pos_d4', 'field.pos_d5']]
        s = np.array(s)
        for i in range(s.shape[0]):
            if s[i][0] == 3:
                traj.append(s[i,1:])
    except IOError:
        print "no file"
        return [], []
    traj = np.array(removeDuplicates(traj))
    print traj.shape
    externals.joblib.dump(traj[:,0:3], 'left_traj.p')
    externals.joblib.dump(traj[:,3:6], 'right_traj.p')
    return traj[:,0:3], traj[:,3:6]

def removeDuplicates(traj):
    new_traj = []
    for i in range(1,len(traj)):
        if (traj[i] == traj[i-1]).all():
            pass
        else:
            new_traj.append(traj[i])
    return new_traj

def generatePlots(dataFile):
    left, right = readData(dataFile)
    left_z = []
    right_z = []
    for i in range (left.shape[0]):
        left_z.append((left[i][0]**2 + left[i][1]**2 + left[i][2]**2)**0.5)
        right_z.append((right[i][0]**2 + right[i][1]**2 + right[i][2]**2)**0.5)
    graph_plot(left_z, right_z, figName = dataFile)


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

        traj = scaler.transform(traj)
        #demonstrations.append(traj)
        #transcripts.append(transcript)
        graph_plot(traj, traj, videoFile)
        clusters(transcriptFile, np.array(traj), np.array(transcript))


def clustersNDP(demonstrations = None):
    """
    First layer of clustering based on cartesian space
    """
    temporal_window = 2
    traj = demonstrations
    print demonstrations.shape
    traj = generate_transition_features(traj, temporal_window)
    bic =[]
    cv_types = ['full']
    n_components_range = range(6,7)
    n_splits = 2

    gmm = mixture.BayesianGaussianMixture(n_components = 6, covariance_type='full', max_iter = 10000, tol = 1e-7, random_state = 00)
    gmm.fit(traj)
    results = gmm.predict(traj)
    for i in range(len(results)-1):
        if results[i]!=results[i+1]:
            print i

def append_cp_array(cp):
    global changepoints
    changepoints = safe_concatenate(changepoints, cp)

def clusters(demonstrations = None):
    traj = demonstrations
    temporal_window = 2
    #traj = generate_transition_features(traj, temporal_window)
    print traj.shape

    cv_types = ['full']
    vdp = VDP(GaussianNIW(18))
    vdp.batch_learn(vdp.distr.sufficient_stats(traj))
    likelihoods = vdp.pseudo_resp(np.ascontiguousarray(traj))[0]
    real_clusters = 1
    cluster_s = vdp.cluster_sizes()
    total = np.sum(cluster_s)
    running_total = cluster_s[0]
    for i in range(1, len(vdp.cluster_sizes())):
        running_total = running_total + cluster_s[i]
        real_clusters = i + 1
        if running_total>0.95:
            break
    print len(set(cluster_s))
    score = 0
    cp_times = []
    prev = 0


def subClusters(data):

    global prev
    gmm = mixture.BayesianGaussianMixture(n_components = 3, covariance_type = 'diag', max_iter = 10000,tol = 1e-7, random_state = 00)
    gmm.fit(data)
    sub_results = gmm.predict(data)

    gmm = mixture.GaussianMixture(n_components = 3, covariance_type = 'full', max_iter = 10000,tol = 1e-5, random_state = 00)
    gmm.fit(data)
    sub_results = gmm.predict(data)
    for i in range(len(sub_results)-1):
        if sub_results[i]!=sub_results[1+i]:
            print "subClusters: {}".format(data[i][0])
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
    figName = figName.replace(".csv", ".png")

    plt.grid(False)
    y_label = "euclidean value"
    x_label = "steps in trajectory"
    y_true = np.transpose(np.array(y_true))
    y_pred = np.transpose(np.array(y_pred))
    plt.plot(y_true, 'b')
    plt.plot(y_pred, 'r')
    plt.xlabel(x_label)
    plt.ylabel(y_label,fontsize = 3)

    savefig(figName, dpi = 600, aspect = 'auto')
    plt.close()

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
    filename = filename.replace(".txt", ".csv")
    cp = pd.DataFrame(data = np.array(cp))
    cp.to_csv(filename, sep = ',')
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
    generatePlots('/home/uva-dsa1/Downloads/Suturing fails/debr1.csv')
    left_traj = np.array(externals.joblib.load('left_traj.p'))
    right_traj = np.array(externals.joblib.load('right_traj.p'))
    traj = np.concatenate((left_traj, right_traj), axis = 0)
    #clustersNDP(demonstrations = traj)
    clusters(demonstrations=traj)

if __name__ == '__main__':
    main()
