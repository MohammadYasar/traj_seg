import numpy as np
import pandas as pd
from sklearn import decomposition, mixture, preprocessing, externals, covariance, ensemble, svm, gaussian_process, model_selection
#from pyemma import msm
from matplotlib import pyplot as plt

summary_traj = []

def loadData(vidFile = None, kinFile = None, transFile = None):
    kin = []
    trans = []
    try:
        df = pd.read_csv(transFile, header = None, delimiter = ' ')
        trans = df.values.tolist()
        newtrans = labelProcess(trans)
        df = pd.read_csv(kinFile, header = None, dtype = np.float64, delimiter = '     ')
        kin = df.values.tolist()
        kin = extractCart(kin)

    except IOError:
        print "no file"
        return [], []

    return newtrans, kin

def extractCart(kinData):
    kinData = np.array(kinData)
    new_s = np.zeros((kinData.shape[0], 6))
    for i, value in enumerate(kinData):
        for j in range(3):
            new_s[i][j] = value[j+38]
            new_s[i][j+3] = value[j+57]
    scaler = preprocessing.MinMaxScaler().fit(new_s)
    #externals.joblib.dump(scaler, 'scaler.p')
    #scaler = externals.joblib.load('scaler.p')

    new_s = scaler.transform(new_s)
    #new_s = preprocessing.normalize(new_s)
    mean, std, max, min = np.mean(new_s), np.std(new_s), np.amax(new_s), np.amin(new_s)
    #print "global mean: {} global std: {} global max: {} global min : {}".format(mean,std, max, min)
    return new_s

def labelProcess(newtrans):
    newtrans = np.array(newtrans)
    for i, value in enumerate(newtrans):
        newtrans[i][2] = value[2].replace('G', '')

    trans = np.zeros((newtrans.shape[0], newtrans.shape[1]-1))
    for i in range(newtrans.shape[0]):
        for j in range(newtrans.shape[1]-1):
            trans[i][j] = newtrans[i][j]

    return trans

def makeSegmentations(kinematics, transcripts,task):
    transcripts = np.sort(transcripts, axis = 0)
    seg_kinematics = []
    prev_seg = transcripts[0][transcripts.shape[1]-1]
    for i in range(transcripts.shape[0]):
        if transcripts[i][transcripts.shape[1]-1] != prev_seg:
            processSegments(np.array(seg_kinematics), prev_seg, task)
            prev_seg = transcripts[i][transcripts.shape[1]-1]
            seg_kinematics = []
        for n in range (int(transcripts[i][0]), int(transcripts[i][1])):
            seg_kinematics.append(kinematics[n])
    processSegments(np.array(seg_kinematics), prev_seg, task)

def processSegments(kinData, prev_seg, task):
    global summary_traj
    gesture = 'segmented_trajectories/{}G{}.p'.format(task,prev_seg)
    externals.joblib.dump(kinData, gesture)
    print kinData.shape
    _mean, _std, _max, _min = [], [], [], []
    _mean = np.mean(kinData, axis=0)
    _std = np.std(kinData, axis=0)
    _max = np.amax(kinData, axis = 0)
    _min = np.amin(kinData, axis = 0)
    temp_array = np.concatenate((np.concatenate((_mean, _std), axis = 0), np.concatenate((_max, _min), axis = 0)), axis = 0)
    temp_array = np.append(temp_array,prev_seg)

    summary_traj.append(temp_array)
    #print "prev_seg : {}, mean: {} std: {} max: {} min : {}".format(prev_seg,mean,std, max, min)

def loopFiles(root):
    alpha = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for i in range (len(alpha)):
        for j in range (1,6):
            taskPath = 'Suturing/kinematics/AllGestures/Suturing_{}00{}.txt'.format(alpha[i], j)
            taskFile = root + taskPath
            transcriptFile = root + 'Suturing/transcriptions/Suturing_{}00{}.txt'.format(alpha[i], j)
            videoFile = root + 'Suturing/video/Suturing_{}00{}.avi'.format(alpha[i], j)
            print taskFile
            trans, kin = loadData(vidFile = videoFile, kinFile = taskFile, transFile = transcriptFile)
            task = taskFile.replace('.txt','').split('/')[8]
            if len(trans)>0 and len(kin)>0:
                makeSegmentations(kin, trans, task)
    doProcessing()

def doProcessing():
    global summary_traj
    writeFile = '/home/uva-dsa1/Downloads/dVRK videos/Suturing/kinematics/AllGestures/summary.csv'
    cp = pd.DataFrame(data = np.array(summary_traj), columns = ['mean_x1', 'mean_y1', 'mean_z1', 'mean_x2', 'mean_y2', 'mean_z2', 'std_x1', 'std_y1', 'std_z1', 'std_x2', 'std_y2', 'std_z2', 'max_x1', 'max_y1', 'max_z1', 'max_x2', 'max_y2', 'max_z2', 'min_x1', 'min_y1', 'min_z1', 'min_x2', 'min_y2', 'min_z2', 'label'])
    cp.to_csv(writeFile, sep = ',')

def detectAnomaly(sampleFile):
    sampleTraj = externals.joblib.load(sampleFile)
    #markovAnomaly(sampleTraj, 2, 0.01)
    gpVerification(sampleTraj)

def getDistanceByPoint(data,model):
    distance = pd.Series()
    for i in range (0, len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance

def getTransitionMatrix(df):
    df = np.array(df)
    model = msm.estimate_markov_model(df,1)
    return model.transition_matrix

def markovAnomaly(df, windows_size, threshold):
    transition_matrix = getTransitionMatrix(df)
    real_threshold = threshold**windows_size
    df_anomaly = []
    for j in range (0, len(df)):
        if (j<windows_size):
            df_anomaly.append(0)
        else:
            sequence = df[j-window_size:j]
            sequence = sequence.reset_index(drop=True)
            df_anomaly.append(anomalyElement(sequence, real_threshold, transition_matrix))
    return df_anomaly

def elipticEnvelope(data):
    data_1 = data
    data_1[20] = data_1[20]*0.95
    outliers_fraction = 0.2
    envelope = covariance.EllipticEnvelope(contamination = outliers_fraction, random_state = 0)
    envelope.fit(data_1)
    df_class0 = pd.DataFrame(data_1)
    df_class0['deviation'] = envelope.decision_function(data_1)
    df_class0['anomaly'] = envelope.predict(data_1)
    print len(df_class0['anomaly'])
    time = np.zeros(len(df_class0['anomaly']))
    print len(time)
    for i in range(len(time)):
        time[i] = i
    fig, ax = plt.subplots()
    a = df_class0.loc[df_class0['anomaly']==1]
    print len(a)
    #ax.plot(df_class0['time_epoch'], df_class0['value'], color = 'blue')
    ax.scatter(time, df_class0['anomaly'], color = 'red')
    plt.show()

def gpVerification(data):
    X = np.array(data)
    y = np.zeros((X.shape[0], X.shape[1]))
    kernel = 1.0 * gaussian_process.kernels.RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + gaussian_process.kernels.WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    for i in range(X.shape[0]-1):
        y[i] = X[i+1]
    y[i] = X[i]
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    #gpr = gaussian_process.GaussianProcessRegressor(n_restarts_optimizer = 5, kernel = kernel)
    #gpr.fit(X,y)
    gpr = externals.joblib.load('gpr.p')
    gpr.predict(X)
    y[0]=y[0]*-.1

    print(gpr.log_marginal_likelihood(theta=None, eval_gradient=False))
    print gpr.score(X, y)
    #externals.joblib.dump(gpr,'gpr.p')


def main():
    root = '/home/uva-dsa1/Downloads/dVRK videos/'
    #loopFiles(root)
    detectAnomaly('/home/uva-dsa1/Downloads/ML/traj_seg/segmented_trajectories/Suturing_I005G3.0.p')
if __name__ == '__main__':
    main()
