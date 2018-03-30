import numpy as np
import pandas as pd
from sklearn import decomposition, mixture, preprocessing, externals, covariance, ensemble, svm, gaussian_process, model_selection
from matplotlib import pyplot as plt
from pylab import savefig

imageFile = None
summary_traj = []

def loadData(vidFile = None, kinFile = None, transFile = None):
    kin = []
    vel = []
    trans = []
    setImageFile(transFile)
    try:
        df = pd.read_csv(transFile, header = None, delimiter = ' ')
        trans = df.values.tolist()
        newtrans = labelProcess(trans)
        df = pd.read_csv(kinFile, header = None, dtype = np.float64, delimiter = '     ')
        kin = df.values.tolist()
        kin, vel = extractCart(kin)

    except IOError:
        print "no file"
        return [], [], []

    return newtrans, kin, vel

def setImageFile(transFile):
    global imageFile
    imageFile =transFile.replace('.txt', '.jpg')
    imageFile = imageFile.replace('/transcriptions', '')

def getImageFile():
    global imageFile
    return imageFile

def extractCart(kinData):
    kinData = np.array(kinData)
    new_s = np.zeros((kinData.shape[0], 6))
    new_v = np.zeros((kinData.shape[0],14))
    for i, value in enumerate(kinData):
        for j in range(3):
            new_s[i][j] = value[j+38]
            new_s[i][j+3] = value[j+57]
        for k in range(7):
            new_v[i][j] = value[k+50]
            new_v[i][j+7] = value[k+69]

    scaler = preprocessing.MinMaxScaler().fit(new_v)
    scaler1 = preprocessing.MinMaxScaler().fit(new_s)
    #externals.joblib.dump(scaler, 'scaler.p')
    #scaler = externals.joblib.load('scaler.p')
    #new_v = scaler.transform(new_v)
    new_s = scaler1.transform(new_s)
    #new_s = preprocessing.normalize(new_s)
    mean, std, max, min = np.mean(new_s), np.std(new_s), np.amax(new_s), np.amin(new_s)
    #print "global mean: {} global std: {} global max: {} global min : {}".format(mean,std, max, min)
    return new_s,new_v

def labelProcess(newtrans):
    newtrans = np.array(newtrans)
    for i, value in enumerate(newtrans):
        newtrans[i][2] = value[2].replace('G', '')

    trans = np.zeros((newtrans.shape[0], newtrans.shape[1]-1))
    for i in range(newtrans.shape[0]):
        for j in range(newtrans.shape[1]-1):
            trans[i][j] = newtrans[i][j]

    return trans

def makeSegmentations(kinematics, transcripts, vel, task):
    acc = getAcceleration(vel, transcripts)
    transcripts = np.sort(transcripts, axis = 0)
    seg_kinematics = []
    seg_vel = []
    prev_seg = transcripts[0][transcripts.shape[1]-1]
    for i in range(transcripts.shape[0]):
        if transcripts[i][transcripts.shape[1]-1] != prev_seg:
            processSegments(np.array(seg_kinematics),np.array(seg_vel), prev_seg, task)
            prev_seg = transcripts[i][transcripts.shape[1]-1]
            seg_kinematics = []
            seg_vel = []
        for n in range (int(transcripts[i][0]), int(transcripts[i][1])):
            seg_kinematics.append(kinematics[n])
            seg_vel.append(vel[n])
    processSegments(np.array(seg_kinematics), np.array(seg_vel), prev_seg, task)

def processSegments(kinData, vel, prev_seg, task):

    global summary_traj
    gesture = 'segmented_trajectories/{}G{}.p'.format(task,prev_seg)
    externals.joblib.dump(kinData, gesture)

    _mean, _std, _max, _min = [], [], [], []
    l_mean = np.mean(kinData[:,0:3])
    r_mean = np.mean(kinData[:,3:6])
    l_std = np.std(kinData[:,0:3])
    r_std = np.std(kinData[:,3:6])
    l_max = np.amax(kinData[:,0:3])
    r_max = np.amax(kinData[:,3:6])
    l_min = np.amin(kinData[:,0:3])
    r_min = np.amin(kinData[:,3:6])

    vl_mean = np.mean(vel[:,0:3])
    vlw_mean = np.mean(vel[:,3:6])
    vlg_mean = np.mean(vel[:,6])
    vr_mean = np.mean(vel[:,7:10])
    vrw_mean = np.mean(vel[:,10:13])
    vrg_mean = np.mean(vel[:,13])

    vl_max = np.amax(vel[:,0:3])
    vlw_max = np.amax(vel[:,3:6])
    vlg_max = np.amax(vel[:,6])
    vr_max = np.amax(vel[:,7:10])
    vrw_max = np.amax(vel[:,10:13])
    vrg_max = np.amax(vel[:,13])

    vl_min = np.amin(vel[:,0:3])
    vlw_min = np.amin(vel[:,3:6])
    vlg_min = np.amin(vel[:,6])
    vr_min = np.amin(vel[:,7:10])
    vrw_min = np.amin(vel[:,10:13])
    vrg_min = np.amin(vel[:,13])

    summary_traj.append([prev_seg, l_mean, r_mean, l_std, r_std, l_max, r_max, l_min, r_min, vl_mean, vr_mean, vl_max, vr_max, vl_min, vr_min])


def loopFiles(root):
    alpha = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    for i in range (len(alpha)):
        for j in range (1,6):
            taskPath = 'Suturing/kinematics/AllGestures/Suturing_{}00{}.txt'.format(alpha[i], j)
            taskFile = root + taskPath
            transcriptFile = root + 'Suturing/transcriptions/Suturing_{}00{}.txt'.format(alpha[i], j)
            videoFile = root + 'Suturing/video/Suturing_{}00{}.avi'.format(alpha[i], j)
            print taskFile
            trans, kin, vel = loadData(vidFile = videoFile, kinFile = taskFile, transFile = transcriptFile)
            task = taskFile.replace('.txt','').split('/')[8]
            if len(trans)>0 and len(kin)>0:
                makeSegmentations(kin, trans, vel, task)
    doProcessing()

def doProcessing():
    global summary_traj
    writeFile = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/Suturing/kinematics/AllGestures/summary.csv'
    cp = pd.DataFrame(data = np.array(summary_traj), columns = ['seg','l_mean', 'r_mean', 'l_std', 'r_std', 'l_max', 'r_max', 'l_min', 'r_min', 'vl_mean',  'vr_mean', 'vl_max', 'vr_max', 'vl_min', 'vr_min'])
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
    z = gpr.predict(X)
    #print z[0]
    #print y[0]*-1
    y[0]=y[0][3]*-1

    print gpr.score(X, y)
    #externals.joblib.dump(gpr,'gpr.p')

def getAcceleration (velData, transcripts):
    acceleration = np.zeros((velData.shape[0], 2))
    for i in range(1,velData.shape[0]):
        acceleration[i-1][0] = (np.mean(velData[i,0:3]) - np.mean(velData[i-1,0:3]))/(1./30.)
        acceleration[i-1][1] = (np.mean(velData[i,7:10]) - np.mean(velData[i-1,7:10]))/(1./30.)

    graph_plot(acceleration = acceleration, transcripts=transcripts)
    return acceleration

def graph_plot(acceleration, transcripts, root = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/'):
    figName = getImageFile().replace(root, '')
    figName = root + figName
    figName = figName.replace('/Suturing/', '/Suturing/acceleration/')
    plt.grid(False)
    y_label = "acceleration value"
    x_label = "steps in trajectory"
    plt.plot(acceleration[:,0], 'b')
    plt.plot(acceleration[:,1], 'r')
    plt.xlabel(x_label)
    plt.ylabel(y_label,fontsize = 3)
    for i in range(transcripts.shape[0]):
        plt.axvline(transcripts[i][0],color='k', linestyle='--')
        plt.text(transcripts[i][0],3.5, int(transcripts[i][2]), fontsize = 4)

    savefig(figName, dpi = 600, aspect = 'auto')
    plt.close()

def main():
    #root = '/home/uva-dsa1/Downloads/dVRK videos/'
    root = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/'
    loopFiles(root)
    #'/Users/mohammadsaminyasar/Downloads/Lfd/segmented_trajectories'
    #detectAnomaly('/Users/mohammadsaminyasar/Downloads/Lfd/segmented_trajectories/Suturing_B001G3.0.p') #Model trained on G3

if __name__ == '__main__':
    main()
