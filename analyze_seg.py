import numpy as np
import pandas as pd
from sklearn import decomposition, mixture, preprocessing

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
    new_s = preprocessing.StandardScaler().fit(new_s).transform(new_s)

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

def makeSegmentations(kinematics, transcripts):
    transcripts = np.sort(transcripts, axis = 0)
    seg_kinematics = []
    prev_seg = transcripts[0][transcripts.shape[1]-1]
    for i in range(transcripts.shape[0]):
        if transcripts[i][transcripts.shape[1]-1] != prev_seg:
            print prev_seg
            processSegments(np.array(seg_kinematics), prev_seg)
            prev_seg = transcripts[i][transcripts.shape[1]-1]
            seg_kinematics = []
        for n in range (int(transcripts[i][0]), int(transcripts[i][1])):
            seg_kinematics.append(kinematics[n])
    processSegments(np.array(seg_kinematics), prev_seg)
    
def processSegments(kinData, prev_seg):
    global summary_traj
    mean, std, max, min = np.mean(kinData), np.std(kinData), np.amax(kinData), np.amin(kinData)
    summary_traj.append([prev_seg,mean,std, max, min])
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
            if len(trans)>0 and len(kin)>0:
                makeSegmentations(kin, trans)
    doProcessing()

def doProcessing():
    global summary_traj
    writeFile = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/Suturing/kinematics/AllGestures/summary.csv'

    print summary_traj
    cp = pd.DataFrame(data = np.array(summary_traj))
    cp.to_csv(writeFile, sep = ',')
def main():
    root = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/'
    loopFiles(root)
if __name__ == '__main__':
    main()
