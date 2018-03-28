import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from sys import argv
<<<<<<< HEAD
from sklearn import preprocessing, mixture
=======
from sklearn.mixture import GaussianMixture
from sklearn import mixture
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
from sklearn.cluster import KMeans
import itertools
#from scipy import linalg
import matplotlib as mpl
from sklearn.externals import joblib
import pickle
from pylab import savefig
<<<<<<< HEAD
import pandas as pd


global root

root = '/home/uva-dsa1/Downloads/dVRK videos/'
=======
from skimage.measure import compare_ssim as ssim
import pandas as pd
from skimage.feature import hog
from skimage import data, exposure

global root
root = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/Suturing/'
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
homingFile = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/homing.mov'

def loadImage(file = None):
    imageFile = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/Knot_Tying/video/Knot_Tying_B001_capture1.avi'
    imageFile = homingFile #homing video
    cap = cv2.VideoCapture(imageFile)
    count = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(2000000)

    ret, frame = cap.read()
    while(ret):
        ret, frame = cap.read()
        if count%50 ==0:
            cv2.imwrite("/Users/mohammadsaminyasar/Downloads/Lfd/JIGSAWS/imgR_2/actual/frame%d.jpg" % count, frame)     # save frame as JPEG file
        count += 1

    cap.release()
    cv2.destroyAllWindows()


def bgsegm(backgroundFile = None):
    print "bgsegm"
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(2000000)
    cap = cv2.VideoCapture(backgroundFile)
    bg_array = []
<<<<<<< HEAD
    count = 0
=======
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
    ret, frame = cap.read()
    while (ret):
        ret, frame = cap.read()
        if (ret == True):
<<<<<<< HEAD
            count = count + 1
=======
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
            #print "blur"
            #frame = cv2.GaussianBlur(frame,(3,3),0)
            #fgmask = fgbg.apply(frame)
            #fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2RGB)
            '''frame = frame.reshape((-1,3))
            Z = np.float32(frame)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 8
            ret1, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((frame.shape))'''
<<<<<<< HEAD


             # Crop image


            # Display cropped image
            #cv2.imshow("Image", frame)

            if (count%1==0):
                bg_array.append(frame)
=======
            #frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            bg_array.append(frame)
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
        else:
            return np.array(bg_array)

def ECR(frame, prev_frame, width, height, crop=False, dilate_rate = 5):
    safe_div = lambda x,y: 0 if y == 0 else x / y
    if crop:
        startY = int(height * 0.3)
        endY = int(height * 0.8)
        startX = int(width * 0.3)
        endX = int(width * 0.8)
        frame = frame[startY:endY, startX:endX]
        prev_frame = prev_frame[startY:endY, startX:endX]

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray_image, 0, 200)
    plt.imshow(edge,cmap = 'gray')
    print "gray"
    dilated = cv2.dilate(edge, np.ones((dilate_rate, dilate_rate)))
    inverted = (255 - dilated)
    gray_image2 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    edge2 = cv2.Canny(gray_image2, 0, 200)
    dilated2 = cv2.dilate(edge2, np.ones((dilate_rate, dilate_rate)))
    inverted2 = (255 - dilated2)
    log_and1 = (edge2 & inverted)
    log_and2 = (edge & inverted2)
    pixels_sum_new = np.sum(edge)
    pixels_sum_old = np.sum(edge2)
    out_pixels = np.sum(log_and1)
    in_pixels = np.sum(log_and2)
    plt.show()
    plt.close()
    return max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))

    # return max(safe_div(float(in_pixels),float(pixels_sum_new)), safe_div(float(out_pixels),float(pixels_sum_old)))

def play_Video():
    imageFile = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/Suturing/video/Suturing_B001_capture1.avi'
    cap =cv2.VideoCapture(imageFile)
    ret, frame = cap.read()
    while(ret):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

def sift_Video(siftFile = '/Users/mohammadsaminyasar/Downloads/JIGSAWS/Knot_Tying/video/Knot_Tying_B001_capture1.avi'):
    cap = cv2.VideoCapture(siftFile)
    ret, frame = cap.read()
    count = 0
    dist = []
    description_array = []
    while (ret):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(gray,None)
            img = cv2.drawKeypoints(gray, kp, frame)
            description_array.append(des)
            #print des
            #for kp_elem in kp:
            #    vector1 = [kp_elem.response, kp_elem.pt[0], kp_elem.pt[1], kp_elem.size, kp_elem.angle]
            #print vector1
            #regr = mixture.BayesianGaussianMixture(covariance_type='full').fit(des)
            #regr.fit(img.reshape(-1,3))
            #dist.append(regr.predict(des))
            #img = plot_results(img.reshape(-1,3), regr.predict(img.reshape(-1,3)), regr.means_, regr.covariances_, 1,'Bayesian Gaussian Mixture with a Dirichlet process prior')
            if count%10 ==0:
                cv2.imwrite("/Users/mohammadsaminyasar/Downloads/Lfd/JIGSAWS/imgR_2/GMM/frame%d.png" % count, frame)

            count +=1
        else:
            break

    #description_array = joblib.load( "dist.p")
    return description_array

def edge_compute(imageFile=None):
    print "edge computation"
    cap = cv2.VideoCapture(imageFile, 0)
    ret =1
    count = 0
    ret, frame = cap.read()
    edges = []
    r_nearfar = []
    r_farlastfar = []
    for_graph = []
    frame_array = bgsegm(imageFile)
    for i in range(1, len(frame_array)-10):
        edge = ECR(frame_array[i], frame_array[i-1], 640, 480)
        for_graph.append(edge)
        if edge!=0:
            edge1 = ECR(frame_array[i+10], frame_array[i], 640, 480)
            edge2 = ECR(frame_array[i+9], frame_array[i-1], 640, 480)
            r_nearfar.append(edge1/edge)
            r_farlastfar.append(edge1/edge2)
        else:
            continue
    score = 0.0

<<<<<<< HEAD
    imageFile = imageFile.replace(".avi", ".jpg")
=======
    imageFile = imageFile.replace(".mov", ".jpg")
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
    graph_plot(for_graph, for_graph, imageFile)
    transcriptFile = imageFile.split('/')[7]
    results = np.array(readTranscriptions(transcriptFile))
    for i in range(1, len(frame_array)-11):
        if r_farlastfar[j] > far_thresh or (r_farlastfar[j] > ratio_thresh and r_nearfar[j] > near_thresh):
            score += check_results(j, results)
        #if for_graph[i]-for_graph[i-1]>=0.33:
        #    score+=check_results(i, results)
    print score/len(results)

    '''prev_best = 0
    new_best = 0
    far_thresh = 4
    near_thresh = 2
    ratio_thresh = 80
    while score/len(results)<=0.7:
        prev_best = new_best
        score =0.0
        for j in range (len(r_nearfar)):
            if r_farlastfar[j] > far_thresh or (r_farlastfar[j] > ratio_thresh and r_nearfar[j] > near_thresh):
                score += check_results(j, results)
        new_best = score/len(results)
        print new_best
        if new_best>prev_best:
            near_thresh =near_thresh + 0.1
            ratio_thresh = ratio_thresh + 5
        else:
            near_thresh =near_thresh - 0.1
            ratio_thresh = ratio_thresh - 5'''



<<<<<<< HEAD
def adaptive_threshold(imageFile = None):
    frame_array = bgsegm(imageFile)
    new_frames = []
    diff_array = []
    for i, frame in enumerate(frame_array):
        new_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frame_array = new_frames
    for i in range(len(new_frames)-1):
        diff = cv2.absdiff(new_frames[i], new_frames[i+1])
        diff = cv2.sumElems(diff)
        diff_array.append(diff[0])
    diff_array = np.array(diff_array)
    diff_array = diff_array.reshape(-1,1)
    print "shape: {}".format(diff_array.shape)
    imageFile = imageFile.replace(".avi", ".png")
    scaler = preprocessing.StandardScaler().fit(diff_array)
    diff_array = scaler.transform(diff_array)
    max_diff = 0.3*max(diff_array)
    for i in range(diff_array.shape[0]):
        if diff_array[i]>=max_diff:
            print i
    graph_plot(diff_array, diff_array, imageFile)
=======
def adaptive_threshold(resuls = None):
    print "null"


>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587

def histogram_Extraction(imageFile=None):
    print "histogram_Extraction"
    #imageFile = homingFile
    cap = cv2.VideoCapture(imageFile, 0)
<<<<<<< HEAD
    print imageFile
=======
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
    ret =1
    count = 0
    ret, frame = cap.read()
    histogram_array = []
    frame_array = bgsegm(imageFile)

    new_frames= []
    count = 0

    for frame in frame_array:
<<<<<<< HEAD
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([frame], [0, 1], None, [16,8],[0, 256, 0, 256])
=======
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame], [0, 1,2], None, [64,64,64],[0, 256, 0, 256, 0,256])
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
        #hist = cv2.calcHist([frame], [0], None, [64],[0, 256])
        hist = cv2.normalize(hist,hist)
        histogram_array.append(hist)
        count+=1
    histogram_array = np.array(histogram_array)
    #plotHistogram(histogram_array)
    diff_array = []
    for i in range (histogram_array.shape[0]-1):
<<<<<<< HEAD
        diff = cv2.compareHist(histogram_array[i], histogram_array[i+1], 0)
        #print "frame no :{} difference :{} " .format(i,diff)
        diff_array.append(diff)
    imageFile = imageFile.replace(".avi", ".jpg")
    diff_array = np.array(diff_array)
    temporal_window = 2
    diff_array = diff_array.reshape(-1,1)
    gmm = mixture.GaussianMixture(n_components = 5, max_iter = 10000, tol = 1e-3)
    gmm.fit(diff_array)
    results = gmm.predict(diff_array)
    for i in range(len(results)-1):
        if results[i]!=results[i+1]:
            print i

=======
        diff = cv2.compareHist(histogram_array[i], histogram_array[i+1], 2)
        #print "frame no :{} difference :{} " .format(i,diff)
        diff_array.append(diff)
        imageFile = imageFile.replace(".avi", ".jpg")
    diff_array = np.array(diff_array)
    for i in range(diff_array.shape[0]):
        for j in range(i, i+100):
            max_val = max(diff_array[i:i+100])


    mean_diff = np.mean(diff_array)
    std_diff = np.std(diff_array)
    threshold = mean_diff #+ 6*std_diff
    print threshold
    cuts = []

    for i in range (len(cuts)):
        for j in range (diff_array.shape[0]):
            if cuts[i] ==diff_array[j]:
                print j
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
    graph_plot(diff_array, diff_array, imageFile)
    temp_array = diff_array
    diff_array = np.sort(diff_array)
    transcriptFile = imageFile.split('/')[7]
    print transcriptFile
    results = np.array(readTranscriptions(transcriptFile))
<<<<<<< HEAD

=======
    print results
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
    score = 0.0
    for i in range(results.shape[0]):
        for j in range (len(temp_array)):
            if (diff_array[i]==temp_array[j]):
                #print j
                score+=check_results(j, results)
                break
    print score
    print results.shape[0]
    print score/results.shape[0]

def plotHistogram(array=None):
    for i in range(array.shape[0]-1):
        n, bins, patches = plt.hist(array[i])
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        fig = plt.hist(array[i], bins=8)
        savefig(fig, dpi = 300)
        plt.close()
        #plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram:{}'.format(i))

def check_results(frame_num, trans_array):
    for j, value in enumerate(trans_array):
        if frame_num-value[0]<=100 and value[1]-frame_num<=100:
            print "frame: {} boundary :{}".format(frame_num, value)
            return 1
    return 0

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def graph_plot(y_true, y_pred,figName = None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print y_true.shape
    #e_traj = np.transpose(np.array(e_traj))
    no_graphs = y_true.shape[0]
    new_y_true= []
    new_y_pred= []
    plt.grid(True)
    y_label = "match index"
    x_label = "steps in trajectory"
    for i in range (6):
        subplot_num = "32{}" .format(i+1)
        plt.subplot(int(subplot_num))

<<<<<<< HEAD
        #plt.plot(y_pred[i*(y_pred.shape[0]/6):(i+1)*y_pred.shape[0]/6], 'b')
=======
        plt.plot(y_pred[i*(y_pred.shape[0]/6):(i+1)*y_pred.shape[0]/6], 'b')
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
        plt.plot(y_true[i*(y_pred.shape[0]/6):(i+1)*y_pred.shape[0]/6], 'r')
        plt.xlabel(x_label)
        plt.ylabel(y_label,fontsize = 5)
    #if e_traj == '':
    #    plt.plot(e_traj[i][0:e_traj.shape[1]], 'g')

    savefig(figName, dpi = 300)
    plt.close()


def clusters():
    global root
    print "clusters"
    alphabets = ["B", "C", "D", "E", "F", "G", "H", "I"]
    num = [1,2,3,4,5]
    super_count_array = []
    hierarchical_clusters = []
    for a in range(len(alphabets)):
        if alphabets[a] == "H":
            num = [1,3,4,5]
        for n in range(len(num)):
<<<<<<< HEAD
            imageFile = root + "Knot_Tying/video/Knot_Tying_{}00{}_capture1.avi" .format(alphabets[a], num[n])
=======
            imageFile = root + "video/Knot_Tying_{}00{}_capture1.avi" .format(alphabets[a], num[n])
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
            #sift_Video(imageFile)
            count = 0
            descr = []
            count_array = []
<<<<<<< HEAD
            #imageFile = "/home/uva-dsa1/Downloads/output.avi"
            cap = cv2.VideoCapture(imageFile)
            ret, frame = cap.read()
            #edge_compute(imageFile)
            histogram_Extraction(imageFile)
=======

            cap = cv2.VideoCapture(imageFile)
            ret, frame = cap.read()
            edge_compute(imageFile)
            #histogram_Extraction(imageFile)
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
            imageFile = imageFile.replace(".avi", ".jpg")
            print imageFile
            '''while (cap.isOpened()):
                ret, frame = cap.read()
                if (ret == True):
                    #frame = cv2.imread("/Users/mohammadsaminyasar/Downloads/Lfd/JIGSAWS/imgR_2/actual/frame%d.jpg" % i)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sift = cv2.xfeatures2d.SIFT_create()
                    kp, des = sift.detectAndCompute(gray,None)
                    descr.append(des)
                else:
                    break
            descr = np.array(descr)
            joblib.dump(descr, 'descr.p')
            descr = joblib.load ('descr.p')

            for j in range(descr.shape[0]-1):
                #print "before matching : {} : {}".format(descr[j].shape, descr[j+1].shape)
                matches = np.concatenate((descr[j], descr[j+1]), axis = 0)

                clf = KMeans().fit(matches)
                nearest_clusters = clf.fit(matches)
                hierarchical_clusters.append(nearest_clusters)
            hierarchical_clusters = np.array(hierarchical_clusters)
            for j in range(hierarchical_clusters.shape[0]-1):
                #print "before matching : {} : {}".format(descr[j].shape, descr[j+1].shape)
                matches = np.concatenate((hierarchical_clusters[j], hierarchical_clusters[j+1]), axis = 0)

                clf = KMeans().fit(matches)
                nearest_clusters = clf.fit(matches)


                graph_plot (nearest_clusters, nearest_clusters, imageFile)'''

def clusters_raven(imageFile ="/Users/mohammadsaminyasar/Downloads/JIGSAWS/FLS.mp4"):
    imageFile = homingFile
    count = 0
    descr = []
    count_array = []
    descr = np.array(sift_Video(imageFile))
    imageFile = imageFile.replace(".mov", ".png")
    #descr = joblib.load ('descr.p')
    #descr = np.array(descr)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    seach_params = dict(checks = 100)
    flann = cv2.FlannBasedMatcher (index_params, seach_params)
    #matches = [[0,0] for i in xrange(len(matches))]

    for i in range(descr.shape[0]-1):
        bf = cv2.BFMatcher()
        matches = flann.knnMatch(descr[i],descr[i+1], k=2)
        # Apply ratio test
        count = 0
        good =[]
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append([m])

        if len(good)>0.55*np.array(matches).shape[0]:
            count_array.append(1)
        else:
            count_array.append(0)

    graph_plot (count_array, count_array, imageFile)

def readTranscriptions(transcriptFile = None):
    global root
    #transcriptFile =  + Suturing_B001_capture1.jpg"
    transcriptFile = "/transcriptions/" + transcriptFile.replace('_capture1.jpg', '.txt')
    File = root + transcriptFile
<<<<<<< HEAD
    #print File
    #print root
=======
    print File
    print root
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
    df = pd.read_csv(File, header = None, delimiter=' ')
    df = df.values.tolist()
    frame_boundary = []
    gesture = []
    for i, value in enumerate(df):
        frame_boundary.append([value[0], value[1]])
        gesture.append(value[2])
    return frame_boundary
    #print ": {} : {}".format(end_frame, gesture)
def main():
<<<<<<< HEAD
    videoFile = root + 'Knot_Tying/video/Knot_Tying_B001_capture1.avi'
=======
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587
    cv2.useOptimized()
    usage = "0:loadImage| 1:play_Video| 2:load_Video| 3:sift_Video| 4:foreground_Extraction"
    try:
        script,mode = argv
    except:
        print "Error: missing parameters"
        print usage
        sys.exit(0)
    #readTranscriptions()
    if mode == "0":
        clusters()
        #clusters_raven()
    elif mode == "1":
        loadImage()
    elif mode == "2":
        bgsegm()
    elif mode == "3":
        sift_Video()
    elif mode == "4":
        histogram_Extraction()
    elif mode == "5":
<<<<<<< HEAD
        adaptive_threshold(videoFile)
        #optical_flow()
=======
        optical_flow()
>>>>>>> 983b2de3a894689d4a3501fd285a4787041b1587

if __name__ == '__main__':
    main()
