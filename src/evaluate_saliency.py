import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import math
from PIL import Image


def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):

    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score
    
    new_size = np.shape(fixationMap)
    if not np.shape(saliencyMap) == np.shape(fixationMap):
        new_size = np.shape(fixationMap)
        np.array(Image.fromarray(saliencyMap).resize((new_size[1], new_size[0])))

    if jitter:
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()
        tp[i + 1] = float(i + 1) / Nfixations 
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations) 

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score

def KLdiv(saliencyMap, fixationMap):

    map1 = saliencyMap.astype(float)
    map2 = fixationMap.astype(float)

    new_size = np.shape(map2)
    np.array(Image.fromarray(map1).resize((new_size[1], new_size[0])))

    if map1.any():
        map1 = map1 / map1.sum()
    if map2.any():
        map2 = map2 / map2.sum()

    eps = 10 ** -12
    score = map2 * np.log(eps + map2 / (map1 + eps))

    return score.sum()

def NSS(saliencyMap, fixationMap):

    if not fixationMap.any():
        print('Error: no fixationMap')
        score = np.nan
        return score

    new_size = np.shape(fixationMap)
    map1 = np.array(Image.fromarray(saliencyMap).resize((new_size[1], new_size[0])))

    if not map1.max() == 0:
        map1 = map1.astype(float) / map1.max()

    if not map1.std(ddof=1) == 0:
        map1 = (map1 - map1.mean()) / map1.std(ddof=1)

    score = map1[fixationMap.astype(bool)].mean()

    return score

def CC(s_map,gt):

    s_map = s_map.astype(float)
    gt = gt.astype(float)

    new_size = np.shape(gt)
    np.array(Image.fromarray(s_map).resize((new_size[1], new_size[0])))

    gt_norm = (gt - np.mean(gt)) / np.std(gt)

    if not s_map.max() == 0:
        s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
        r = (s_map_norm * gt_norm).sum() / math.sqrt((s_map_norm * s_map_norm).sum() * (gt_norm * gt_norm).sum());
    else:
        r=0

    return r



def sim(s_map,gt):

    s_map = s_map.astype(float)
    gt = gt.astype(float)


    new_size = np.shape(gt)
    np.array(Image.fromarray(s_map).resize((new_size[1], new_size[0])))


    gt = (gt - np.min(gt)) / ((np.max(gt) - np.min(gt)) * 1.0)
    if not s_map.max() == 0:
        s_map=(s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) * 1.0)
        s_map = s_map/(np.sum(s_map)*1.0)
        gt = gt/(np.sum(gt)*1.0)
        x,y = np.where(gt>0)
        sim = 0.0
        for i in zip(x,y):
            sim = sim + min(gt[i[0],i[1]],s_map[i[0],i[1]])
    else:
        sim =0
    return sim
