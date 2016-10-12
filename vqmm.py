#!/usr/bin/python
#
# Author    Yann Bayle
# E-mail    bayle.yann@live.fr
# License   MIT
# Created   09/09/2016
# Updated   12/10/2016
# Version   1.0.0
#
# Object    Preprocess file from YAAFE and launch VQMM from Thibault Langlois
#           You can find the latest version of his algorithm here:
#           https://bitbucket.org/ThibaultLanglois/vqmm/downloads
#
# OS        Only work on UNIX for the moment
#
# Details   - Remove header from CSV (5 lines)
#           - Replace scientifc notation by float
#           - Replace commas by spaces
#           - Check correct numbers of col = 13
#           - Check that there are no empty files
#           - Check minimum number of frames
#           - All non conform files are deplaced in the error folder
#           - Manage folds and launch VQMM
#
# Manual    1 Install YAAFE and analyse your songs in a folder ex: /path/YAAFE/
#           2 Download https://github.com/ybayle/vqmm
#           3 Create a fileList.txt containing path & class of YAAFE's files
#           4 Launch: python vqmm.py -d /path/YAAFE/ -f /path/fileList.txt
#
# TODOs     - Optimize: 700ms/file for ~1Mo/file ~4800lines/file 13feat/line
#               on i7-3770k, 16Go RAM, 3.5GHz, 64bits, Debian 8.2
#           - Tell user not to use "NOT_" in his class name or better manage it
#           - Train/Test more folds
#           - Make parallel: preprocess, each fold and test
#           - Instead of display results txt, display fig
#           - X "files to classify" in main.c remove dot . and disp progression
#           - Enhance help and automatic produce of man and README
#           - Assert
#           - parameterized epsilon for codebook
#           - take in account global var verbose with -v
#           - Write a txt file which indicates all param chosen
#

import time
import argparse
import sys
import os
import csv
import shutil
import subprocess
import re
import json
import multiprocessing
import fnmatch
import webbrowser
from functools import partial
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

begin = int(round(time.time() * 1000))

VERBOSE = False
PRINTDEBUG = True

class bcolors:
    HOUR = '\033[96m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    FILE = '\033[37m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plotResults(dirName):
    # Format of csv files created by VQMM:
    #   ClassName (tag), tp, tn, fp, fn, precision, recall, fscore
    files = []
    for fileName in os.listdir(dirName):
        if fnmatch.fnmatch(fileName, '*perTag.csv'):
            files.append(fileName)
    data = {}
    for fileName in files:
        with open(dirName + fileName, 'r') as res:
            resReader = csv.reader(res, delimiter=',')
            for row in resReader:
                tag = row[0]
                if tag in data:
                    data[tag]["precision"].append(row[5])
                    data[tag]["recall"].append(row[6])
                    data[tag]["fScore"].append(row[7])
                else:
                    data[tag] = {
                        "precision":[row[5]],
                        "recall":[row[6]],
                        "fScore":[row[7]]
                    }

    nbMeasure = 3
    nbClass = len(data)
    nbFolds = len(data[tag]["precision"])
    dataPlot = np.zeros((nbClass, nbFolds, nbMeasure))
    classIndex = 0
    tagName = []
    for tag in data:
        tagName.append(tag)
        dataPlot[classIndex, :, 0] = data[tag]["precision"]
        dataPlot[classIndex, :, 1] = data[tag]["recall"]
        dataPlot[classIndex, :, 2] = data[tag]["fScore"]
        classIndex = classIndex + 1

    # Figure display part
    plt.close("all")
    fig, axes = plt.subplots(nrows=1, ncols=nbClass, figsize=(12, 5))

    # rectangular box plot
    bplot1 = axes[0].boxplot(dataPlot[0],
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color
    bplot2 = axes[1].boxplot(dataPlot[1],
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    index = 0
    # tagName = ["Class 1", "Class 2"]
    for ax in axes:
        ax.yaxis.grid(True)
        ax.set_ylabel('Value of measure')
        ax.set_xlabel(tagName[index])
        index = index + 1
        ax.set_ylim([0.0, 1.0])

    # add x-tick labels
    plt.setp(axes, xticks=[y+1 for y in range(nbMeasure)],
             xticklabels=['Precision', 'Recall', 'F-Score'])

    imgName = dirName + "figure.png"
    plt.savefig(imgName, dpi=100)
    webbrowser.open(imgName)

def extractPathAndClass(s):
    delimiter = '/'
    insertStr = "processed/"
    if insertStr in s:
        insertStr = ""
    limit = s.rindex(delimiter) + len(delimiter)
    line = s[:limit] + insertStr + s[limit:]
    index = line.index('\t')
    return line[:index], line[index+1:-1]

def validScientificNotation(val):
    pattern = re.compile("-?[0-9]\.[0-9]+[Ee][+-][0-9]{2}")
    if val:
        if pattern.match(val):
            return True
        else:
            return False
    else:
        return False

def validAndConvertFile(inDIR, outDIR, errDIR, filename):
    inFileName = inDIR + filename
    errFileName = errDIR + filename
    # Remove empty file or with too few lines to be analyzed
    if os.stat(inFileName).st_size > 4000:
        # Above line is slightly faster (20microsec) than
        # os.stat(inFileName).st_size
        outFileName = outDIR + filename
        outFile = open(outFileName, 'w')
        fileInvalid = False
        try:
            with open(inFileName, 'r') as data:
                reader = csv.reader(data)
                dismissedLines = 5
                requiredNumCol = 13
                for row in reader:
                    if 0 == dismissedLines:
                        if not len(row) < requiredNumCol:
                            str2write = ""
                            for col in row:
                                if validScientificNotation(col):
                                    str2write = str2write + "{0:.14f}".format(float(col)) + " "
                                else:
                                    fileInvalid = True
                                    break
                            outFile.write(str2write[:-1] + "\n")
                        else:
                            fileInvalid = True
                            break
                    else:
                        dismissedLines = dismissedLines - 1
        finally:
            outFile.close()
            if fileInvalid:
                # os.remove(outFileName) # TODO uncomment
                if not os.path.exists(errDIR):
                    os.makedirs(errDIR)
                shutil.move(inFileName, errFileName)
    else:
        if not os.path.exists(errDIR):
            os.makedirs(errDIR)
        shutil.move(inFileName, errFileName)

def find_between_r(s, first, last):
    """Description of find_between_r (s, first, last)

    :param s: A string
    :param first: Beginning delimiter for substring
    :param last: Ending delimiter for substring
    :type s: string
    :type first: string
    :type last: string
    :return: Substring contained between first and last by reading s from the
             end to the beginning
    :rtype: string

    :Example:

    s = "/media/sf_DATA/results/VQMM_YAAFE/ZAA641200304_audio_full_mono_22k.wav.mfcc.csv    I"
    print(find_between_r(s, "/", "\t"))
    ZAA641200304_audio_full_mono_22k.wav.mfcc.csv

    """
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""

def curTime():
    return bcolors.HOUR + datetime.now().time().strftime("%Hh%Mm%Ss") + " " + bcolors.ENDC

def printError(msg):
    print(bcolors.BOLD + bcolors.ERROR + "ERROR:\n" + msg + "\nProgram stopped" + bcolors.ENDC)
    sys.exit()

def printTitle(msg):
    if PRINTDEBUG:
        print(curTime() + bcolors.BOLD + bcolors.OKGREEN + msg + bcolors.ENDC)

def printMsg(msg):
    if VERBOSE:
        print(bcolors.HEADER + msg + bcolors.ENDC)

def printInfo(msg):
    if PRINTDEBUG:
        print(curTime() + bcolors.OKBLUE + msg + bcolors.ENDC)

def printWarning(msg):
    if PRINTDEBUG:
        print(bcolors.WARNING + msg + bcolors.ENDC)

def printFile(fileName):
    if os.path.isfile(fileName):
        printInfo(fileName + ":")
        print(bcolors.FILE)
        with open(fileName, 'r') as fn:
            for line in fn:
                print(line[:-1])
        print(bcolors.ENDC)
    else:
        printWarning("File not found: " + fileName)

def runTrainTestOnFold(i, args):
    tmpDIR = args["tmpDIR"]
    codebookFile = args["cbkDir"] + "codebook.cbk"
    resultsDir = tmpDIR + "Results/"
    modelsDir = tmpDIR + "Models/"

    # print("runTrainTestOnFold")
    # print("i " + str(i) + " args " + str(args))
    trainFileList = args["foldsName"][i]
    trainOn = list(set(args["foldsName"]) - set([trainFileList]))
    tmpNb = [str(val) for val in range(1, args["nbFolds"]+1)]
    tmpNb.remove(trainFileList[-5])
    foldsNumber = ''.join(str(x) for x in tmpNb)
    testFileList = args["tmpDIR"] + "testFolds_" + foldsNumber + ".csv"
    os.system("cat " + " ".join(trainOn) + " > " + testFileList)

    printInfo("Training Model on Fold " + str(i+1))
    with open(args["cbkDir"]+args["projName"]+"/"+str(i)+"_train.txt", 'w') as f:
        subprocess.call([args["pathVQMM"] + 'vqmm', '-quiet', 'n', '-output-dir', modelsDir, '-list-of-files', trainFileList, '-epsilon', args["epsilon"], '-smoothing', args["smoothing"], '-codebook', codebookFile, '-make-tag-models'], stdout=f, stderr=f)
        # subprocess.call([args["pathVQMM"] + 'vqmm', '-quiet', 'n', '-output-dir', modelsDir, '-list-of-files', trainFileList, '-epsilon', args["epsilon"], '-codebook', codebookFile, '-make-tag-models'], stdout=f, stderr=f)

    modelsFile = tmpDIR + "Models" + str(i) + ".csv"
    with open(modelsFile, 'w') as mf:
        for className in args["classNames"]:
            mf.write(modelsDir + className + "$"+ find_between_r(trainFileList, "/", ".") + ".mm\n")

    printInfo("Testing Model on Fold " + str(i+1))
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
    # printInfo("Approx 515ms per file")
    with open(args["cbkDir"]+args["projName"]+"/"+str(i)+"_test.txt", 'w') as f:
        subprocess.call([args["pathVQMM"] + 'vqmm', '-tagify', '-output-dir', resultsDir, '-models', modelsFile, '-codebook', codebookFile, '-list-of-files', testFileList], stdout=f, stderr=f)
    # os.remove(testFileList) # TODO uncomment
    # os.remove(modelsFile) # TODO uncomment
    printInfo("Fold " + str(i+1) + " tested")

def runVQMM(args):
    fileListWithClass = args["fileListWithClass"]
    tmpDIR = args["tmpDIR"]
    randomSeed = str(args["randSeedCbk"])
    codebookSize = str(args["cbkSize"])
    codebookFile = args["cbkDir"] + "codebook.cbk"
    resultsDir = tmpDIR + "Results/"
    tmpModels = tmpDIR + "tmpModels.csv"
    modelsDir = tmpDIR + "Models/"
    if not os.path.exists(modelsDir):
        os.makedirs(modelsDir)

    printTitle("Compiling VQMM")
    os.system("make -C " + args["pathVQMM"] + "src/")

    if os.path.isfile(codebookFile):
        printTitle("VQMM Codebook already created for this codebook size")
    else:
        printTitle("Creating VQMM Codebook")
        with open(args["cbkDir"]+"cbk_stderr.txt", 'w') as f:
            subprocess.call([args["pathVQMM"] + 'vqmm', '-quiet', 'n', '-list-of-files', fileListWithClass, '-random', randomSeed, '-codebook-size', codebookSize, '-codebook', codebookFile], stderr=f)

    if args["nbFolds"] == 1:
        printTitle("Training Model")
        # subprocess.call([args["pathVQMM"] + 'vqmm', '-quiet', 'n', '-output-dir', modelsDir, '-list-of-files', fileListWithClass, '-epsilon', args["epsilon"], '-smoothing', args["smoothing"], '-codebook', codebookFile, '-make-tag-models'])
        subprocess.call([args["pathVQMM"] + 'vqmm', '-quiet', 'n', '-output-dir', modelsDir, '-list-of-files', fileListWithClass, '-epsilon', args["epsilon"], '-codebook', codebookFile, '-make-tag-models'])

        modelsFile = tmpDIR + "Models.csv"
        os.system("readlink -f $(echo \"" + modelsDir + "*\") >> " + tmpModels)
        os.system("sed -n '/NOT_/!p' " + tmpModels + " >> " + modelsFile)
        # os.remove(tmpModels) # TODO uncomment

        printTitle("Testing Model")
        if not os.path.exists(resultsDir):
            os.makedirs(resultsDir)
        printInfo("Approx 515ms per file")
        subprocess.call([args["pathVQMM"] + 'vqmm', '-tagify', '-output-dir', resultsDir, '-models', modelsFile, '-codebook', codebookFile, '-list-of-files', fileListWithClass])
        # os.remove(modelsFile) # TODO uncomment

        printTitle("Results:")
        displayedRes = False
        for fileName in os.listdir(resultsDir):
            if fileName.endswith("summary.txt"):
                printFile(resultsDir+fileName)
                displayedRes = True
            elif fileName.endswith("perTag.txt"):
                printFile(resultsDir+fileName)
                displayedRes = True
        if not displayedRes:
            printError("Error during VQMM, no results to display, see " + args["analysisFolder"] + " for more details.")
    else:
        generateFolds(args)
        printWarning("TODO manage inversion of Train and Test Set")

        # Parallel computing on each TrainTestFolds
        printTitle("Parallel train & test of folds")
        partialRunTrainTestOnFold = partial(runTrainTestOnFold, args=args)
        pool = multiprocessing.Pool(args["nbFolds"])
        pool.map(partialRunTrainTestOnFold, range(args["nbFolds"])) #make our results with a map call
        pool.close() #we are not adding any more processes
        pool.join() #tell it to wait until all threads are done before going on

        printTitle("Display resulting image in default browser")
        plotResults(resultsDir)

def createDir(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        return True
    else:
        return False

def preprocess(args):
    printTitle("Preprocessing")
    inDIR = args["inDIR"]
    fileWithClass = args["fileWithClass"]
    if inDIR[-1] != '/' and inDIR[-1] != '\\':
        inDIR = inDIR + '/'
    errDIR = inDIR + "error/"
    outDIR = inDIR + "processed/"

    # Create folder and subfolders if does not exists
    createDir(args["analysisFolder"])

    args["projDir"] = args["analysisFolder"] + inDIR[:-1][inDIR[:-1].rindex("/")+1:] + "/"
    createDir(args["projDir"])

    args["cbkDir"] = args["projDir"] + "CodeBookSize_" + str(args["cbkSize"]).zfill(3) + "/"
    createDir(args["cbkDir"])

    projName = str(args["randSeedCbk"]) + "RandCbk_"
    projName = projName + str(args["randSeedFold"]) + "RandFold_"
    projName = projName + str(args["nbFolds"]) + "Fold"
    if args["nbFolds"] > 1:
        projName = projName + "s_"
    else:
        projName = projName + "_"
    projName = projName + str(args["epsilon"]) + "Eps_"
    projName = projName + str(args["smoothing"]) + "Smooth"
    if args["invertTrainTest"]:
        projName = projName + "_I"
    args["projName"] = projName
    tmpDIR = args["cbkDir"] + projName + "/"

    if not createDir(tmpDIR):
        printError("A project with same params exists")
    tmpFileNames = args["projDir"] + "files.txt"
    fileListWithClassJSON = args["projDir"] + "filelist.json"
    fileListWithClass = args["projDir"] + "filelist.csv"

    classes = None
    classNames = []
    if not os.path.exists(outDIR):
        os.system("ls " + inDIR + " > " + tmpFileNames)
        os.makedirs(outDIR)
        printTitle("Validating and converting files")
        with open(tmpFileNames, 'r') as filenames:
            curFileNum = 0
            for filename in filenames:
                curFileNum = curFileNum + 1
                sys.stdout.write("\r\t" + str(curFileNum))
                sys.stdout.flush()
                filename = filename[:-1]
                if not os.path.isdir(filename):
                    validAndConvertFile(inDIR, outDIR, errDIR, filename)
            sys.stdout.write('\n')
            sys.stdout.flush()
        printTitle("Associating classes")
        os.system("ls " + outDIR + " > " + tmpFileNames)
        with open(tmpFileNames) as f:
            linesNoClass = f.readlines()
        # os.remove(tmpFileNames) # TODO uncomment
        with open(fileWithClass) as f:
            linesWithClass = f.readlines()
        curLine = 0
        for line in linesWithClass:
            curLine = curLine + 1
            sys.stdout.write("\r\t" + str(curLine))
            sys.stdout.flush()
            tmpLine = find_between_r(line, "/", "\t") + "\n"
            if tmpLine in linesNoClass:
                itemPath, itemClass = extractPathAndClass(line)
                if not classes:
                    classes = {itemClass: [itemPath]}
                elif not itemClass in classes:
                    classes[itemClass] = [itemPath]
                else:
                    classes[itemClass].append(itemPath)
        sys.stdout.write('\n')
        if args["nbFolds"] > 1:
            with open(fileListWithClassJSON, 'w') as fp:
                json.dump(classes, fp, sort_keys=True, indent=2)

        for key in classes:
            with open(fileListWithClass, 'a') as fp:
                for line in classes[key]:
                    fp.write(str(line) + "\t" + str(key) + "\n")
            classNames.append(key)
        printTitle("Preprocessing done")
    else:
        if not os.path.isfile(fileListWithClass):
            fileListWithClass = args["fileWithClass"]
        with open(fileListWithClass, 'r') as fp:
            for line in fp:
                itemPath, itemClass = extractPathAndClass(line)
                if not classes:
                    classes = {itemClass: [itemPath]}
                elif not itemClass in classes:
                    classes[itemClass] = [itemPath]
                else:
                    classes[itemClass].append(itemPath)
                classNames.append(itemClass)
        fileListWithClass = args["projDir"] + "filelist.csv"
        if not os.path.isfile(fileListWithClass):
            for key in classes:
                with open(fileListWithClass, 'a') as fp:
                    for line in classes[key]:
                        fp.write(str(line) + "\t" + str(key) + "\n")
        with open(fileListWithClassJSON, 'w') as fp:
            json.dump(classes, fp, sort_keys=True, indent=2)
        printTitle("Files already preprocessed")

    args["classNames"] = list(set(classNames))
    args["tmpDIR"] = tmpDIR
    args["fileListWithClass"] = fileListWithClass
    args["fileListWithClassJSON"] = fileListWithClassJSON

    return args

def gatherArgs(argv):
    parser = argparse.ArgumentParser(description="Use extracted features from YAAFE and classify them with VQMM.")
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    parser.add_argument(
        "-i",
        "--invert",
        help="invert train and test set",
        action="store_true")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        metavar="DIR",
        help="directory where YAAFE features are stored")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        metavar="FILE",
        help="file containing paths and classes separated by a tab")
    parser.add_argument(
        "-n",
        "--nbFolds",
        default=1,
        type=int,
        metavar="NBFOLDS",
        help="number of Folds to be used for the classification, must be >= 1")
    parser.add_argument(
        "-r",
        "--randFolds",
        default=28,
        type=int,
        metavar="RANDFOLDS",
        help="random seed used for splitting folds")
    parser.add_argument(
        "-s",
        "--seedCbk",
        type=int,
        metavar="SEEDCBK",
        help="random seed for vqmm codebook")
    parser.add_argument(
        "-c",
        "--cbkSize",
        type=int,
        metavar="CBKSIZE",
        help="size of the codebook")
    parser.add_argument(
        "-p",
        "--pathVQMM",
        type=str,
        metavar="PATHVQMM",
        help="path to Thibault Langlois' VQMM folder")
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        metavar="EPSILON",
        help="Epsilon defines the stopping criteria of k-means. It must be >= 0.")
    parser.add_argument(
        "-m",
        "--smoothing",
        type=float,
        metavar="SMOOTHING",
        help="Models' smoothing parameter. The events that has \
            not occured during training cannot have null probability. \
            It corresponds to the probability mass which is allocated to \
            events which have not occured during training. \
            Ex: -m 0.1 attributes 10 percent of probability to those events. \
            If it is close to zero, the model does not generalize. \
            The lower the parameter the higher the precision \
            The higher the paramter, the higher the recall. \
            In principle this parameter should be < 0.5 \
            It must varies between [0.0 ; 1.0]")
    tmpArgs = parser.parse_args()
    pathVQMM = "./ThibaultLanglois_VQMM/"
    if tmpArgs.pathVQMM:
        pathVQMM = tmpArgs.pathVQMM
    inDIR = "./data/"
    fileWithClass = "./filelist.txt"
    verbose = False
    if tmpArgs.verbose:
        verbose = True
        VERBOSE = True
    invertTrainTest = False
    if tmpArgs.invert:
        invertTrainTest = True
    nbFolds = 1
    if tmpArgs.nbFolds:
        if tmpArgs.nbFolds >= 1:
            nbFolds = tmpArgs.nbFolds
        else:
            printError("Wrong number of Folds")
    randSeedFold = 1
    if tmpArgs.randFolds:
        randSeedFold = tmpArgs.randFolds
    randSeedCbk = 50
    if tmpArgs.seedCbk:
        randSeedCbk = tmpArgs.seedCbk
    epsilon = 0.00001
    if tmpArgs.epsilon:
        epsilon = tmpArgs.epsilon
        if epsilon < 0:
            epsilon = 0.00001
            printWarning("Epsilon cannot be lower than 0\nEpsilon set to " + str(epsilon))
    epsilon = str(epsilon)
    smoothing = 0.000000001
    if tmpArgs.smoothing:
        smoothing = tmpArgs.smoothing
        if smoothing < 0:
            smoothing = 0.00001
            printWarning("Smoothing cannot be lower than 0\nSmoothing set to " + str(smoothing))
        elif smoothing >= 0.5:
            printWarning("Unexpected behavior when Smoothing is greater than 0.5")
    smoothing = str(smoothing)
    cbkSize = 75
    if tmpArgs.cbkSize:
        cbkSize = tmpArgs.cbkSize

    if tmpArgs.dir and tmpArgs.file:
        if os.path.exists(tmpArgs.dir):
            inDIR = tmpArgs.dir
        else:
            printError("Folder does not exists : " + tmpArgs.dir)
        if os.path.isfile(tmpArgs.file):
            fileWithClass = tmpArgs.file
        else:
            printError("File does not exists : " + tmpArgs.dir)
    elif tmpArgs.dir != tmpArgs.file:
        printError("You must input an input dir AND a filelist with paths and classes")
    printMsg("Sample folder " + inDIR)
    printMsg("Path and classes stored in " + fileWithClass)
    printMsg("Number of Folds " + str(nbFolds))
    printMsg("Random Seed for Folds " + str(randSeedFold))
    printMsg("Random Seed for Codebook " + str(randSeedCbk))
    printMsg("Codebook size " + str(cbkSize))
    printMsg("Invert Train and Test Set " + str(invertTrainTest))
    printMsg("Path to Thibault Langlois' VQMM folder " + pathVQMM)

    args = {"inDIR":inDIR}
    args["fileWithClass"] = fileWithClass
    args["verbose"] = verbose
    args["invertTrainTest"] = invertTrainTest
    args["nbFolds"] = nbFolds
    args["randSeedFold"] = randSeedFold
    args["randSeedCbk"] = randSeedCbk
    args["cbkSize"] = cbkSize
    args["pathVQMM"] = pathVQMM
    args["epsilon"] = epsilon
    args["smoothing"] = smoothing
    args["analysisFolder"] = "./analysis/"

    return args

def generateFolds(args):
    printTitle("Generating random split for folds")
    fileListWithClassJSON = args["fileListWithClassJSON"]
    nbFolds = args["nbFolds"]
    randSeedFold = args["randSeedFold"]
    invertTrainTest = args["invertTrainTest"]
    np.random.seed(randSeedFold)
    with open(fileListWithClassJSON) as data_file:
        paths = json.load(data_file)
    # os.remove(fileListWithClassJSON) # TODO uncomment
    tmpSelected = []
    for i in range(0, args["nbFolds"]):
        tmpSelected.append([])
    for key in paths:
        newSize = int(round(len(paths[key])/nbFolds))
        if 0 == newSize:
            printError("The number of folds is greater than the number of data available")
        selected = np.random.choice(paths[key], size=newSize, replace=False)
        tmpSelected[0] = selected
        remain = list(set(paths[key]) - set(selected))
        for n in range(1, args["nbFolds"]-1):
            tmpSel = np.random.choice(remain, size=newSize, replace=False)
            sel = tmpSel
            remain = list(set(remain) - set(sel))
            tmpSelected[n] = sel
        tmpSelected[-1] = remain
        foldsName = []
        for i in range(0, nbFolds):
            foldFileName = args["tmpDIR"] + "fold" + str(i+1) + ".csv"
            with open(foldFileName, "a") as fold:
                try:
                    for line in tmpSelected[i]:
                        fold.write(str(line) + "\t" + str(key) + "\n")
                except:
                    printError("The number of folds is greater than the number of data available")
            foldsName.append(foldFileName)
    args["foldsName"] = list(set(foldsName))

def main(argv):
    """Description of main

    :param argv[1]: Folder containing raw data from YAAFE
    :param argv[2]: File containing path to previous files
    :type argv[1]: string
    :type argv[2]: string

    :Example:

    python vqmm.py /path/YAAFE/ /path/fileList.txt

    .. warnings:: argv[1] must finish by '/'
    .. note:: argv[2] must contain path followed by a tab and the item's class
    """
    args = gatherArgs(argv)
    printInfo("Approx. 700ms per file: go grab a tea!")
    args = preprocess(args)
    runVQMM(args)
    printInfo("More details available in " + os.path.abspath(args["analysisFolder"]))
    printTitle("Finished in " + str(int(round(time.time() * 1000)) - begin) + "ms")

if __name__ == "__main__":
    main(sys.argv[1:])
