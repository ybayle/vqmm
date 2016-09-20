#!/usr/bin/python
#
# Author	Yann Bayle
# E-mail	bayle.yann@live.fr
# License   MIT
# Created	09/09/2016
# Updated	15/09/2016
# Version	2
# Object	Preprocess file from YAAFE and launch VQMM from Thibault Langlois
#			You can find the latest version of his algorithm here:
# 			https://bitbucket.org/ThibaultLanglois/vqmm/downloads
# OS 		Only work on UNIX for the moment
# Details 	- Remove header from CSV (5 lines)
#			- Replace scientifc notation by float
# 			- Replace commas by spaces
#			- Check correct numbers of col = 13
# 			- Check that there are no empty files
# 			- Check minimum number of frames
# 			- All non conform files are deplaced in the error folder
# INPUT		1 Path to folder containing all analysed raw data from YAAFE
#			2 File containing of path to YAAFE resulsts file
# OUTPUT	1 In the folder containing all raw data :
#				A folder named "processed/" containing all processed files
#				An eventual folder named "error/" containing all invalid files
#			2 In analysis/ folder :
#				Models and Resulting files from VQMM analysis
# MANUAL	1 Install YAAFE and analyse your songs in a folder ex: /path/YAAFE/
# 			2 Download https://github.com/ybayle/vqmm
# 			3 Launch: python vqmm.py /path/YAAFE/ /path/fileList.txt
# TODOs		- Optimize code: 700ms/file for 1Mo/file 4800lines/file 13feat/line
# 			- Input params: randomseed, codebooksize, nb Folds
#			- Manage vqmm verbose
#			- Tell user not to use "NOT_" in his class name or better manage it
#			- Train/Test more folds
#			- Make parallel: preprocess, each fold and test
#			- Instead of display results txt, display fig
# 			- X "files to classify" in main.c remove dot . and disp progression
# 			- Enhance help and automatic produce of man and README
#			- Assert
#			- make available import vqmm :
#				vqmm.vqmm with no arg launch default
#				vqmm.vqmm(lot of tmpArgs non mandatory)
#			- Main : enhance getting of tmpArgs with option
#			- parameterized VQMM folder and epsilon for codebook
#			- modifier global var verbose quand -v est active
# 

import time
begin = int(round(time.time() * 1000))
import argparse
import sys, getopt
import os
import csv
import shutil 
from os import listdir
from os.path import isfile, join
import subprocess
import re
import numpy as np
import json

VERBOSE = False
PRINTDEBUG = True

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	ERROR = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def extractPathAndClass(s):
	delimiter = '/'
	insertStr = "processed/"
	limit = s.rindex( delimiter ) + len( delimiter )
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
				os.remove(outFileName)
				if not os.path.exists(errDIR):
					os.makedirs(errDIR)
				shutil.move(inFileName, errFileName)
	else:
		if not os.path.exists(errDIR):
			os.makedirs(errDIR)
		shutil.move(inFileName, errFileName)

def find_between_r( s, first, last ):
	"""Description of find_between_r ( s , first , last )

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

	s = "/media/sf_DATA/results/VQMM_YAAFE/ZAA641200304_audio_full_mono_22k.wav.mfcc.csv	I"
	print(find_between_r( s, "/", "\t" ))
	ZAA641200304_audio_full_mono_22k.wav.mfcc.csv

	"""
	try:
		start = s.rindex( first ) + len( first )
		end = s.rindex( last, start )
		return s[start:end]
	except ValueError:
		return ""

def printError(msg):
	print(bcolors.BOLD + bcolors.ERROR + "ERROR:\n" + msg + "\nProgram stopped" + bcolors.ENDC)
	sys.exit()

def printTitle(msg):
	if PRINTDEBUG:
		print(bcolors.BOLD + bcolors.OKGREEN + msg + bcolors.ENDC)

def printMsg(msg):
	if VERBOSE:
		print(bcolors.HEADER + msg + bcolors.ENDC)

def printInfo(msg):
	if PRINTDEBUG:
		print(bcolors.OKBLUE + msg + bcolors.ENDC)

def printWarning(msg):
	if PRINTDEBUG:
		print(bcolors.WARNING + msg + bcolors.ENDC)

def usage():
	printError('You must indicate the folder where raw data from YAAFE are stored and the class file:\nvqmm.py ./data/ ./filelist.txt')

def runVQMM(*arg):
	args = arg[0]
	fileListWithClass = args["fileListWithClass"]
	if len(arg) == 1:
		trainFileList = fileListWithClass
		testFileList = fileListWithClass
	else:
		trainFileList = arg[1]
		testFileList = arg[2]
	tmpDIR = args["tmpDIR"]
	randomSeed = str(args["randSeedCbk"])
	codebookSize = str(args["cbkSize"])
	codebookFile = tmpDIR + "codebook.cbk"
	resultsDir = tmpDIR + "Results/"
	modelsFile = tmpDIR + "Models.csv"
	tmpModels = tmpDIR + "tmpModels.csv"
	modelsDir = tmpDIR + "Models/"
	if not os.path.exists(modelsDir):
		os.makedirs(modelsDir)
	
	printTitle("Compiling VQMM")
	os.system("make -C ./ThibaultLanglois-VQMM/")

	printTitle("Creating VQMM Codebook")
	subprocess.call(['./ThibaultLanglois-VQMM/vqmm', '-quiet', 'y', '-list-of-files', fileListWithClass, '-random', randomSeed, '-codebook-size', codebookSize, '-codebook', codebookFile])
	
	printTitle("Training Model")
	subprocess.call(['./ThibaultLanglois-VQMM/vqmm', '-quiet', 'y', '-output-dir', modelsDir, '-list-of-files', trainFileList, '-epsilon', '0.00001', '-codebook', codebookFile, '-make-tag-models'])
	os.system("readlink -f $(echo \"" + modelsDir + "*\") >> " + tmpModels)
	os.system("sed -n '/NOT_/!p' " + tmpModels + " >> " + modelsFile)
	os.remove(tmpModels)
	
	printTitle("Testing Model")
	if not os.path.exists(resultsDir):
		os.makedirs(resultsDir)
	printInfo("Approx 515ms per file")
	subprocess.call(['./ThibaultLanglois-VQMM/vqmm', '-tagify', '-output-dir', resultsDir, '-models', modelsFile, '-codebook', codebookFile, '-list-of-files', testFileList])
	
	printTitle("Results:")
	resultFile1 = resultsDir + "filelist.cbkcodebook.summary.txt"
	resultFile2 = resultsDir + "filelist.cbkcodebook.perTag.txt"
	if os.path.isfile(resultFile1):
		with open(resultFile1, 'r') as filename:
			for line in filename:
				print(line[:-1])
	if os.path.isfile(resultFile2):
		with open(resultFile2, 'r') as filename:
			for line in filename:
				print(line[:-1])
	else:
		printError("Error during VQMM, no results to display, see ./analysis/ for more details.")

def preprocess(args):
	printTitle("Starting preprocessing")
	inDIR = args["inDIR"]
	fileWithClass = args["fileWithClass"]
	if inDIR[-1] != '/' and inDIR[-1] != '\\':
		inDIR = inDIR + '/'
	errDIR = inDIR + "error/"
	outDIR = inDIR + "processed/"		
	projName = inDIR[:-1]
	projName = projName[projName.rindex("/")+1:] + "_"
	projName = projName + str(args["cbkSize"]) + "cbkSize_"
	projName = projName + str(args["randSeedCbk"]) + "RandCbk_"
	projName = projName + str(args["randSeedFold"]) + "RandFold_"
	projName = projName + str(args["nbFolds"]) + "Fold"
	if args["nbFolds"] > 1:
		projName = projName + "s"
	if args["invertTrainTest"]:
		projName = projName + "_I"
	args["projName"] = projName
	tmpDIR = "./analysis/" + projName + "/"
	if not os.path.exists("./analysis/"):
		os.makedirs("./analysis/")
	if not os.path.exists(tmpDIR):
		os.makedirs(tmpDIR)
	else:
		printError("A project with same params exists")
	tmpFileNames = tmpDIR + "files.txt"
	fileListWithClassJSON = tmpDIR + "filelist.json"
	fileListWithClass = "./analysis/filelist.csv"

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
		os.remove(tmpFileNames)
		with open(fileWithClass) as f:
			linesWithClass = f.readlines()
		classes = None
		curLine = 0
		for line in linesWithClass:
			curLine = curLine + 1
			sys.stdout.write("\r\t" + str(curLine))
			sys.stdout.flush()
			tmpLine = find_between_r( line, "/", "\t" ) + "\n"
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
		printTitle("Preprocessing done")
	else:
		classes = None
		with open(fileListWithClass, 'r') as fp:
			for line in fp:
				itemPath, itemClass = extractPathAndClass(line)
				if not classes:
					classes = {itemClass: [itemPath]}
				elif not itemClass in classes:
					classes[itemClass] = [itemPath]
				else:
					classes[itemClass].append(itemPath)
		with open(fileListWithClassJSON, 'w') as fp:
			json.dump(classes, fp, sort_keys=True, indent=2)
		printTitle("Files already preprocessed")

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
	tmpArgs = parser.parse_args()
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

	args = {"inDIR":inDIR}
	args["fileWithClass"] = fileWithClass
	args["verbose"] = verbose
	args["invertTrainTest"] = invertTrainTest
	args["nbFolds"] = nbFolds
	args["randSeedFold"] = randSeedFold
	args["randSeedCbk"] = randSeedCbk
	args["cbkSize"] = cbkSize

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
	os.remove(fileListWithClassJSON)
	for key in paths:
		newSize = round(len(paths[key])/nbFolds)
		selected = np.random.choice(paths[key], size=newSize, replace = False)
		selected = [selected, list(set(paths[key]) - set(selected))]
		for i in range(0, nbFolds):
			with open(args["tmpDIR"] + "fold" + str(i+1) + ".csv", "a") as fold:
				try:
					for line in selected[i]:
						fold.write(str(line) + "\t" + str(key) + "\n")
				except:
					printError("The number of folds is greater than the number of data available")
	# if nbFolds > 2:
	# 	if invertTrainTest:
	# 		print("TODO special computation where 1/5 train and 4/5 test")
	# 	else:
	# 		print("TODO special computation where 4/5 train and 1/5 test")

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
	if args["nbFolds"] == 1:
		runVQMM(args)
	else:
		generateFolds(args)
		printWarning("TODO : runVQMM on new folds")
	printInfo("More details available in ./analysis/")
	printTitle("Finished in " + str(int(round(time.time() * 1000)) - begin) + "ms")

if __name__ == "__main__":
	main(sys.argv[1:])
