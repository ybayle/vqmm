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
#			2 In temp/ folder :
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
#				vqmm.vqmm(lot of args non mandatory)
# 

import time
begin = int(round(time.time() * 1000))
import sys, getopt
import os
import csv
import shutil 
from os import listdir
from os.path import isfile, join
import subprocess

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

def validAndConvertFile(inDIR, outDIR, errDIR, filename):
	inFileName = inDIR + filename
	errFileName = errDIR + filename
	# Remove empty file or with too few lines to be analyzed
	if os.stat(inFileName).st_size > 4000:
		# Above line is slightly faster (20µs) than 
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
								str2write = str2write + "{0:.14f}".format(float(col)) + " "
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
	if PRINTDEBUG:
		print(bcolors.BOLD + bcolors.ERROR + msg + bcolors.ENDC)

def printTitle(msg):
	if PRINTDEBUG:
		print(bcolors.BOLD + bcolors.OKGREEN + msg + bcolors.ENDC)

def printInfo(msg):
	if PRINTDEBUG:
		print(bcolors.OKBLUE + msg + bcolors.ENDC)

def usage():
	printError('You must indicate the folder where raw data from YAAFE are stored and the class file:')
	printError('vqmm.py ./data/ ./filelist.txt')

def runVQMM(tmpDIR, fileListWithClass):
	randomSeed = "75"
	codebookSize = "50"
	codebookFile = tmpDIR + "CAL500.TOP97.r50.fold1.cbk"
	resultsDir = tmpDIR + "Results/"
	modelsDir = tmpDIR + "Models/CAL500.TOP97-r50.s75/"
	modelsFile = tmpDIR + "CAL500.TOP97.r50.s75.fold1234.cbkCAL500.ALL.YMFCC2.r50.s75.models.csv"
	tmpModels = tmpDIR + "tmpCAL500.TOP97.r50.s75.fold1234.cbkCAL500.ALL.YMFCC2.r50.s75.models.csv"
	if not os.path.exists(tmpDIR + "Models/"):
		os.makedirs(tmpDIR + "Models/")
	if not os.path.exists(modelsDir):
		os.makedirs(modelsDir)
	
	printTitle("Compiling VQMM")
	os.system("make -C ./ThibaultLanglois-VQMM/")

	printTitle("Creating VQMM Codebook")
	subprocess.call(['./ThibaultLanglois-VQMM/vqmm', '-quiet', 'y', '-list-of-files', fileListWithClass, '-random', randomSeed, '-codebook-size', codebookSize, '-codebook', codebookFile])
	
	printTitle("Training Model")
	subprocess.call(['./ThibaultLanglois-VQMM/vqmm', '-quiet', 'y', '-output-dir', modelsDir, '-list-of-files', fileListWithClass, '-epsilon', '0.00001', '-codebook', codebookFile, '-make-tag-models'])
	os.system("readlink -f $(echo \"" + modelsDir + "*\") >> " + tmpModels)
	os.system("sed -n '/NOT_/!p' " + tmpModels + " >> " + modelsFile)
	
	printTitle("Testing Model")
	if not os.path.exists(resultsDir):
		os.makedirs(resultsDir)
	printInfo("Approx 515ms per file")
	subprocess.call(['./ThibaultLanglois-VQMM/vqmm', '-tagify', '-output-dir', resultsDir, '-models', modelsFile, '-codebook', codebookFile, '-list-of-files', fileListWithClass])
	
	printTitle("Results:")
	resultFile1 = resultsDir + "CAL500.TOP97.r50.fold1.cbkCAL500.TOP97.r50.fold1.summary.txt"
	resultFile2 = resultsDir + "CAL500.TOP97.r50.fold1.cbkCAL500.TOP97.r50.fold1.perTag.txt"
	if os.path.isfile(resultFile1):
		with open(resultFile1, 'r') as filename:
			for line in filename:
				print(line[:-1])
	if os.path.isfile(resultFile2):
		with open(resultFile2, 'r') as filename:
			for line in filename:
				print(line[:-1])
	else:
		printError("Error during VQMM, no results to display, see ./temp/ for more details.")

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
	inDIR = ''
	if len(sys.argv) == 2:
		if sys.argv[1] != '-h':
			print(bcolors.ERROR)
			usage()
			sys.exit()
		else:
			print(bcolors.OKBLUE)
			usage()
			sys.exit()
	elif len(sys.argv) == 3:
		inDIR = sys.argv[1]
		fileWithClass = sys.argv[2]
	else:
		print(bcolors.ERROR)
		usage()
		sys.exit()
	printInfo("Approx. 700ms per file: go grab a tea!")
	printTitle("Starting preprocessing")
	if inDIR[-1] != '/' or inDIR[-1] != '\\':
		inDIR = inDIR + '/'
	outDIR = inDIR + "processed/"
	errDIR = inDIR + "error/"
	tmpDIR = "./temp/"
	if not os.path.exists(tmpDIR):
		os.makedirs(tmpDIR)
	tmpFileNames = tmpDIR + "files.txt"
	os.system("ls " + inDIR + " > " + tmpFileNames)
	# Previous line is 32 times faster than
	# filesInDir = [f for f in listdir(inDIR) if isfile(join(inDIR, f))]
	if not os.path.exists(outDIR):
		os.makedirs(outDIR)
	printTitle("Validating and converting files")
	with open(tmpFileNames, 'r') as filenames:
		curFileNum = 0
		for filename in filenames:
			curFileNum = curFileNum + 1
			sys.stdout.write("\r\t" + str(curFileNum))
			sys.stdout.flush()
			filename = filename[:-1]
			validAndConvertFile(inDIR, outDIR, errDIR, filename)
		sys.stdout.write('\n')
		sys.stdout.flush()
	printTitle("Associating classes")
	os.system("ls " + outDIR + " > " + tmpFileNames)
	with open(tmpFileNames) as f:
		linesNoClass = f.readlines()
	with open(fileWithClass) as f:
		linesWithClass = f.readlines()
	fileListWithClass = tmpDIR + "CAL500.TOP97.r50.fold1.cbkCAL500.ALL.YMFCC2.r50.s75.csv"
	resultFile = open(fileListWithClass, "w")
	try:
		curLine = 0
		for line in linesWithClass:
			curLine = curLine + 1
			sys.stdout.write("\r\t" + str(curLine))
			sys.stdout.flush()
			tmpLine = find_between_r( line, "/", "\t" ) + "\n"
			if tmpLine in linesNoClass:
				resultFile.write(line)
	finally:
		resultFile.close()    
		sys.stdout.write('\n')
	printTitle("Preprocessing done")
	printTitle("VQMM launched")
	runVQMM(tmpDIR, fileListWithClass)
	# TODO not rm tmpDIR because FileListWithClass needed for VQMM
	# shutil.rmtree(tmpDIR)
	printTitle("Finished")
	printInfo("More details available in ./temp/Results/")

if __name__ == "__main__":
	main(sys.argv[1:])

print("Time to compute: " + str(int(round(time.time() * 1000)) - begin) + "ms")
