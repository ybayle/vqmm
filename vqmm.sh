#!/bin/bash

#
# Author	Yann Bayle
# E-mail	bayle.yann@live.fr
# License   MIT
# Created	06/09/2016
# Updated	09/09/2016
# Version	2
# Object	Write in specified output file invalid ISRCs as defined by
# 			http://isrc.ifpi.org/en/isrc-standard/code-syntax
# Input		The name of the folder containing all resulting files produced by YAAFE
# Output 	Classification results and multiple files in tmp/
# TODOs		- Add tag to each file
# 			- Comment each line
#			- Multiple function
#			- Check if dirs already created
#			- Gather params such randomseed and codebooksize and make superscript with use different value
#			- Choose name of file
#			- Il faut avoir pour chaque file sa classe
#			- Manage vqmm verbose
#			- tell user not to use "NOT_" in his class name or better manage it
#			- Train/Test more folds
#

if [ "$#" -ne 1 ]; then
    echo -e "\e[1m\e[91mIllegal number of parameters.\e[0m"
  	echo -e "\e[1m\e[91mYou must input one parameter which is the file containing paths to all resulting files produced by YAAFE and their corresponding class.\e[0m"
  	exit 1
fi
if [ ! -d "tmp" ]; then
	mkdir tmp
fi
FILELIST="$(pwd)/tmp/CAL500.TOP97.r50.fold1.cbkCAL500.ALL.YMFCC2.r50.s75.csv"
cat $1 >> $FILELIST
RANDSEED=75
CBKSIZE=50
CODEBOOK="$(pwd)/tmp/CAL500.TOP97.r50.fold1.cbk"
RESULTSDIR="$(pwd)/tmp/Results/"
MODELSDIR="$(pwd)/tmp/Models/CAL500.TOP97-r50.s75/"
MODELS="$(pwd)/tmp/CAL500.TOP97.r50.s75.fold1234.cbkCAL500.ALL.YMFCC2.r50.s75.models.csv"
TMPMODELS="$(pwd)/tmp/tmpCAL500.TOP97.r50.s75.fold1234.cbkCAL500.ALL.YMFCC2.r50.s75.models.csv"
if [ ! -d "$(pwd)/tmp/Models" ]; then
	mkdir "$(pwd)/tmp/Models"
fi
if [ ! -d "$MODELSDIR" ]; then
	mkdir $MODELSDIR
fi
cd "$(pwd)/ThibaultLanglois-VQMM/"
echo -e "\e[1m\e[92mCreating VQMM Codebook\e[0m"
./makecodebook.sh $RANDSEED $CBKSIZE $FILELIST $CODEBOOK
echo -e "\e[1m\e[92mTraining Model\e[0m"
./traintagmodels.sh $CODEBOOK $FILELIST $MODELSDIR
if [ ! -d "$RESULTSDIR" ]; then
	mkdir $RESULTSDIR
fi
readlink -f $(echo "$MODELSDIR*") >> $TMPMODELS
sed -n '/NOT_/!p' $TMPMODELS >> $MODELS
echo -e "\e[1m\e[92mTesting Model\e[0m"
./testtagmodels.sh $MODELS $CODEBOOK $FILELIST $RESULTSDIR
cd ..
echo -e "\e[1m\e[92mResults :\e[0m"
cat ./tmp/Results/CAL500.TOP97.r50.fold1.cbkCAL500.TOP97.r50.fold1.summary.txt
cat ./tmp/Results/CAL500.TOP97.r50.fold1.cbkCAL500.TOP97.r50.fold1.perTag.txt
