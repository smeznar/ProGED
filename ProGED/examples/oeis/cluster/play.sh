#!/bin/bash
# Self replicating machine:
thisfile=~/play.sh
jobfile=~/submit.xrsl
echo " "
echo "I am the script called: "$thisfile
echo "and I was submitted by xrsl file: "$jobfile"."
echo " "
echo " --- first line of "$jobfile" below : --- "
cat $jobfile
echo " --- this was the last line of "$thisfile" above.  --- "
echo " "
echo " --- first line of "$thisfile" below : --- "
cat $thisfile
echo " --- this was the last line of "$thisfile" above.  --- "
echo " "
echo "now for real:"

echo "I (cluster) started to run .sh, first checking the imports: python3 import proged,numpy, itd."
singularity exec ~/pyProGED.simg python3 -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"

echo "I (cluster) am testing proged with pytest ..."
singularity exec ~/pyProGED.simg python3 -m pytest ~/test_core.py

echo "Let's finally execute the big program: "
singularity exec ~/pyProGED.simg python3 ~/fibonacci_oneby1.py --order=2 --is_direct=False --sample_size 50  # online
echo "... This is the end of your personal cluster script. Everything was executed successfully!"

