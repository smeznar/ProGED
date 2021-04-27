#!/bin/bash
# This is script for login node.

# git clone --branch ode https://github.com/brencej/ProGED
# cp ProGED/ProGED/examples/oeis/fibonacci_oneby1.py .
# cp ProGED/ProGED/examples/oeis/oeis_selection.csv .
# cp ProGED/ProGED/examples/oeis/cluster/play.sh .
# cp ProGED/ProGED/examples/oeis/cluster/submit.xrsl .
# cp ProGED/tests/test_core.py .


# play.sh:


#!/bin/bash
# Self replicating machine:
echo "first show current dir aka. pwd"
thisfile="~/play.sh"
echo " "
echo "I am the script called:"$thisfile
echo " "
echo " --- first line of "$thisfile" below : --- "
cat $thisfile
echo " --- this was the last line of "$thisfile" above.  --- "
echo " "

echo "one more try for git clone, ignore this"
git clone --branch ode https://github.com/brencej/ProGED clusterProGED/
ls 
echo "now for real:"

echo "I (cluster) started to run .sh, first checking the imports: python3 import proged,numpy, itd."
singularity exec ~/pyProGED.simg python3 -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"

echo "I (cluster) am testing proged with pytest ..."
singularity exec ~/pyProGED.simg python3 -m pytest ~/test_core.py

echo "Let's finally execute the big program: "
# First copy oeis data to current dir to avoid (no file) error.
singularity exec ~/pyProGED.simg python3 ~/fibonacci_oneby1.py  # online
echo "... This is the end of your personal cluster script. Everything was executed successfully!"

