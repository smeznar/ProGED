#!/bin/bash
# tar -xf ProGED.tar
# Self replicating machine:
echo "first show current dir aka. pwd"
pwd
thisfile="cluster/play.sh"
echo " "
echo "I am the script called:"$thisfile
echo " "
echo " --- first line of "$thisfile" below : --- "
cat $thisfile
echo " --- this was the last line of "$thisfile" above.  --- "
echo " "
git clone --branch ode https://github.com/brencej/ProGED clusterProGED/
echo "I (cluster) started to run .sh, first checking the imports: python3 import proged,numpy, itd."
# singularity exec python-proged.simg python3 -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"
# singularity exec pyProGED.simg python3 -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"
singularity exec ~/pyProGED.simg python3 -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"

echo "I (cluster) am testing proged with pytest ..."
# singularity exec python-proged.simg python3 -m pytest ProGED/tests/
singularity exec pyProGED.simg python3 -m pytest clusterProGED/tests/
# singularity exec ~/pyproged python3 -m pytest ../../../tests/test_core.py
# singularity exec ~/pyProGED.simg python3 -m pytest clusterProGED/tests/test_core.py
# singularity exec ~/pyProGED.simg python3 -m pytest ../../../tests/test_core.py
echo "Let's finally execute the big program: "
# First copy oeis data to current dir to avoid (no file) error.
cp clusterProGED/ProGED/examples/oeis/oeis_selection.csv .
singularity exec pyProGED.simg python3 clusterProGED/ProGED/examples/oeis/fibonacci_oneby1.py
# singularity exec pyProGED.simg python3 fibonacci_oneby1.py  # online
# singularity exec ~/pyProGED.simg python3 fibonacci_oneby1.py  # offline
echo "... This is the end of your personal cluster script. Everything was executed successfully!"

