#!/bin/bash
# This is script for login node.

pwd
ls
cd ProGED
pwd
ls
cd ..
git clone --branch ode https://github.com/brencej/ProGED
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

