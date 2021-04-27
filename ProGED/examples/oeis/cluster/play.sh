#!/bin/bash
# tar -xf ProGED.tar
echo "I (cluster) started to run .sh, first checking the imports: python3 import proged,numpy, itd."
# singularity exec python-proged.simg python3 -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"
# singularity exec pyProGED.simg python -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"
singularity exec ~/pyproged python -c "import ProGED, numpy, pandas, scipy, sympy, nltk, hyperopt, sklearn, pytest;print('Importing seems to work.')"

echo "I (cluster) am testing proged with pytest ..."
# singularity exec python-proged.simg python3 -m pytest ProGED/tests/
# singularity exec pyProGED.simg python -m pytest ProGED/tests/
singularity exec ~/pyproged python -m pytest ../../../../tests/test_core.py
echo "Let's finally execute the big program: "
# First copy oeis data to current dir to avoid (no file) error.
# cp ProGED/ProGED/examples/oeis/oeis_selection.csv .
# singularity exec pyProGED.simg python ProGED/ProGED/examples/oeis/fibonacci_oneby1.py
# singularity exec pyProGED.simg python fibonacci_oneby1.py
singularity exec ~/pyproged python fibonacci_oneby1.py
echo "... This is the end of your personal cluster script. Everything was executed successfully!"

