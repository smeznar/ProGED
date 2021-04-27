#!/bin/bash
# This is script for login node.

# git clone --branch ode https://github.com/brencej/ProGED
cp ProGED/ProGED/examples/oeis/fibonacci_oneby1.py .
cp ProGED/ProGED/examples/oeis/oeis_selection.csv .
cp ProGED/ProGED/examples/oeis/cluster/play.sh .
cp ProGED/ProGED/examples/oeis/cluster/submit.xrsl .
cp ProGED/tests/test_core.py .

# Further details:
# arcsub -c cluster.net submit.xrsl
# or
# ar submit.xrsl  # ar is in myarcrunner/

