#!/bin/bash
# This is script for login node.

# git clone --branch ode https://github.com/brencej/ProGED
cp ProGED/ProGED/examples/oeis/fibonacci_oneby1.py .
cp ProGED/ProGED/examples/oeis/oeis_selection.csv .
cp ProGED/ProGED/examples/oeis/cluster/play.sh .
cp ProGED/ProGED/examples/oeis/cluster/login.sh .
cp ProGED/ProGED/examples/oeis/cluster/arcsuball.sh .
cp ProGED/tests/test_core.py .
general=oeis-
direct=direct
order=-ord

cp ProGED/ProGED/examples/oeis/cluster/submit.xrsl" . &&
echo cp submit.xrsl $general$order"2-no"$direct".xrsl" &&
echo cp submit.xrsl $general$order"4-no"$direct".xrsl" &&
echo cp submit.xrsl $general$order"0-"$direct".xrsl" &&
echo cp submit.xrsl $general$order"2-"$direct".xrsl" &&
echo cp submit.xrsl $general$order"4-"$direct".xrsl"

# Further details:
# arcsub -c cluster.net submit.xrsl
# or
# ar submit.xrsl  # ar is in myarcrunner/

