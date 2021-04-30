#!/bin/bash
# This is script for login node.

# git clone --branch ode https://github.com/brencej/ProGED

# cp ProGED/ProGED/examples/oeis/cluster/login.sh .

cp ProGED/ProGED/examples/oeis/fibonacci_oneby1.py .
cp ProGED/ProGED/examples/oeis/oeis_selection.csv .
cp ProGED/ProGED/examples/oeis/cluster/play.sh .
cp ProGED/ProGED/examples/oeis/cluster/arcsuball.sh .
cp ProGED/tests/test_core.py .
general=oeis-
direct=direct
order=ord

cp ProGED/ProGED/examples/oeis/cluster/submit.xrsl . &&

ord0dir=$general$order"0-"$direct".xrsl"
ord2dir=$general$order"2-"$direct".xrsl"
ord4dir=$general$order"4-"$direct".xrsl"
ord2nodir=$general$order"2-no"$direct".xrsl"
ord4nodir=$general$order"4-no"$direct".xrsl"

cp submit.xrsl $ord0dir &&
cp submit.xrsl $ord2dir &&
cp submit.xrsl $ord4dir &&
cp submit.xrsl $ord2nodir &&
cp submit.xrsl $ord4nodir

submitfile=submit.xrsl
scriptfile=play.sh
echo "inside "$submitfile" :"
cat $submitfile | grep $submitfile
echo "inside "$scriptfile" :"
cat $scriptfile | grep $submitfile

# Further details:
# edit play.sh
# arcsub -c cluster.net submit.xrsl
# or
# ars submit.xrsl  # ars is in myarcrunner/

