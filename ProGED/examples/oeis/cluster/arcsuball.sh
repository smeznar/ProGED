#!/bin/bash
# This is script for login node.
general=oeis-
direct="direct"
order=ord

aru $general$order"0-"$direct".xrsl" &&
aru $general$order"2-"$direct".xrsl" &&
aru $general$order"4-"$direct".xrsl" &&
aru $general$order"2-no"$direct".xrsl" &&
aru $general$order"4-no"$direct".xrsl"
# for i in $general$order[024]-*$direct".xrsl"; do echo ar $i; done
