#!bin/bash -i

for i in sli*.vtex; do echo cp $i sharelatex/${i::-4}txt; done
# for i in sli*.vtex; do cp $i sharelatex/${i::-4}txt; done
