#!/bin/bash

{
while read SED
do
    BASE=`basename $SED .sed`
    echo $SED
    for EBV in 0.1 0.2 0.3 0.4 0.5
    do
	./redden_SED_Calzetti.sh $SED $EBV
    echo ${BASE}_Calzetti_EBV$EBV.sed
    done
done<COSMOS_MOD_BPZ.list
}>COSMOS_MOD_BPZ_Calzetti.list