#!/bin/sh

export tmp1=`mktemp`

for i in `find`
#`ls -d a/a*`
do
       if [ -f $i ]
       then
               md5sum $i >> $tmp1
       fi
done

echo dups:
export dups=`mktemp`
sort $tmp1 | awk '{if (prev == $1) {print ;} prev = $1;}' >> $dups

cat $dups

