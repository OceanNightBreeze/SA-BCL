#!/bin/bash
for proj in */; do
  #echo "Doing $proj"
  cd "$proj";
  for dir in *-fault; do
    cd "$dir";
    for f in *.txt; do
      if [ $(sed -n '/#uuts/,$p' "$f" | grep " | " | wc -l)  == "0" ]; then
        echo "$proj $dir $f"
      fi
    done
    cd ../;
  done
  #echo "Done in $proj"
  cd ../;
done #| sort -n | uniq -c > hist.txt;
