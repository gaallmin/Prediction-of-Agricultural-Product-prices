items=('건고추' '사과' '감자' '배' '깐마늘(국산)' '무' '상추' '배추' '양파' '대파')

for item in ${items[@]}
do
    python optimizer.py ${item} > "./optimize_results/${item}.out"
done
