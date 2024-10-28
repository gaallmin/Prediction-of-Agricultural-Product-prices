items=("배추" "무" "양파" "감자 수미" "대파(일반)" "건고추" "깐마늘(국산)" "상추" "사과" "배")

for item in "${items[@]}"; do
    echo "${item}"
    python -u conv1d_dense.py --item "${item}"
done
