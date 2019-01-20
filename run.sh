mkdir -p output
type=uint8
for type in bool uint8 uint32; do
echo "Type $type"
echo "--Small"
python main.py data/grammar_small.txt data/graph_small.txt -s -t $type > output/res_small.txt && python compare_answers.py output/res_small.txt data/correct_output_small.txt
echo "--Big"
python main.py data/grammar_big.txt data/graph_big.txt -s -t $type > output/res_big.txt && python compare_answers.py output/res_big.txt data/correct_output_big.txt
echo "----------"
done
