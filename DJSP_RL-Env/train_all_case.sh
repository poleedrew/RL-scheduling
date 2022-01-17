set -x
for i in $(seq 4 13)
do 
    python3 test_heuristic.py --args_json=args_arxiv/args_case${i}.json --job_type_file=Case${i} >> heur_out.txt
    python3 Thesis.py --args_json=args_arxiv/args_case${i}.json --job_type_file=Case${i}
done