#!/bin/bash

set -x
RESULT='ray_results'
for i in {13..4..-1}
do 
    for ddt_type in Tight Loose
    do
        for rpt_effect in Deterministic Gaussian Rework
        do
            echo ${rpt_effect}_Case${i}_${ddt_type}
            cd ./args_arxiv
            python3 args_manager.py --args_json=args_case${i}.json --DDT_type=${ddt_type} --RPT_effect=${rpt_effect}
            # >&2 cat tmp.json
            cd ..
            python3 test_heuristic.py --args_json=args_arxiv/tmp.json --job_type_file=./test_instance/Case${i}
            echo '\n'
        done
    done
done