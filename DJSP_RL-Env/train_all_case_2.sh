#!/bin/bash

set -x
RESULT='ray_results'
for ddt_type in Tight Loose
do
    for rpt_effect in Deterministic Gaussian Rework
    do
        ### Baseline
        cd ./args_arxiv
        python3 args_manager.py --args_json=args_case${1}.json --DDT_type=${ddt_type} --RPT_effect=${rpt_effect}
        >&2 cat tmp.json
        cd ..
        experiment_name=Baseline_BJTH_Rule_${rpt_effect}_Case${1}_${ddt_type}
        python3 BJTH_2.py --experiment_name=${experiment_name} --args_json=args_arxiv/tmp.json --job_type_file=Case${1}
        
        ### Ours-CR
        cd ./args_arxiv
        python3 args_manager.py --basic_rule --args_json=args_case${1}.json --DDT_type=${ddt_type} --RPT_effect=${rpt_effect}
        >&2 cat tmp.json
        cd ..
        experiment_name=Ours-CR_Basic_Rule_${rpt_effect}_Case${1}_${ddt_type}
        python3 BJTH_2.py --experiment_name=${experiment_name} --args_json=args_arxiv/tmp.json --job_type_file=Case${1}
        
        ### Ours-JT
        cd ./args_arxiv
        python3 args_manager.py --args_json=args_case${1}.json --DDT_type=${ddt_type} --RPT_effect=${rpt_effect}  
        >&2 cat tmp.json
        cd ..
        experiment_name=Ours-JT-BJTH_Rule_${rpt_effect}_Case${1}_${ddt_type}
        python3 Thesis.py --experiment_name=${experiment_name} --args_json=args_arxiv/tmp.json --job_type_file=Case${1}
        
        ### Ours-CR&JT
        cd ./args_arxiv
        python3 args_manager.py --basic_rule --args_json=args_case${1}.json --DDT_type=${ddt_type} --RPT_effect=${rpt_effect}  
        >&2 cat tmp.json
        cd ..
        experiment_name=Ours-CR-JT-Basic_Rule_${rpt_effect}_Case${1}_${ddt_type}
        python3 Thesis.py --experiment_name=${experiment_name} --args_json=args_arxiv/tmp.json --job_type_file=Case${1}
    done
done
