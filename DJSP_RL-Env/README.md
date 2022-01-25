# DJSP_RL
https://hackmd.io/@CGI-Lab/BkahAircY

## Training
``` bash
./train_all_case.sh
### or
### ./train_all_case_2.sh <case_num(4~13)>
### example
./train_all_case_2.sh 6
```

## Testing
```
cd ray_result
### python3 parse_log.py --log_path=<checkpoint_path>
### example
python3 parse_log.py --log_path=Baseline_BJTH_Rule_Gaussian_Case12_Tight/DQN_djsp_env_6ab6e_00000_0_2022-01-25_07-00-25
```

## 數據紀錄
https://docs.google.com/spreadsheets/d/11xHSxEhXY83JCEHfiYJKPGUT_3QbxfE6JYzEM2iGakk/edit#gid=977726332
