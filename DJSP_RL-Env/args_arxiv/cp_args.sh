set -x
for i in $(seq 4 13)
do
    cp -v 'args.json' 'args_case'${i}'.json'
    # echo ${i}
done