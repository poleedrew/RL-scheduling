import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Manager')
    parser.add_argument('--RPT_effect', type=str, default='Deterministic', help='Processing Time Effect')
    parser.add_argument('--DDT_type', type=str, default='Loose', help='Due Day Tightness type: Loose, Tight')
    parser.add_argument('--args_json', type=str, default='args.json', help='argument file')
    parser.add_argument('--basic_rule', action='store_true')
    args = parser.parse_args()
    with open(args.args_json, newline='') as jsonfile:    
        data = json.load(jsonfile)
        ### RPT effect
        if args.RPT_effect == 'Deterministic':
            data['RPT_effect']['flag'] = False
        if args.RPT_effect == 'Gaussian':
            data['RPT_effect']['flag'] = True
            data['RPT_effect']['type'] = 'Gaussian'
        if args.RPT_effect == 'Rework':
            data['RPT_effect']['flag'] = True
            data['RPT_effect']['type'] = 'Rework'
        
        ### Basic Rule or BJTH Rule
        if args.basic_rule:
            data['ENV']['basic_rule'] = True
        else:
            data['ENV']['basic_rule'] = False

        ### Due Day Tightness
        if args.DDT_type == 'Tight':
            data['DDT'] = 1.0
            data['DDT_dt'] = 0.5
            data['Mean_of_Arrival'] = 25
        if args.DDT_type == 'Loose':
            data['DDT'] = 1.5
            data['DDT_dt'] = 0.8
            data['Mean_of_Arrival'] = 50
        with open('tmp.json', 'w') as f:
            print(data)
            json.dump(data, f, indent=4)
        
