import os

if __name__ == '__main__':
    path = 'ray_results'
    depth = 1
    # for d in os.listdir(path):
    #     print(d)
    #     if not os.path.isdir(d):
    #         continue
        
    #     for c in os.listdir(d):
    #         print('\t', c)
    #         if c[:3] != 'DQN':
    #             continue
    #         checkpoint_path = os.path.join(d, c)
    #         print(checkpoint_path)

    checkpoints = []
    for root, dirs, files in os.walk(path):
        if root[len(path):].count(os.sep) == depth:
            for d in dirs:
                # if ('Baseline' in d) or ('Our(CR)' in d):
                #     print(os.path.join(root,d))
                checkpoint = os.path.join(root,d)
                if ('Baseline' in checkpoint) or ('Ours(CR)' in checkpoint):
                    print(checkpoint)
                # checkpoints.append(os.path.join(root,d))
    # print(checkpoints)
        