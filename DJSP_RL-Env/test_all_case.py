import os

if __name__ == '__main__':
    path = 'ray_results'
    depth = 1

    checkpoints = []
    for root, dirs, files in os.walk(path):
        if root[len(path):].count(os.sep) == depth:
            for d in dirs:
                # if ('Baseline' in d) or ('Our(CR)' in d):
                #     print(os.path.join(root,d))
                checkpoint = os.path.join(root,d)
                print(checkpoint)
                # checkpoints.append(os.path.join(root,d))
    # print(checkpoints)
        