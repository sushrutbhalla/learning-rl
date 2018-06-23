import numpy as np

def print_policy_word(policy):
    policy_word = []
    for state,action in enumerate(policy):
        if action == 0:
            policy_word.append(str(state)+':up')
        elif action == 1:
            policy_word.append(str(state)+':down')
        elif action == 2:
            policy_word.append(str(state)+':left')
        elif action == 3:
            policy_word.append(str(state)+':right')
        else:
            print ("[ERROR] wrong action chosen in policy: \n{}".format(policy))
            exit(-1)
    assert len(policy) == len(policy_word), "Length of policy_word doesn't match length of policy"
    print ("policy word: \n{}".format(policy_word))
