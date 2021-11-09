import numpy as np
from collections import Counter
from scipy.special import softmax


def get_accuracies(class_seen, class_correct):
    total_seen = sum(class_seen.values())
    accuracy = sum(class_correct.values()) / total_seen

    avg_class_acc = 0.
    w_class_acc = 0.

    # python order of values guaranteed
    class_probs = softmax(class_seen.values())
    for i, (c, c_seen) in enumerate(class_seen.items()):
        avg_class_acc += (1/len(class_seen)) * class_correct[c] / c_seen
        w_class_acc += class_probs[i] * class_correct[c] / c_seen

    return {'accuracy': accuracy,
            'average class accuracy': avg_class_acc,
            'weighted class accuracy': w_class_acc}
