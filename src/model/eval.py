import sys
import numpy as np
import subprocess
import shlex


def evaluate(trec_eval_path, predictions_file, qrels_file):
    # TODO: add metrics args
    cmd = trec_eval_path + ' {judgement} {output} -m map -m recip_rank -m P.30'.format(
        judgement=qrels_file, output=predictions_file)
    pargs = shlex.split(cmd)
    print('Running {}'.format(cmd))
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()

    if sys.version_info[0] < 3:
        lines = pout.split(b'\n')
    else:
        lines = pout.split(b'\n')

    map = float(lines[0].strip().split()[-1])
    mrr = float(lines[1].strip().split()[-1])
    p30 = float(lines[2].strip().split()[-1])

    return map, mrr, p30

def evaluate_classification(prediction_index_list, labels):
    acc = get_acc(prediction_index_list, labels)
    pre, rec, f1 = get_pre_rec_f1(prediction_index_list, labels)
    return acc, pre, rec, f1

def get_acc(prediction_index_list, labels):
    acc = sum(np.array(prediction_index_list) == np.array(labels))
    return acc / (len(labels) + 1e-9)


def get_pre_rec_f1(prediction_index_list, labels):
    tp, tn, fp, fn = 0, 0, 0, 0
    # print("prediction_index_list: ", prediction_index_list)
    # print("labels: ", labels)
    assert len(prediction_index_list) == len(labels)
    for p, l in zip(prediction_index_list, labels):
        if p == l:
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    eps = 1e-8
    precision = tp * 1.0 / (tp + fp + eps)
    recall = tp * 1.0 / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1