import copy
import heapq
import itertools
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from tqdm import tqdm

from dataset import *
from utils import *
from models import *

criteria = torch.nn.CrossEntropyLoss()

def main():
    set_logger()
    seed = 2
    set_seed(seed)
    logging.info('Client Side PFL Training Starts')
    global_iterations = 40
    local_epoch = 5
    # algorithm = 'sv-no-delta-no-freespace' # sv-...-...-...-... / fomo / local
    # algorithm = 'sv-loss-no-delta-no-freespace'
    algorithm = 'fomo'
    R = 3
    sv_eval_method = 'acc' if 'acc' in algorithm else 'loss'
    whether_delta = not 'no-delta' in algorithm
    whether_free_space = not 'no-freespace' in algorithm

    print('-' * 50)
    print('Experience Details:')
    print('Algorithm: {}'.format(algorithm))
    if 'sv' in algorithm:
        print('SV-Eval-Method: {}'.format(sv_eval_method))
        print('Delta: {}'.format(whether_delta))
        print('Free-Space: {}'.format(whether_free_space))
    print('Seed: {}'.format(seed))
    print('Global Iterations: {}'.format(global_iterations))
    print('Local Epoch: {}'.format(local_epoch))
    print('-' * 50)

    data = 'cifar10'
    path = '../data'
    batch_size = 40
    clients = 30
    class_per_user = 2

    # participant
    usr_dataset_loaders, usr_val_loaders, usr_test_loaders = gen_random_loaders(data, path, clients, batch_size, class_per_user)
    print(len(usr_dataset_loaders))
    users = [i for i in range(clients)]
    user_models = [CNNCifar() for i in users]
    user_optimizers = [torch.optim.SGD(user_models[i].parameters(), lr=0.1, momentum=0, weight_decay=1e-4)
                      for i in users]
    sv_eval_model = CNNCifar()

    # 100 seed 42
    # allocation = [[7, 3], [2, 0], [6, 8], [4, 9], [5, 1], [8, 2], [0, 6], [3, 4], [5, 7], [1, 9], [8, 3], [0, 5], [1, 9], [7, 6], [2, 4], [0, 1], [9, 5], [4, 6], [3, 8], [7, 2], [5, 6], [4, 2], [9, 3], [7, 1], [0, 8], [0, 1], [6, 5], [8, 4], [9, 7], [3, 2], [0, 1], [5, 6], [9, 8], [3, 2], [7, 4], [2, 9], [5, 6], [0, 8], [4, 7], [3, 1], [0, 4], [8, 6], [7, 1], [9, 5], [3, 2], [1, 9], [2, 8], [7, 3], [5, 4], [6, 0], [1, 6], [8, 2], [4, 3], [5, 7], [0, 9], [7, 0], [1, 4], [9, 8], [3, 6], [5, 2], [2, 1], [0, 7], [6, 4], [8, 3], [9, 5], [3, 0], [9, 7], [8, 5], [4, 6], [1, 2], [1, 2], [9, 8], [7, 4], [0, 5], [3, 6], [5, 6], [0, 4], [2, 7], [9, 1], [3, 8], [7, 4], [3, 5], [6, 9], [2, 1], [0, 8], [3, 2], [8, 5], [7, 1], [9, 4], [6, 0], [2, 0], [9, 8], [4, 6], [3, 7], [5, 1], [4, 5], [7, 9], [0, 6], [1, 2], [8, 3]]
    # 50 seed 42
    # allocation = [[6, 0], [4, 7], [5, 9], [3, 8], [1, 2], [3, 7], [2, 4], [9, 6], [1, 5], [0, 8], [8, 4], [6, 3], [5, 7], [9, 1], [2, 0], [9, 6], [3, 0], [8, 1], [2, 4], [5, 7], [8, 3], [5, 0], [4, 6], [1, 7], [2, 9], [0, 3], [2, 6], [5, 1], [9, 4], [8, 7], [0, 4], [5, 8], [2, 6], [1, 3], [7, 9], [7, 0], [3, 1], [4, 5], [9, 6], [2, 8], [4, 0], [1, 5], [9, 3], [2, 7], [8, 6], [5, 7], [3, 9], [2, 4], [0, 8], [6, 1]]
    # 30 seed 42
    # allocation = [[1, 5], [9, 2], [0, 7], [8, 3], [4, 6], [0, 9], [7, 1], [2, 6], [8, 5], [3, 4], [2, 0], [9, 4], [5, 1], [7, 3], [8, 6], [6, 9], [3, 8], [0, 5], [2, 1], [7, 4], [6, 7], [5, 4], [2, 8], [9, 1], [0, 3], [2, 0], [4, 7], [9, 5], [1, 3], [8, 6]]
    # 10 seed 42
    # allocation = [[9, 2], [4, 8], [5, 0], [6, 1], [3, 7], [6, 4], [0, 9], [2, 7], [1, 5], [8, 3]]
    # 15 seed 42

    # 30 seed 2
    allocation = [[4, 6], [2, 0], [5, 8], [9, 1], [3, 7], [5, 8], [3, 7], [4, 0], [1, 9], [2, 6], [5, 1], [9, 8], [7, 3], [6, 0],
     [4, 2], [6, 1], [5, 3], [7, 0], [8, 2], [4, 9], [5, 4], [9, 3], [2, 1], [8, 6], [7, 0], [2, 1], [9, 5], [0, 4],
     [3, 6], [7, 8]]
    # allocation = [[2, 7], [5, 0], [3, 8], [1, 6], [9, 4], [9, 8], [1, 5], [2, 6], [7, 4], [3, 0], [6, 8], [7, 2], [9, 0], [5, 3], [4, 1]]
    relationship = {i: [] for i in range(clients)}
    findRelationship(allocation, relationship)
    print(relationship)

    # indicate user
    user_id = 0
    partner = relationship[user_id]

    # partcipant training:
    user_state_lst = [0 for i in range(clients)]
    for id in partner:
        user_state_lst[id] = 1
    user_state_lst[user_id] = 1

    wacc_history = []
    wloss_history = []
    user_only_loss, user_only_acc = evaluate_model(user_models[user_id], test_loader=usr_test_loaders[user_id])
    wacc_history.append(user_only_acc)
    wloss_history.append(user_only_loss)

    for iter in range(1, global_iterations + 1):
        for id in users:
            if user_state_lst[id] == 1:
                print('ID: {} Local Training'.format(id))
                # Local training
                cur_loss, cur_acc = evaluate_model(user_models[id], test_loader=usr_test_loaders[id])
                print('Before Acc: {}, Loss {}'.format(cur_acc * 100, cur_loss))
                for epoch in range(1, local_epoch + 1):
                    for i, data in enumerate(usr_dataset_loaders[id], 0):
                        user_models[id].train()
                        user_optimizers[id].zero_grad()
                        img, label = data
                        pred = user_models[id](img)
                        loss = criteria(pred, label)
                        loss.backward()
                        user_optimizers[id].step()
                cur_loss, cur_acc = evaluate_model(user_models[id], test_loader = usr_test_loaders[id])
                print('After Acc: {}, Loss {}'.format(cur_acc * 100, cur_loss))

        if algorithm == 'local':
            wacc_history.append(cur_acc)
            wloss_history.append(cur_loss)

        length = len(partner)
        # fomo_loss
        if 'fomo' in algorithm.lower():
            print('-'*80)
            print('start fomo')
            user_only_loss, user_only_acc = evaluate_model(user_models[user_id], test_loader = usr_test_loaders[user_id])
            loss_diff_lst = []
            for p in partner:
                cur_loss, cur_acc = evaluate_model(user_models[p], test_loader = usr_test_loaders[user_id])
                loss_diff_lst.append(user_only_loss - cur_loss)

            # positive_loss
            for i in range(length):
                loss_diff_lst[i] = 0 if loss_diff_lst[i] < 0 else loss_diff_lst[i]

            # parameter_diff
            para_diff_lst = []
            for p in partner:
                para_diff = []
                para_diff = torch.Tensor(para_diff)
                for para_usr, para_p in zip(user_models[user_id].parameters(), user_models[p].parameters()):
                    para_diff = torch.cat((para_diff, ((para_usr - para_p).view(-1))), 0)
                para_diff_lst.append(torch.norm(para_diff))

            # weights
            weights = []
            for i in range(length):
                weights.append(loss_diff_lst[i] / para_diff_lst[i])

            print(weights)

            for i in range(length):
                p = partner[i]
                for para_usr, para_p in zip(user_models[user_id].parameters(), user_models[p].parameters()):
                    para_usr.data += (para_p.data.clone() - para_usr.data.clone()) * weights[i]

            cur_loss, cur_acc = evaluate_model(user_models[user_id], test_loader=usr_test_loaders[user_id])

            wacc_history.append(cur_acc)
            wloss_history.append(cur_loss)
            print('Before fomo Acc: {}, Loss {}'.format(user_only_acc * 100, user_only_loss))
            print('After fomo Acc: {}, Loss {}'.format(cur_acc * 100, cur_loss))
            print('-'*80)
        # sv
        if 'sv' in algorithm.lower():
            print('-'*80)
            user_only_loss, user_only_acc = evaluate_model(user_models[user_id], test_loader=usr_test_loaders[user_id])
            length = len(partner)
            participate = []
            participate.extend(partner)
            participate.append(user_id)
            perm_list = []
            perm_list += list(itertools.permutations(participate, length + 1))
            random_perm_index_list = np.random.choice([i for i in range(len(perm_list))], R * (length + 1), replace=False)
            random_perm = [perm_list[i] for i in random_perm_index_list]
            print('[SV] Random Select: {}'.format(len(random_perm)))
            evaluate_sv_info_dict = {i: [] for i in participate}
            evaluated_sv_dict = {i: 0. for i in participate}
            for item in random_perm:
                print('[SV] Processing: {}'.format(item))
                evaluated_sv = []
                weights_queue = []
                node_queue = []
                acc_history = [0.]
                loss_history = [2.4]
                for member in item:
                    node_queue.append(member)
                    weights_queue.append(user_models[member].state_dict())
                    avg = average_weights(weights_queue)
                    sv_eval_model.load_state_dict(avg)
                    cur_loss, cur_acc = evaluate_model(sv_eval_model, usr_test_loaders[user_id])
                    if sv_eval_method == 'acc':
                        # logging.info('[EVAL] {}({}) on {} with accuracy influence {} - {} = {}%'.format(member, len(node_queue), idx, cur_acc * 100, acc_history[-1] * 100, (cur_acc - acc_history[-1]) * 100))
                        evaluate_sv_info_dict[member].append(cur_acc - acc_history[-1])
                        acc_history.append(cur_acc)
                    if sv_eval_method == 'loss':
                        # logging.info('[EVAL] {}({}) on {} with loss influence {} - {} = {}'.format(member, len(node_queue), idx, loss_history[-1], cur_loss, loss_history[-1] - cur_loss))
                        evaluate_sv_info_dict[member].append(loss_history[-1] - cur_loss)
                        loss_history.append(cur_loss)

            print('[SV] SV RECORDS {}'.format(evaluate_sv_info_dict))
            for i in participate:
                evaluated_sv_dict[i] = np.mean(evaluate_sv_info_dict[i])
            print('[SV] MEAN SV: {}'.format(evaluated_sv_dict))

            # parameter_diff
            models_difference = {i: 0. for i in participate}
            for p in participate:
                if p != user_id:
                    para_diff = []
                    para_diff = torch.Tensor(para_diff)
                    for para_usr, para_p in zip(user_models[user_id].parameters(), user_models[p].parameters()):
                        para_diff = torch.cat((para_diff, ((para_usr - para_p).view(-1))), 0)
                    models_difference[p] = torch.norm(para_diff)
                else:
                    models_difference[p] = 1.

            # Positive sv:
            positive_idx = []
            positive_sv = []
            for i in participate:
                if evaluated_sv_dict[i] > 0:
                    positive_idx.append(i)
                    positive_sv.append(evaluated_sv_dict[i] / models_difference[i])
            positive_sv = [i / sum(positive_sv) for i in positive_sv]
            print('[SV] Positive Index {}'.format(positive_idx))
            print('[SV] Positive Weights {}'.format(positive_sv))

            # if not using delta update itself param firstly
            free_space = 1.0 - positive_sv[positive_idx.index(user_id)]
            self_weight = positive_sv[-1]
            if not whether_delta:
                print('[SV] Hold {}% self info'.format(self_weight * 100))
                for para in user_models[user_id].parameters():
                    para.data = para.data.clone() * self_weight

            # weather using free_space
            weights = {i: 0. for i in participate}
            for i in participate:
                if i in positive_idx and i != user_id:
                    if whether_free_space:
                        weights[i] = free_space * positive_sv[positive_idx.index(i)]  # add free_space
                    else:
                        weights[i] = positive_sv[positive_idx.index(i)]  # not add free_space

            # wather using delta
            for id in participate:
                for param, param_request in zip(user_models[user_id].parameters(), user_models[id].parameters()):
                    if whether_delta:
                        param.data += (param_request.data.clone() - param.data.clone()) * weights[id]
                    else:
                        param.data += param.data.clone() * weights[id]
            cur_loss, cur_acc = evaluate_model(user_models[user_id], test_loader = usr_test_loaders[user_id])
            wloss_history.append(cur_loss)
            wacc_history.append(cur_acc)
            print('Before sv Acc: {}, Loss {}'.format(user_only_acc * 100, user_only_loss))
            print('After sv Acc: {}, Loss {}'.format(cur_acc * 100, cur_loss))
            print('-' * 80)

    with open('ACC_LE{}_GI{}_SEED{}_{}.csv'.format(local_epoch, global_iterations, seed, algorithm), 'w') as f:
        f.write('{}, '.format(algorithm))
        f.write('{}'.format(wacc_history).strip('[]'))

    with open('LOSS_LE{}_GI{}_SEED{}_{}.csv'.format(local_epoch, global_iterations, seed, algorithm), 'w') as f:
        f.write('{}, '.format(algorithm))
        f.write('{}'.format(wloss_history).strip('[]'))


def findRelationship(allocation, relationship):
    for i in range(len(allocation)):
        for j in range(len((allocation))):
            join = set()
            if i != j:
                join = set(allocation[i]) & set(allocation[j])
            if len(join) != 0:
                relationship[i].append(j)


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def evaluate_model(model, test_loader):
    loss, correct, samples = 0., 0., 0.
    batch_count = 0
    for batch_count, batch in enumerate(test_loader):
        img, label = tuple(batch)
        pred = model(img)
        loss += criteria(pred, label).item()
        correct += pred.argmax(1).eq(label).sum().item()
        samples += len(label)
    loss /= batch_count + 1
    return loss, correct / samples


if __name__ == '__main__':
    main()

