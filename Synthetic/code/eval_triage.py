import sys
from myutil import *
from triage_human_machine import triage_human_machine
import getopt


def parse_command_line_input(list_of_option, list_of_file_name):
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 's:l:f:', ['std', 'lamb', 'file_name'])

    std = 0.001
    lamb = 0.005
    file_name = 'Gauss'

    for opt, arg in opts:
        if opt == '-s':
            std = float(arg)

        if opt == '-l':
            lamb = float(arg)

        if opt == '-f':
            for file_name_i in list_of_file_name:
                if file_name_i.startswith(arg):
                    file_name = file_name_i

    return std, lamb, file_name


class eval_triage:
    def __init__(self, data_file, real_flag=None, real_wt_std=None):
        self.data = load_data(data_file)
        self.real = real_flag
        self.real_wt_std = real_wt_std

    def eval_loop(self, param, res_file, option):
        res = load_data(res_file, 'ifexists')
        for std in param['std']:
            if self.real:
                data_dict = self.data
                triage_obj = triage_human_machine(data_dict, self.real)
            else:
                if self.real_wt_std:
                    data_dict = {'X': self.data['X'], 'Y': self.data['Y'], 'c': self.data['c'][str(std)]}
                    triage_obj = triage_human_machine(data_dict, self.real_wt_std)
                else:
                    test = {'X': self.data.Xtest, 'Y': self.data.Ytest,
                            'human_pred': self.data.human_pred_test[str(std)]}
                    data_dict = {'test': test, 'dist_mat': self.data.dist_mat, 'X': self.data.Xtrain,
                                 'Y': self.data.Ytrain, 'human_pred': self.data.human_pred_train[str(std)]}
                    triage_obj = triage_human_machine(data_dict, False)
            if str(std) not in res:
                res[str(std)] = {}
            for K in param['K']:
                if str(K) not in res[str(std)]:
                    res[str(std)][str(K)] = {}
                for lamb in param['lamb']:
                    if str(lamb) not in res[str(std)][str(K)]:
                        res[str(std)][str(K)][str(lamb)] = {}
                    print 'std-->', std, 'K--> ', K, ' Lamb--> ', lamb
                    res_dict = triage_obj.algorithmic_triage({'K': K, 'lamb': lamb, 'DG_T': param['DG_T']},
                                                             optim=option)
                    res[str(std)][str(K)][str(lamb)][option] = res_dict
        save(res, res_file)


def main():
    list_of_option = ['greedy', 'distort_greedy', 'kl_triage', 'diff_submod']
    list_of_file_name = ['sigmoid_n500d5', 'Gauss_n500d5']

    # specify std, lamb and file_name as specified in ReadMe.txt
    std, lamb, file_name = parse_command_line_input(list_of_option, list_of_file_name)

    list_of_std = [std]
    list_of_lamb = [lamb]
    list_of_K = [0.1, 0.2, 0.3, .4, 0.5, .6, .7, .8, .9]

    if 'Gauss' in file_name:
        DG_T = 10
    if 'sigmoid' in file_name:
        DG_T = 20

    data_file = '../Synthetic_data/data_dict_' + file_name
    res_file = '../Synthetic_data/res_' + file_name
    obj = eval_triage(data_file, real_wt_std=True)

    for option in list_of_option:
        if option == 'diff_submod':
            param = {'std': list_of_std, 'K': list_of_K, 'lamb': list_of_lamb, 'DG_T': DG_T}
        else:
            param = {'std': list_of_std, 'K': [0.99], 'lamb': list_of_lamb, 'DG_T': DG_T}
        obj.eval_loop(param, res_file, option)


if __name__ == "__main__":
    main()
