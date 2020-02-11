from triage_class import *
import numpy as np
import numpy.linalg as LA
import time


class triage_human_machine:
    def __init__(self, data_dict, real=None):
        self.X = data_dict['X']
        self.Y = data_dict['Y']

        if real:
            self.c = data_dict['c']

        else:
            self.c = np.square(data_dict['human_pred'] - data_dict['Y'])

        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.epsilon = float(1)
        self.BIG_VALUE = 100000
        self.real = real

    def get_c(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def get_minus(self, subset, elm):
        return np.array([i for i in subset if i != elm])

    def get_added(self, subset, elm):
        return np.concatenate((subset, np.array([int(elm)])), axis=0)

    def check_delete(self, g_m, subset, approx):
        if subset.size == 0:
            return False, subset

        g_m_subset = g_m.eval(subset)
        g_m_subset_vector = g_m.eval_vector(subset)

        if np.max(g_m_subset_vector) >= g_m_subset * approx:
            item_to_del = subset[np.argmax(g_m_subset_vector)]
            subset_left = self.get_minus(subset, item_to_del)
            return True, subset_left

        return False, subset

    def check_exchange_greedy(self, g_m, subset, ground_set, approx, K):

        g_m_subset = g_m.eval(subset)
        g_m_exchange, subset_with_null, subset_c_gr = g_m.eval_exch_or_add(subset, ground_set, K)

        if np.max(g_m_exchange) > g_m_subset * approx:
            r, c = np.unravel_index(np.argmax(g_m_exchange, axis=None), g_m_exchange.shape)
            # print 'index of max element ',r,c
            e = subset_with_null[r]
            d = subset_c_gr[c]
            # print e,' is exchanged with ',d
            if e == -1:
                subset_with_null[r] = d
                return True, subset_with_null
            else:
                ind_e = np.where(subset == e)[0]
                subset[ind_e] = d
                return True, subset
        return False, subset

    def approx_local_search(self, g_m, K, ground_set):
        # max_A (g-m)(A) given |A|<=k  	implementing local search by J.Lee 2009 STOC
        approx = 1 + self.epsilon / float(self.n ** 4)
        curr_subset = np.array([g_m.find_max_elm(ground_set)])
        while True:
            # start = time.time()
            # print ' ---   Delete ----- '
            flag_delete, curr_subset = self.check_delete(g_m, curr_subset, approx)
            if flag_delete:
                # print 'deleted'
                pass
            else:
                # print ' --- Exchange ---- '
                flag_exchange, curr_subset = self.check_exchange_greedy(g_m, curr_subset, ground_set, approx, K)
                # time.sleep(100000)
                if flag_exchange:
                    pass  # print 'exchanged'
                else:
                    break
        # print '---------------Subset----------------------------------'
        # finish = time.time()
        # print '-----------------------------'
        # print 'Time -- > ', finish-start
        # print 'Subset', curr_subset
        return curr_subset

    def constr_submod_max_greedy(self, g_m, K):
        # print 'constr submod max greedy'
        curr_set = np.array([]).astype(int)
        # print 'K',K
        # print 'start val---------->', g_m.eval(curr_set)
        for itr in range(K):

            vector, subset_left = g_m.get_inc_arr(curr_set)
            if np.max(vector) <= 0:
                break
            idx_to_add = subset_left[np.argmax(vector)]
            curr_set = self.get_added(curr_set, idx_to_add)
        # print 'Iteration ',itr,'_______',g_m.eval(curr_set)
        # print 'final------------> ',g_m.eval(curr_set)
        return curr_set

    def constr_submod_max(self, g_m, K):

        ground_set = self.V
        # print '----- local search 1 '
        start = time.time()
        subset_1 = self.approx_local_search(g_m, K, ground_set)
        ground_set = self.get_c(subset_1)
        # print '----- local search 2 '
        subset_2 = self.approx_local_search(g_m, K, ground_set)
        finish = time.time()
        print 'Time -- > ', (finish - start)
        if g_m.eval(subset_1) > g_m.eval(subset_2):
            return subset_1
        else:
            return subset_2

    def sel_subset_diff_submod_greedy(self):
        # solve difference of submodular functions
        subset_old = np.array([])
        g_f = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        val_old = g_f.eval(subset_old)
        # print 'VAl---------------> ',val_old
        itr = 0
        while True:
            # print '-----Diff submod greedy----------Iter ', itr, '  ---------------------------------------'
            # print 'modular upper bound '
            # return
            f = F({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
            m_f = f.modular_upper_bound(subset_old)
            g_m = SubMod({'X': self.X, 'lamb': self.lamb, 'm': m_f})
            subset = self.constr_submod_max_greedy(g_m, self.K)

            # check whether g-f really improve
            val_curr = g_f.eval(subset)
            # print 'VAl---------------> ',val_curr
            if val_curr <= val_old:
                return subset_old
            if set(subset) == set(subset_old):
                return subset
            else:
                subset_old = subset
                val_old = val_curr
            itr += 1

    def sel_subset_diff_submod(self):
        # solve difference of submodular functions
        subset_old = np.array([])

        itr = 0
        while True:
            # print '-------------------------------Iter ', itr, '  ---------------------------------------'
            # print 'modular upper bound '
            f = F({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
            m_f = f.modular_upper_bound(subset_old)
            # m_f=self.modular_upper_bound(subset_old)
            # print 'constr submodular max '
            g_m = SubMod({'X': self.X, 'lamb': self.lamb, 'm': m_f})
            subset = self.constr_submod_max(g_m, self.K)
            # print '--------OLD-------------------'
            print 'subset length', subset.shape
            # print '---------New-------------------'
            # print subset
            if set(subset) == set(subset_old):
                return subset
            else:
                subset_old = subset
            itr += 1

    def set_param(self, lamb, K):
        self.lamb = lamb
        self.K = K

    def get_optimal_pred(self, subset):
        subset_c = self.get_c(subset)
        X_sub = self.X[subset_c].T
        Y_sub = self.Y[subset_c]
        subset_c_l = self.n - subset.shape[0]
        return LA.inv(self.lamb * subset_c_l * np.eye(self.dim) + X_sub.dot(X_sub.T)).dot(X_sub.dot(Y_sub))

    def plot_subset(self, w, subset, K):
        plt_obj = {}

        x = self.X[subset, 0].flatten()
        y = self.Y[subset]
        # plt.scatter(x,y,c='red',label='human')
        plt_obj['human'] = {'x': x, 'y': y}

        c_subset = self.get_c(subset)
        x = self.X[c_subset, 0].flatten()
        y = self.Y[c_subset]
        # plt.scatter(x,y,c='blue',label='machine')
        plt_obj['machine'] = {'x': x, 'y': y}

        x = self.X[:, 0].flatten()
        y = self.X.dot(w).flatten()
        plt_obj['prediction'] = {'x': x, 'y': y, 'w': w}
        # plt.scatter(x,y,c='black',label='prediction')
        # plt.ylim([-1,1])
        # plt.legend()
        # plt.grid(True)
        # plt.title('Fraction of sample to human'+str(K) )
        # plt.show()
        return plt_obj

    def distort_greedy(self, g, K, gamma):

        c_mod = modular_distort_greedy({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        subset = np.array([]).astype(int)
        g.reset()
        for itr in range(K):
            frac = (1 - gamma / float(K)) ** (K - itr - 1)
            # print frac
            subset_c = self.get_c(subset)
            # print subset_c.shape
            c_mod_inc = c_mod.get_inc_arr(subset)
            c_mod_inc = np.reshape(c_mod_inc, c_mod_inc.shape[0])
            # print 'c',c_mod_inc.shape
            g_inc_arr, subset_c_ret = g.get_inc_arr(subset)
            g_pos_inc = g_inc_arr + c_mod_inc
            inc_vec = frac * g_pos_inc - c_mod_inc
            if np.max(inc_vec) <= 0:
                return subset
            sel_ind = np.argmax(inc_vec)
            elm = subset_c[sel_ind]
            subset = self.get_added(subset, elm)
            g.update_data_str(elm)
        return subset

    def gamma_sweep_distort_greedy(self, T=100):
        g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        delta = 0.05
        subset = {}
        G_subset = []
        gamma = 1.0
        # print T
        start = time.time()
        for r in range(T + 1):
            subset_sel = self.distort_greedy(g, self.K, gamma)
            subset[str(r)] = subset_sel
            G_subset.append(g.eval(subset_sel))
            gamma = gamma * (1 - delta)
        empty_set = np.array([]).astype(int)
        subset[str(T + 1)] = empty_set
        G_subset.append(g.eval(empty_set))
        max_set_ind = np.argmax(np.array(G_subset))

        return subset[str(max_set_ind)]

    def max_submod_greedy(self):

        curr_set = np.array([]).astype(int)
        g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        print 'Need to select ', self.K, ' items'
        start = time.time()
        for itr in range(self.K):
            vector, subset_left = g.get_inc_arr(curr_set)
            # print 'inc',np.max(vector)
            if np.max(vector) < 0:
                print 'dataset size', curr_set.shape[0]
                return curr_set
            idx_to_add = subset_left[np.argmax(vector)]
            curr_set = self.get_added(curr_set, idx_to_add)
            g.update_data_str(idx_to_add)

            if itr % 50 == 0:
                time_r = time.time() - start
                print 'itr ', itr
                print time_r, ' seconds'
        print 'dataset size', curr_set.shape[0]
        return curr_set

    def kl_triage_subset(self):
        kl_obj = kl_triage({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb})
        return kl_obj.get_subset(self.K)

    def algorithmic_triage(self, param, optim):
        self.set_param(param['lamb'], int(param['K'] * self.n))
        if optim == 'diff_submod':
            subset = self.sel_subset_diff_submod()
        if optim == 'greedy':
            subset = self.max_submod_greedy()
        if optim == 'diff_submod_greedy':
            subset = self.sel_subset_diff_submod_greedy()
        if optim == 'distort_greedy':
            subset = self.gamma_sweep_distort_greedy(param['DG_T'])
        if optim == 'kl_triage':
            subset = self.kl_triage_subset()
        w_m = self.get_optimal_pred(subset)
        plt_obj = {'w': w_m, 'subset': subset}
        return plt_obj
