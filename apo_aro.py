import numpy as np
from scipy.special import gamma

class APO_ARO:
    def __init__(self, population_size, dim, max_iter, lb, ub, fitness_func, k_share=1):
        self.N = population_size
        self.dim = dim
        self.T = max_iter
        self.lb = lb
        self.ub = ub
        self.fobj = fitness_func
        self.k = k_share

        self.current_iter = 0
        self.fitness_log = []

        self.PopPos_APO = np.random.randint(1, self.ub + 1, size=(self.N, self.dim))
        self.PopPos_ARO = np.random.randint(1, self.ub + 1, size=(self.N, self.dim))

        self.PopFit_APO = np.zeros(self.N)
        self.PopIL_APO = np.zeros(self.N)
        self.PopM_APO = np.zeros(self.N)
        self.PopCA_APO = np.zeros(self.N)
        self.PopFit_ARO = np.zeros(self.N)
        self.PopIL_ARO = np.zeros(self.N)
        self.PopM_ARO = np.zeros(self.N)
        self.PopCA_ARO = np.zeros(self.N)

        self._initialize_fitness()

        self.best_fitness = float('inf')
        self.best_solution = None
        self.best_il = 0.0
        self.best_m = 0.0
        self.best_ca = 0.0
        self._update_best_solution()

    def _initialize_fitness(self):
        for i in range(self.N):
            fitness_result = self.fobj(self.PopPos_APO[i, :])
            self.PopFit_APO[i], self.PopIL_APO[i], self.PopM_APO[i], self.PopCA_APO[i] = fitness_result

            fitness_result = self.fobj(self.PopPos_ARO[i, :])
            self.PopFit_ARO[i], self.PopIL_ARO[i], self.PopM_ARO[i], self.PopCA_ARO[i] = fitness_result

    def _update_best_solution(self):
        best_idx_APO = np.argmin(self.PopFit_APO)
        best_idx_ARO = np.argmin(self.PopFit_ARO)
        if self.PopFit_APO[best_idx_APO] < self.PopFit_ARO[best_idx_ARO]:
            best_f = self.PopFit_APO[best_idx_APO]
            best_x = self.PopPos_APO[best_idx_APO].copy()
            best_i = self.PopIL_APO[best_idx_APO]
            best_m = self.PopM_APO[best_idx_APO]
            best_c = self.PopCA_APO[best_idx_APO]
        else:
            best_f = self.PopFit_ARO[best_idx_ARO]
            best_x = self.PopPos_ARO[best_idx_ARO].copy()
            best_i = self.PopIL_ARO[best_idx_ARO]
            best_m = self.PopM_ARO[best_idx_ARO]
            best_c = self.PopCA_ARO[best_idx_ARO]

        if best_f < self.best_fitness and best_f != float('inf'):
            self.best_fitness = best_f
            self.best_solution = best_x.copy()
            self.best_il = best_i
            self.best_m = best_m
            self.best_ca = best_c

    def space_bound(self, x):
        return np.clip(x, self.lb, self.ub)

    def levy(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def aro(self, PopPos, num_iter):
        PopPos2 = PopPos.copy()
        pop_fit = np.zeros(self.N)
        pop_il = np.zeros(self.N)
        pop_m = np.zeros(self.N)
        pop_ca = np.zeros(self.N)

        for i in range(self.N):
            fitness_result = self.fobj(PopPos2[i, :])
            pop_fit[i], pop_il[i], pop_m[i], pop_ca[i] = fitness_result

        best_f = float('inf')
        best_x = None
        best_idx = np.argmin(pop_fit)
        if pop_fit[best_idx] != float('inf'):
            best_f = pop_fit[best_idx]
            best_x = PopPos2[best_idx, :].copy()

        local_fitness_log = []

        for it in range(num_iter):
            direct1 = np.zeros((self.N, self.dim))
            direct2 = np.zeros((self.N, self.dim))
            theta = 2 * (1 - (it + 1) / num_iter)
            for i in range(self.N):
                L = (np.e - np.exp((((it + 1) - 1) / num_iter) ** 2)) * np.sin(2 * np.pi * np.random.rand())
                rd = np.floor(np.random.rand() * self.dim)
                rand_dim = np.random.permutation(self.dim)
                direct1[i, rand_dim[:int(rd)]] = 1
                c = direct1[i, :]
                R = L * c
                A = 2 * np.log(1 / np.random.rand()) * theta
                if A > 1:
                    K = np.r_[0:i, i + 1:self.N]
                    RandInd = K[np.random.randint(0, self.N - 1)]
                    newPopPos = PopPos2[RandInd, :] + R * (PopPos2[i, :] - PopPos2[RandInd, :]) + 0.5 * (0.05 + np.random.rand()) * np.random.randn()
                else:
                    ttt = int(np.floor(np.random.rand() * self.dim))
                    direct2[i, ttt] = 1
                    gr = direct2[i, :]
                    H = ((num_iter - (it + 1) + 1) / num_iter) * np.random.randn()
                    b = PopPos2[i, :] + H * gr * PopPos2[i, :]
                    newPopPos = PopPos2[i, :] + R * (np.random.rand() * b - PopPos2[i, :])

                newPopPos = np.round(newPopPos)
                newPopPos = self.space_bound(newPopPos)
                fitness_result = self.fobj(newPopPos)
                newPopFit, newPopIL, newPopM, newPopCA = fitness_result

                if newPopFit < pop_fit[i]:
                    pop_fit[i] = newPopFit
                    PopPos2[i, :] = newPopPos
                    pop_il[i] = newPopIL
                    pop_m[i] = newPopM
                    pop_ca[i] = newPopCA

                if pop_fit[i] < best_f and pop_fit[i] != float('inf'):
                    best_f = pop_fit[i]
                    best_x = PopPos2[i, :].copy()

            local_fitness_log.append({'Iteration': it + 1, 'BestFitness': best_f})

        return local_fitness_log, PopPos2

    def apo(self, PopPos, num_iter):
        PopPos1 = PopPos.copy()
        PopFit = np.zeros(self.N)
        PopIL = np.zeros(self.N)
        PopM = np.zeros(self.N)
        PopCA = np.zeros(self.N)

        for i in range(self.N):
            fitness_result = self.fobj(PopPos1[i, :])
            PopFit[i], PopIL[i], PopM[i], PopCA[i] = fitness_result

        best_f = float('inf')
        best_x = None
        best_idx = np.argmin(PopFit)
        if PopFit[best_idx] != float('inf'):
            best_f = PopFit[best_idx]
            best_x = PopPos1[best_idx, :].copy()

        local_fitness_log = []

        for It in range(num_iter):
            rand = np.random.rand()
            for i in range(self.N):
                theta1 = (1 - It / num_iter)
                B = 2 * np.log(1 / rand) * theta1

                if B > 0.5:
                    step1 = None
                    for _ in range(num_iter):
                        K = [j for j in range(self.N) if j != i]
                        if not K:
                            break
                        RandInd = np.random.choice(K)
                        step1 = PopPos[i] - PopPos[RandInd]
                        if np.linalg.norm(step1) != 0:
                            break
                    if step1 is None or np.linalg.norm(step1) == 0:
                        newPopPos = PopPos1[i, :].copy()
                    else:
                        R = 0.5 * (0.05 + rand) * np.random.normal(0, 1)
                        Y = PopPos1[i, :] + 0.01 * self.levy() * step1 + R
                        step2 = (rand - 0.5) * np.pi
                        S = np.tan(step2)
                        Z = Y * S

                        Y = np.round(Y)
                        Z = np.round(Z)
                        Y = self.space_bound(Y)
                        Z = self.space_bound(Z)
                        NewPop = np.array([Y, Z])

                        fitness_Y = self.fobj(Y)
                        fitness_Z = self.fobj(Z)
                        NewPopFit = np.array([fitness_Y[0], fitness_Z[0]])
                        sorted_indexes = np.argsort(NewPopFit)
                        newPopPos = NewPop[sorted_indexes[0], :]

                else:
                    F = 0.5
                    K = [j for j in range(self.N) if j != i]
                    step1 = None
                    for _ in range(num_iter):
                        available_indices = [j for j in range(self.N) if j != i]
                        if len(available_indices) < 3:
                            break
                        RandInd = np.random.choice(available_indices, 3, replace=False)
                        step1 = PopPos[RandInd[1]] - PopPos[RandInd[2]]
                        if np.linalg.norm(step1) != 0:
                            break
                    if step1 is None or np.linalg.norm(step1) == 0:
                        newPopPos = PopPos1[i, :].copy()
                    else:
                        if rand < 0.5:
                            W = PopPos1[RandInd[0], :] + F * step1
                        else:
                            W = PopPos1[RandInd[0], :] + F * 0.01 * self.levy() * step1
                        f = 0.1 * (rand - 1) * ((num_iter - It) / num_iter)
                        Y = (1 + f) * W

                        step2 = None
                        for _ in range(num_iter):
                            rand_leader_index1 = np.random.randint(0, self.N)
                            rand_leader_index2 = np.random.randint(0, self.N)
                            X_rand1 = PopPos1[rand_leader_index1, :]
                            X_rand2 = PopPos1[rand_leader_index2, :]
                            step2 = X_rand1 - X_rand2
                            if np.linalg.norm(step2) != 0 and not np.array_equal(X_rand1, X_rand2):
                                break
                        if step2 is None or np.linalg.norm(step2) == 0:
                            newPopPos = PopPos1[i, :].copy()
                        else:
                            Epsilon = np.random.uniform(0, 1)
                            if rand < 0.5:
                                Z = PopPos1[i, :] + Epsilon * step2
                            else:
                                Z = PopPos1[i, :] + F * 0.01 * self.levy() * step2

                            W = np.round(W)
                            Y = np.round(Y)
                            Z = np.round(Z)
                            W = self.space_bound(W)
                            Y = self.space_bound(Y)
                            Z = self.space_bound(Z)
                            NewPop = np.array([W, Y, Z])

                            fitness_W = self.fobj(W)
                            fitness_Y = self.fobj(Y)
                            fitness_Z = self.fobj(Z)
                            NewPopFit = np.array([fitness_W[0], fitness_Y[0], fitness_Z[0]])
                            sorted_indexes = np.argsort(NewPopFit)
                            newPopPos = NewPop[sorted_indexes[0], :]

                newPopPos = np.round(newPopPos)
                newPopPos = self.space_bound(newPopPos)
                fitness_result = self.fobj(newPopPos)
                newPopFit, newPopIL, newPopM, newPopCA = fitness_result

                if newPopFit < PopFit[i]:
                    PopFit[i] = newPopFit
                    PopPos1[i, :] = newPopPos
                    PopIL[i] = newPopIL
                    PopM[i] = newPopM
                    PopCA[i] = newPopCA

            for i in range(self.N):
                if PopFit[i] < best_f and PopFit[i] != float('inf'):
                    best_f = PopFit[i]
                    best_x = PopPos1[i, :].copy()

            local_fitness_log.append({'Iteration': It + 1, 'BestFitness': best_f})

        return local_fitness_log, PopPos1

    def iterate(self):
        if self.current_iter >= self.T:
            return

        num_iter = min(self.k, self.T - self.current_iter)
        fitness_log_APO, self.PopPos_APO = self.apo(self.PopPos_APO, num_iter)
        fitness_log_ARO, self.PopPos_ARO = self.aro(self.PopPos_ARO, num_iter)

        for i in range(self.N):
            fitness_result = self.fobj(self.PopPos_APO[i, :])
            self.PopFit_APO[i], self.PopIL_APO[i], self.PopM_APO[i], self.PopCA_APO[i] = fitness_result

            fitness_result = self.fobj(self.PopPos_ARO[i, :])
            self.PopFit_ARO[i], self.PopIL_ARO[i], self.PopM_ARO[i], self.PopCA_ARO[i] = fitness_result

        self._update_best_solution()

        best_idx_APO = np.argmin(self.PopFit_APO)
        best_idx_ARO = np.argmin(self.PopFit_ARO)
        best_APO = self.PopPos_APO[best_idx_APO].copy()
        best_ARO = self.PopPos_ARO[best_idx_ARO].copy()

        self.PopPos_APO[np.random.randint(0, self.N)] = best_ARO
        self.PopPos_ARO[np.random.randint(0, self.N)] = best_APO

        for entry in fitness_log_APO:
            entry['Iteration'] += self.current_iter
        for entry in fitness_log_ARO:
            entry['Iteration'] += self.current_iter
        self.fitness_log.extend(fitness_log_APO)
        self.fitness_log.extend(fitness_log_ARO)

        self.current_iter += num_iter

    def run(self):
        while self.current_iter < self.T:
            self.iterate()

        if self.best_solution is None:
            print("Không tìm thấy giải pháp tối ưu thỏa mãn k-anonymity!")
            PopPos = np.vstack((self.PopPos_APO, self.PopPos_ARO))
            generalization_sums = np.sum(PopPos, axis=1)
            best_idx = np.argmax(generalization_sums)
            self.best_solution = PopPos[best_idx, :].copy()
            self.best_fitness, self.best_il, self.best_m, self.best_ca = self.fobj(self.best_solution)
            print(f"Chọn cá thể có mức tổng quát hóa cao nhất: {self.best_solution}, Fitness: {self.best_fitness:.4f}")

        return self.best_solution, self.best_fitness