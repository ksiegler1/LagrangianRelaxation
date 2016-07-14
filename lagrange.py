import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def create_demand_df(demand_file):
    demand = []
    with open(demand_file) as f:
        for line in f:
            s = line.split("\t")
            value = int(s[0])
            demand.append(value)

    demand_df = pd.DataFrame(demand, columns=["demand"])
    return demand_df


def create_distances_matrix(node_file, num_nodes):
    distances = []
    with open(node_file) as f:
        for line in f:
            split = line.split("\t")
            dist = int(split[0])
            distances.append(dist)
    distances_matrix = np.matrix(distances).reshape((num_nodes, num_nodes))
    return distances_matrix


def create_boolean_matrix(distance_mat, d_c):
    bool = distance_mat <= d_c
    binary_matrix = bool.astype(int)
    return binary_matrix


def calc_subproblem_1(demand_df):
    demand_df['difference'] = demand_df['demand'] - demand_df['lambda_0']
    indicator = lambda x: 1 if x > 0 else 0
    demand_df['Z'] = demand_df['difference'].apply(indicator)
    demand_df['sub1output'] = demand_df['difference'] * demand_df['Z']
    return demand_df


def calc_subproblem_2(demand_df, adj_matrix, p):
    adj_df = pd.DataFrame(adj_matrix)
    ai_lambda = adj_df.dot(demand_df['lambda_0'])
    demand_df['ai_lambda'] = ai_lambda
    demand_df['rank'] = demand_df['ai_lambda'].rank(ascending=False)
    min_rank = list(demand_df['rank'].nsmallest(p).index)
    demand_df['x_ij'] = 0
    demand_df['x_ij'].ix[min_rank] = 1
    demand_df['sub2output'] = demand_df['ai_lambda'] * demand_df['x_ij']
    return demand_df


def calc_relaxed_cons(adj_mat, dem_df):
    '''
    :param adj_mat:
    :param dem_df: demand dataframe from subproblem 2 with output from subproblem 1
    :return:
    '''
    adj_df = pd.DataFrame(adj_mat)
    dem_df['aij_xj'] = adj_df.dot(dem_df['x_ij'])
    dem_df['const_output'] = dem_df['aij_xj'] - dem_df['Z']
    return dem_df


def calc_upper_lower_bound(dem_df, adj_mat):
    ub = dem_df['sub1output'].sum() + dem_df['sub2output'].sum()
    # min_rank = list(dem_df['rank'].nsmallest(p).index)
    adj_df = pd.DataFrame(adj_mat)
    the_chosen_ones = list(dem_df[dem_df['x_ij'] == 1].index)
    adj_sums = adj_df.ix[the_chosen_ones].sum(axis=0)
    dem_df['adj_indicator'] = adj_sums
    func2 = lambda x: 1 if x > 0 else 0
    dem_df['adj_indicator'] = dem_df['adj_indicator'].apply(func2)
    lub = dem_df['adj_indicator'].dot(dem_df['demand'])
    return dem_df, ub, lub


def update_lambdas(dem_df, alpha, curr_ub, blb):
    t_n = (alpha * (curr_ub - blb)) / sum(dem_df['const_output']**2)
    compute_lambda = dem_df['lambda_0'] - t_n * dem_df['const_output']
    func = lambda x: 0 if x < 0 else x
    updated_lambdas = compute_lambda.apply(func)
    return updated_lambdas, t_n


def main(P, num_nodes, D_c, demand_file, distance_file):

    bub = 1500000000000
    blb = -100000000000
    alpha = 2
    colnames = ['Iteration', 'BestLB', 'CurrUB', 'BestUB', 'alpha', 't_n']
    list_of_df = list()

    toy_demand = create_demand_df(demand_file)
    toy_distances = create_distances_matrix(distance_file, num_nodes)
    toy_binary = create_boolean_matrix(toy_distances, D_c)
    avg_toy_demand = toy_demand['demand'].mean()
    toy_demand['lambda_0'] = avg_toy_demand + 0.5 * (toy_demand['demand'] - avg_toy_demand)

    check_4_times = 0
    count = 0
    curr_gap = (bub - blb) / bub
    count_gaps = 0
    beg_time = time.time()
    while True:
    # for i in range(500):

        tmp = calc_subproblem_1(toy_demand)
        tmp = calc_subproblem_2(tmp, toy_binary, P)
        tmp = calc_relaxed_cons(toy_binary, tmp)
        tmp, ub, lb = calc_upper_lower_bound(tmp, toy_binary)

        # update the best upper bound and best lower bound
        if ub < bub:
            bub = ub
            check_4_times = 0
        else:
            check_4_times += 1

        if lb > blb:
            blb = lb
        # reduce lambda if best upper bound does not decrease 4 times in a row
        if check_4_times == 4:
            alpha /= 2.0

        new_gap = (bub - blb) / bub
        if new_gap == curr_gap:
            count_gaps += 1
        else:
            count_gaps = 0

        if count_gaps == 500:
            print("optimal gap: {0}".format((bub - blb) / bub))
            break
        curr_gap = new_gap

        tmp['lambda_0'], t_n = update_lambdas(tmp, alpha, curr_ub=ub, blb=blb)

        # creating summary table
        Z_i = ''.join(str(e) for e in list(tmp['Z']))
        X_i = ''.join(str(e) for e in list(tmp['x_ij']))
        pct_covered = sum(tmp['Z']) / float(len(tmp))

        iter = "Iteration {0}".format(count)

        # tmp_df = pd.DataFrame([[iter, Z_i, X_i, blb, ub, bub, alpha, t_n]], columns=colnames)

        tmp_df = pd.DataFrame([[iter, blb, ub, bub, alpha, t_n]], columns=colnames)
        list_of_df.append(tmp_df)
        count += 1
        if count == 5000:
            print "Maxed out iterations"
            break
    end_time = time.time()
    total_time = end_time - beg_time
    summary_df = pd.concat(list_of_df)
    print summary_df.head(10)
    print '======='
    print summary_df.tail(10)
    print "Number of Iterations: {0}".format(count)
    print "Coverage: {0}".format(Z_i)
    print "Percent Covered: {0}".format(pct_covered)
    return pct_covered, p, total_time, curr_gap


if __name__ == '__main__':

    D_c = 400
    num_nodes = 49
    p_vals = []
    pct_vals = []
    time_vals = []
    gap_vals = []
    for p in range(3, 20):
        temp_pct, temp_p, temp_time, temp_gap = main(p, 49, 400, "/Users/ksiegler/Documents/USFClasses/SupplyChain/Project/supply-chain-project/49 NodeDemandData.txt", "/Users/ksiegler/Documents/USFClasses/SupplyChain/Project/supply-chain-project/49Node.txt")
        p_vals.append(temp_p)
        pct_vals.append(temp_pct)
        time_vals.append(temp_time)
        gap_vals.append(temp_gap)
    plt.plot(p_vals, pct_vals)


    # colnames = ['Iteration', 'Z_i', 'X_i', 'BestLB', 'Curr_UB', 'BestUB', 'alpha', 't_n']

