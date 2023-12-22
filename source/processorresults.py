# Calculation recording script
#######################

import statistics
import glob
from prettytable import PrettyTable

def parse_perf_stat_text(file):
    """
    Parses calculation estimates from output of perf stat.
    :param file:
    :return:
    """

    with open(file) as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        temp = line.strip()
        if len(temp) > 1:
            temp = temp.split()
            clean_lines.append(temp)

    ret_instr = 0
    ret_cycles = 0
    for line in clean_lines:
        if 'instructions' in line:
            ret_instr = int(line[0].replace(',',''))
        if 'cycles' in line:
            ret_cycles = int(line[0].replace(',',''))

    return ret_cycles, ret_instr


def parse_perf_mem_text(file):
    """
        Parses calculation estimates from output of perf stat.
        :param file:
        :return:
        """

    with open(file) as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        temp = line.strip()
        if len(temp) > 1:
            temp = temp.split()
            clean_lines.append(temp)

    ret_loads = 0
    ret_stores = 0
    for i, line in enumerate(clean_lines):
        if '\'cpu/mem-loads,ldlat=30/P\'' in line:
            ret_loads = int(clean_lines[i+1][-1])
        if '\'cpu/mem-stores/P\'' in line:
            ret_stores = int(clean_lines[i+1][-1])

    return ret_loads, ret_stores


def print_results():
    """
    Prints the results of the calculation estimates
    :return:
    """

    configs_test = ["control", "smote", "vanillagan", "wgan", "ctgan", "vecgan_sgd", "vecgan_edl", "vecgan_bfa"]
    # configs_test = ["control","vecgan_bfa"]
    print(configs_test)
    cycle_logs = []
    instr_logs = []
    for algo_config in configs_test:

        # Read and average all perf stat outputs
        algo_cycle = []
        algo_instr = []
        for file in glob.glob("./output/powerAnalysis/{}*.txt".format(algo_config)):
            cycles, instrs = parse_perf_stat_text(file)
            algo_cycle.append(cycles/1E9/2) # Average over 5 trials
            algo_instr.append(instrs/1E9/2)

        cycle_logs.append(statistics.mean(algo_cycle))
        instr_logs.append(statistics.mean(algo_instr))

    # Perf mem results

    load_logs = []
    store_logs = []
    for algo_config in configs_test:

        # Read and average all perf stat outputs
        algo_loads = []
        algo_stores = []
        for file in glob.glob("./output/powerAnalysis/{}memresults*.txt".format(algo_config)):
            loads, stores = parse_perf_mem_text(file)
            algo_loads.append(loads/1E9/2)
            algo_stores.append(stores/1E9/2)

        load_logs.append(statistics.mean(algo_loads))
        store_logs.append(statistics.mean(algo_stores))

    x = PrettyTable()

    temp = ["Parameter"]
    temp.extend(configs_test)

    x.field_names = temp

    # Cycle
    row = ['Cycles [E9]']
    row.extend(["{0:0.2f}".format(i) for i in cycle_logs])
    x.add_row(row)

    # Instruction
    row = ['Instr. [E9]']
    row.extend(["{0:0.2f}".format(i) for i in instr_logs])
    x.add_row(row)

    row = ['CPU Loads [E9]']
    row.extend(["{0:0.2f}".format(i) for i in load_logs])
    x.add_row(row)

    row = ['CPU Stores [E9]']
    row.extend(["{0:0.2f}".format(i) for i in store_logs])
    x.add_row(row)

    print("Raw Data")
    print(x)

    # Normalize cycles compared to control

    x_norm = PrettyTable()
    x_norm.field_names = temp
    norm_cycles = [cycle_logs[i] - cycle_logs[0] for i in range(len(cycle_logs))]
    norm_instr = [instr_logs[i] - instr_logs[0] for i in range(len(instr_logs))]
    norm_loads = [load_logs[i] - load_logs[0] for i in range(len(load_logs))]
    norm_stores = [store_logs[i] - store_logs[0] for i in range(len(store_logs))]

    row = ['Cycles [E9]']
    row.extend(["{0:0.2f}".format(i) for i in norm_cycles])
    x_norm.add_row(row)

    row = ['Instr. [E9]']
    row.extend(["{0:0.2f}".format(i) for i in norm_instr])
    x_norm.add_row(row)

    row = ['CPU Loads [E9]']
    row.extend(["{0:0.2f}".format(i) for i in norm_loads])
    x_norm.add_row(row)

    row = ['CPU Stores [E9]']
    row.extend(["{0:0.2f}".format(i) for i in norm_stores])
    x_norm.add_row(row)

    print("Normalized Data")
    print(x_norm)


if __name__ == "__main__":

    # Read test results from recording calculations
    print_results()
