# Resets the postgresql database to conduct a new optuna study

import optuna
import argparse
import yaml
import psycopg2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Output file config')
    parser.add_argument('--augconfig', type=str, default=None, help='Optuna worker seed')
    parser.add_argument('--datasetconfig', type=str, default=None, help='Optuna worker seed')
    parser.add_argument('--gradMode', type=str, default='SGD', help='VecGAN grad mode')
    args = parser.parse_args()

    with open('./output/HOSTS/host.txt') as f:
        lines = f.readlines()

    fileConfig = args.augconfig
    with open(fileConfig, "r") as read_file:
        aug_config = yaml.load(read_file, Loader=yaml.FullLoader)

    dataset = args.datasetconfig
    with open(dataset, "r") as read_file:
        data_config = yaml.load(read_file, Loader=yaml.FullLoader)

    working_host = lines[0].split(' ')[0]

    aug_name = aug_config['augconfig']['augname']
    data_name = data_config['dataconfig']['dataname']
    grad_mode = args.gradMode

    if aug_name != 'VECGAN':
        study_name = 'Study' + aug_name + data_name
    else:
        grad_mode = aug_config['augconfig']['gradmode']
        study_name = 'Study' + aug_name + data_name + grad_mode

    print("Resetting study...")
    optuna.delete_study(study_name=study_name, storage="postgresql://postgres@{}".format(working_host))
    print("Study reset!")

    # Reset worker connections to avoid errors, uncomment

    # con = psycopg2.connect(database='postgres', user="postgres", password="", host=workinghost, port="5432")
    # cur = con.cursor()
    # try:
    #     cur.execute('''select pg_terminate_backend(pid) from pg_stat_activity where datname='{}';'''.format('postgres'))
    #     con.commit()
    #     con.close()
    # except psycopg2.OperationalError:
    #     placeholder = 0
    #
    # print("Worker connections reset!")

