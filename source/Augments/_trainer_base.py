"""Trainer Base

Provides base class for other augmentation techniques.

"""
import sys
import yaml
from os import path
import pickle
sys.path.append('./source/Datasets/')
# noinspection PyUnresolvedReferences
from _bitcoinlate2012 import BitcoinDataset
# noinspection PyUnresolvedReferences
from _creditfraud_dataset import CreditFraudDataset
# noinspection PyUnresolvedReferences
from _pima import PimaDataset
# noinspection PyUnresolvedReferences
from _blobs import BlobsDataset
# noinspection PyUnresolvedReferences
from _circles import CirclesDataset
# noinspection PyUnresolvedReferences
from _ledgerattack import LedgerAttackDataset
# noinspection PyUnresolvedReferences
from _bank_note import BanknoteDataset
# noinspection PyUnresolvedReferences
from _haberman import HabermanDataset
# noinspection PyUnresolvedReferences
from _electric import ElectricDataset
# noinspection PyUnresolvedReferences
from _occupancy import OccupancyDataset
# noinspection PyUnresolvedReferences
from _dota2 import Dota2Dataset
# noinspection PyUnresolvedReferences
from _bank_marketing import BankMarketingDataset
# noinspection PyUnresolvedReferences
from _adult import AdultDataset
# noinspection PyUnresolvedReferences
from _news import NewsDataset
# noinspection PyUnresolvedReferences
from _yeast import YeastDataset
# noinspection PyUnresolvedReferences
from _glass import GlassDataset
# noinspection PyUnresolvedReferences
from _digits import DigitsDataset
# noinspection PyUnresolvedReferences
from _bitcoinlate2012sub import BitcoinLate2012SubDataset
# noinspection PyUnresolvedReferences
from _ethfraud import EthFraudDataset

class TrainerBase:
    """
    Base class for other augmentation types.
    """

    def __init__(self, seed=0):
        self.seed = seed
        self.datasets = {'BITCOINLATE2012': BitcoinDataset,
                         'BITCOINENTIRE': BitcoinDataset,
                         'CREDITFRAUD': CreditFraudDataset,
                         'PIMA': PimaDataset,
                         'LEDGERATTACKNAIVE': LedgerAttackDataset,
                         'LEDGERATTACKINTELLIGENT': LedgerAttackDataset,
                         'HABERMAN': HabermanDataset,
                         'BANKNOTE': BanknoteDataset,
                         'ELECTRIC': ElectricDataset,
                         'OCCUPANCY': OccupancyDataset,
                         'DOTA2':Dota2Dataset,
                         'BANKMARKETING': BankMarketingDataset,
                         'ADULT': AdultDataset,
                         'NEWS': NewsDataset,
                         'YEAST': YeastDataset,
                         'GLASS': GlassDataset,
                         'DIGITS': DigitsDataset,
                         'BITCOINLATE2012SUB': BitcoinLate2012SubDataset,
                         'ETHFRAUD': EthFraudDataset}

        self.dataconfigs = {'BITCOINLATE2012': './config/dataconfig/bitcoinlate2012config.yml',
                            'BITCOINENTIRE': './config/dataconfig/bitcoinentireconfig.yml',
                            'CREDITFRAUD': './config/dataconfig/creditfraudconfig.yml',
                            'PIMA': './config/dataconfig/pimaconfig.yml',
                            'LEDGERATTACKNAIVE': './config/dataconfig/ledgerattacknaiveconfig.yml',
                            'LEDGERATTACKINTELLIGENT': './config/dataconfig/ledgerattackintelligentconfig.yml',
                            'HABERMAN': './config/dataconfig/habermanconfig.yml',
                            'BANKNOTE': './config/dataconfig/banknoteconfig.yml',
                            'ELECTRIC': './config/dataconfig/electricconfig.yml',
                            'OCCUPANCY': './config/dataconfig/occupancyconfig.yml',
                            'DOTA2': './config/dataconfig/dota2config.yml',
                            'BANKMARKETING': './config/dataconfig/bankmarketingconfig.yml',
                            'ADULT': './config/dataconfig/adultconfig.yml',
                            'NEWS': './config/dataconfig/newsconfig.yml',
                            'YEAST': './config/dataconfig/yeastconfig.yml',
                            'GLASS': './config/dataconfig/glassconfig.yml',
                            'DIGITS': './config/dataconfig/digitsconfig.yml',
                            'BITCOINLATE2012SUB': './config/dataconfig/bitcoinlate2012subconfig.yml',
                            'ETHFRAUD': './config/dataconfig/ethfraudconfig.yml'}

        return

    def get_dataset(self, data_name, trainPerc=None, hasIDN=None,idnPerc=None):
        """
        Main handling for loading data for training/testing.
        :param data_name:
        :type data_name:
        :return:
        :rtype:
        """

        if idnPerc is None:
            idnPerc = 0

        file_config = self.dataconfigs[data_name]

        with open(file_config, "r") as read_file:
            config = yaml.load(read_file, Loader=yaml.FullLoader)

        self.working_data_config = config

        if trainPerc is not None:
            config['dataconfig']['trainperc'] = trainPerc

        # if 'BITCOIN' in data_name:
        #
        #     base_file = config['fileconfig']['pickleout']
        #     reduce_amount = config['dataconfig']['totaldownsample']
        #
        #     train_str = '_reduce_{}_'.format(int(reduce_amount))
        #
        #     base_file = base_file[:-2] + train_str + 'seed_' + str(self.seed) + base_file[-2:]
        #
        #     if path.exists(base_file):
        #         bitcoin_data = pickle.load(open(base_file, 'rb'))
        #     else:
        #         bitcoin_data = BitcoinDataset(data_name)
        #         pickle.dump(bitcoin_data, open(base_file, 'wb'))
        #
        #     return bitcoin_data

        base_file = config['fileconfig']['pickleout']

        train_test = config['dataconfig']['trainperc']

        train_str = '_split_{}_'.format(int(100 * train_test))

        base_file = base_file[:-2] + train_str +'seed_' + str(self.seed) + base_file[-2:]

        if data_name == 'CREDITFRAUD':

            if path.exists(base_file):
                credit_fraud = pickle.load(open(base_file, 'rb'))
            else:
                credit_fraud = CreditFraudDataset()
                pickle.dump(credit_fraud, open(base_file, 'wb'))

            return credit_fraud

        if data_name == 'PIMA':

            if hasIDN is not None:
                base_file = base_file[:-2] +'_idn_' + str(idnPerc) + base_file[-2:]

            if path.exists(base_file):
                pima = pickle.load(open(base_file, 'rb'))
            else:
                pima = PimaDataset(trainPerc,idnPerc)
                pickle.dump(pima, open(base_file, 'wb'))

            return pima

        if data_name == 'CIRCLES':
            if path.exists(base_file):
                circles = pickle.load(open(base_file, 'rb'))
            else:
                circles = CirclesDataset()
                pickle.dump(circles, open(base_file, 'wb'))

            return circles

        if data_name == 'BLOBS':
            if path.exists(base_file):
                blobs = pickle.load(open(base_file, 'rb'))
            else:
                blobs = BlobsDataset()
                pickle.dump(blobs, open(base_file, 'wb'))

            return blobs

        if 'LEDGERATTACK' in data_name:
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = LedgerAttackDataset(config['dataconfig']['attackType'])
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'HABERMAN':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = HabermanDataset()
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'YEAST':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = YeastDataset()
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'BANKNOTE':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = BanknoteDataset()
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'ELECTRIC':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = ElectricDataset()
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'OCCUPANCY':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = OccupancyDataset()
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'DOTA2':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = Dota2Dataset()
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'BANKMARKETING':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = BankMarketingDataset(trainPerc)
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'ADULT':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = AdultDataset(trainPerc)
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'NEWS':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = NewsDataset(trainPerc)
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'GLASS':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = GlassDataset(trainPerc)
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'DIGITS':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = DigitsDataset(trainPerc)
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'BITCOINLATE2012SUB':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = BitcoinLate2012SubDataset(trainPerc)
                pickle.dump(data, open(base_file, 'wb'))

            return data

        if data_name == 'ETHFRAUD':
            if path.exists(base_file):
                data = pickle.load(open(base_file, 'rb'))
            else:
                data = EthFraudDataset(trainPerc)
                pickle.dump(data, open(base_file, 'wb'))

            return data


        return self.datasets[data_name](config)

