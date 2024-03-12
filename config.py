from dataclasses import dataclass


@dataclass
class Variables:
    UNARY_COLUMNS = ['feature_0', 'feature_73', 'feature_74', 'feature_75']
    INSIGNIFICANT_COLUMNS = ['feature_1', 'feature_2', 'feature_7', 'feature_8',
                             'feature_9', 'feature_13', 'feature_14', 'feature_17',
                             'feature_23', 'feature_47', 'feature_61', 'feature_62']
    CATEGORIAL_COLUMNS = ['feature_3', 'feature_5', 'feature_10',
                          'feature_11', 'feature_12', 'feature_15']
    HIGH_CORRELATION_COLUMNS = ['feature_4', 'feature_50', 'feature_55', 'feature_59',
                                'feature_60', 'feature_63', 'feature_65', 'feature_72',
                                'feature_76', 'feature_78']
    PARAMETERS = {
        'iterations': 1000,
        'custom_metric': ['NDCG'],
        'verbose': True,
        'depth': 4,
        'l2_leaf_reg': 6,
        'random_seed': 0,
        'loss_function': 'RMSE'
    }
