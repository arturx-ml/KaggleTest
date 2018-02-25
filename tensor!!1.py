import tensorflow as tf
import numpy as np
import sklearn
import pandas as pd

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]


def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('%s not found, check it!' % data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip('feat', columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '<50K')

    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator
    features, labels = iterator.get_next()
    return features, labels


pclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'pclass', [1, 2, 3]
)
sex = tf.feature_column.categorical_column_with_vocabulary_list(
    'sex',['male','female']
)
age = tf.feature_column.numeric_column('age')
sibsp = tf.feature_column.categorical_column_with_hash_bucket(
    'sibsp',hash_bucket_size=20
)
parch = tf.feature_column.categorical_column_with_hash_bucket(
    'parch',hash_bucket_size=20
)
ticket = tf.feature_column.numeric_column('ticket')
fare = tf.feature_column.numeric_column('fare')
cabin = tf.feature_column.categorical_column_with_hash_bucket(
    'cabin',hash_bucket_size=10000
)
embarked = tf.feature_column.categorical_column_with_vocabulary_list(
    'embarked',['C','Q','S']
)