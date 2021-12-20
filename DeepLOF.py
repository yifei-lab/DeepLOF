import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
tfd = tfp.distributions
import argparse


# integrated constraint model
class ConstraintModel(object):
    def __init__(self, feature, count, feature_val, count_val,
                 n_hidden=None, dropout_rate=0.5,
                 l2_reg = 0., grid_points=1000):
        # currently only support float32
        self.dtype = 'float32'
        
        # training data
        self.count = tf.convert_to_tensor(count, self.dtype)
        self.feature = tf.convert_to_tensor(feature, self.dtype)

        # validation data
        self.count_val = tf.convert_to_tensor(count_val, self.dtype)
        self.feature_val = tf.convert_to_tensor(feature_val, self.dtype)
        
        # grid for numerical integration
        self.grid = np.arange(0.5/grid_points, 1.,
                              1./grid_points)[np.newaxis, :]
        self.grid = tf.convert_to_tensor(self.grid, self.dtype)
        
        # input for keras model
        feature_input = Input(shape=(feature.shape[1],), dtype=self.dtype)
        
        # network without likelihood (keras model)
        self.network = ConstraintModel.build_network(feature.shape[1],
                                                     n_hidden,
                                                     dropout_rate,
                                                     l2_reg,
                                                     self.dtype)
        # build model for training
        dist = self.network(feature_input)
        self.training_model = Model(inputs=feature_input, outputs=dist)
        
    def fit(self, model_save_file, learning_rate=0.01, batch_size=64, epochs=10, patience=5):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    restore_best_weights=True,
                                                    patience=patience)

        model_saver = tf.keras.callbacks.ModelCheckpoint(model_save_file,
                                                         save_best_only=True,
                                                         monitor='val_loss')
        
        # compile model
        self.training_model.compile(Adam(learning_rate=learning_rate),
                                    loss=self.nll)
        
        # fit model
        print(self.training_model.summary())
        self.history = self.training_model.fit(self.feature, self.count,
                                               batch_size=batch_size,
                                               validation_data=(self.feature_val,
                                                                self.count_val),
                                               epochs=epochs,
                                               callbacks=[callback, model_saver])

        self.best_val_loss = np.min(self.history.history['val_loss'])

    def predict(self, feature, count):
        y = tf.convert_to_tensor(count, self.dtype)
        
        # beta prior
        pred_dist = self.training_model(feature)
        prior = pred_dist.log_prob(self.grid)
        
        # observed count
        obs_count = y[:, 0][:, tf.newaxis]
        
        # expected count without selection
        exp_count = y[:, 1][:, tf.newaxis]
        
        # predicted rate with selection
        rate = exp_count * self.grid
        
        # partial likelihood without Beta prior
        lik = tfd.Poisson(rate).log_prob(obs_count)
        
        prob = tf.math.exp(lik + prior)
        prob = prob / tf.math.reduce_sum(prob, axis = 1, keepdims=True)
        relative_rate = tf.math.reduce_sum(prob * self.grid, axis=1)
        constraint = 1 - relative_rate
        return constraint.numpy(), 1 - pred_dist.mean().numpy().flatten()
        
    def nll(self, y, dist):
        # observed count
        obs_count = y[:, 0][:, tf.newaxis]
        
        # expected count without selection
        exp_count = y[:, 1][:, tf.newaxis]
        
        # predicted rate with selection
        rate = exp_count * self.grid
        
        # Beta prior
        prior = dist.log_prob(self.grid)
        
        # partial likelihood without Beta prior
        lik = tfd.Poisson(rate).log_prob(obs_count)

        ll = tfp.math.reduce_logmeanexp(lik + prior, axis=1)
        return -ll
    
    @staticmethod
    def beta_dist(params):
        mean = tf.math.sigmoid(params[:, 0])
        kappa = tf.math.exp(params[:, 1])
        alpha = mean * kappa
        beta = (1. - mean) * kappa
        return tfd.Beta(alpha[:, tf.newaxis],
                        beta[:, tf.newaxis])
 
    @staticmethod
    def build_network(n_feat, n_hidden, dropout_rate, l2_reg,
                      dtype, activation='relu'):
        # input
        inputs = Input(shape=(n_feat,))
        
        regularizer = tf.keras.regularizers.l2(l2_reg)
        
        if n_hidden is None:
            params = Dense(2, dtype=dtype)(inputs)
        else:
            hidden = inputs
            for n in n_hidden:
                hidden = Dense(n, activation=activation, dtype=dtype,
                               kernel_regularizer=regularizer)(hidden)
                hidden = Dropout(rate=dropout_rate, dtype=dtype)(hidden)
            params = Dense(2, dtype=dtype)(hidden)
            
        dist = tfp.layers.DistributionLambda(ConstraintModel.beta_dist)(params)
        return Model(inputs=inputs, outputs=dist, name='betaOutputLayer')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', dest='input', type=str, required=True,
                        help='input file')

    parser.add_argument('--output', dest='output', type=str, required=True,
                        help='out file')

    parser.add_argument('--hidden', dest='hidden', type=int, required=False,
                        nargs='+', default=None, help='hidden units')

    parser.add_argument('--learning-rate', dest='rate', type=float, required=False,
                        default=0.001, help='learning rate for the Adam algorithm')

    parser.add_argument('--penalty', dest='penalty', type=float, required=False,
                        default=0, help='L2 penalty')

    parser.add_argument('--dropout', dest='dropout', type=float, required=False,
                        default=0.5, help='dropout rate')

    parser.add_argument('--patience', dest='patience', type=int, required=False,
                        default=5, help='patience for early stopping')

    parser.add_argument('--epochs', dest='epochs', type=int, required=False,
                        default=100, help='number of epochs')

    parser.add_argument('--batch', dest='batch', type=int, required=False,
                        default=64, help='batch size')

    parser.add_argument('--validation-fraction', dest='validation_fraction', type=float, required=False,
                        default=0.2, help='proportion of data for validation')

    parser.add_argument('--data-seed', dest='data_seed', type=int, required=False,
                        default=None, help='random seed for spliting data')

    parser.add_argument('--model-seed', dest='model_seed', type=int, required=False,
                        default=None, help='random seed for model')

    parser.add_argument('--save-model', dest='save_model', type=str, required=False,
                        default=None, help='output directory for the trained keras model')

    args = parser.parse_args()

    if args.model_seed is not None:
        tf.random.set_seed(args.model_seed)

    df = pd.read_csv(args.input, sep='\t')
    count_data = df.iloc[:, 1:3].values
    feature = df.iloc[:, 3:].values

    if args.data_seed is not None:
        feature_train, feature_val, count_train, count_val = train_test_split(feature,
                count_data, test_size=args.validation_fraction, random_state=args.data_seed)
    else:
        feature_train, feature_val, count_train, count_val = train_test_split(feature,
                count_data, test_size=args.validation_fraction)

    model = ConstraintModel(feature_train, count_train, feature_val, count_val,
                            n_hidden=args.hidden, dropout_rate=args.dropout,
                            l2_reg=args.penalty)

    model.fit(args.output + '.hdf5', args.rate, args.batch, args.epochs, args.patience)

    constraint, constraint_by_feature = model.predict(feature, count_data)
    out_data = pd.DataFrame.from_dict({df.columns[0]: df.iloc[:, 0],
        'DeepLOF_score': constraint})

    out_data.to_csv(args.output, index=False, sep='\t')

    print('Best validation loss = {}'.format(model.best_val_loss))

    if args.save_model is not None:
        output_model = Sequential()

        for layer in model.network.layers[:-1]:
            output_model.add(layer)

        output_model.compile()
        output_model.save(args.save_model)
