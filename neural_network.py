import theano as T
from lasagne.layers import DenseLayer, InputLayer
import lasagne
import random
from connect4 import *


class QLearning_NN():
    def __init__(self,
                 n_row,
                 n_col):
        X = T.imatrix()
        y = T.ivector()

        model['input'] = InputLayer(input_var=X,
                                    shape=(None, n_row * n_col))
        model['l_hidden1'] = DenseLayer(model['l_input'],
                                        128,
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform)

        model['l_hidden2'] = DenseLayer(model['l_hidden1'],
                                        64,
                                        nonlinearity=lasagne.nonlinearities.rectify,
                                        W=lasagne.init.GlorotUniform)

        model['l_out'] = DenseLayer(model['l_hidden1'],
                                    8,
                                    nonlinearity=lasagne.nonlinearities.softmax,
                                    W=lasagne.init.GlorotUniform)

        out = lasagne.layers.get_output(model['l_out'])
        all_params = lasagne.layers.get_all_params(model['l_out'])

        loss = lasagne.objectives.binary_crossentropy(out, y)
        optimizer = lasagne.updates.adadelta(loss, all_params)
        self.predict_fn = theano.function([X], out)
        self.train_fn = theano.function([X, y], loss, optimizer)


if __name__ == '__main__':
    n_epochs = 3000
    gamma = 0.975
    epsilon = 1
    batch_size = 40
    buffer = 80
    replay = []
    qnn = QLearning_NN(10, 8)
    h = 0
    for i in xrange(epochs):
        state = init_state()
        status = 1

        while status == 1:
            prediction = qnn.predict_fn(state['map'].reshape(-1, ))
            if random.random() < epsilon:
                action = np.random.randint(0, 8)
            else:
                action = np.argmax(prediction)

            new_state = make_move(0, state, action)
            reward = get_reward(new_state)

            new_state_ennemy = play_IA()
            reward_ennemy = get_reward(new_state_ennemy)

            if len(replay) < buffer:
                replay.append((state, action, reward, new_state, reward_ennemy, new_state_ennemy))
            else:
                if h < (buffer - 1):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state, reward_ennemy, new_state_ennemy)
                mini_batch = random.sample(replay, batch_size)

                X_train = []
                Y_train = []
                for memory in mini_batch:
                    old_state, action, reward, new_state, reward_ennemy, new_state_ennemy = memory
                    old_qval = qnn.predict_fn(old_state['map'].reshape(-1, ))
                    new_qval = qnn.predict_fn(new_state_ennemy['map'].reshape(-1, ))
                    max_qval = np.argmax(new_qval)
                    y = np.zeros((1, 8))
                    y[:] = new_qval[:]
                    if reward == -10 or reward_ennemy == 10:
                        update = -10
                    if reward == 10 or reward_ennemy == -10:
                        update = 10
                    else:
                        update = reward + (gamma * max_qval)
                    y[max_qval] = update
                    X_train.append(old_state['map'].reshape(-1, ))
                    Y_train.append(y)
                qnn.train_fn(X_train,Y_train)
