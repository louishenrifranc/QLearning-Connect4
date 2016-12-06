import theano.tensor as T
from lasagne.layers import DenseLayer, InputLayer
import lasagne
import random
from connect4 import *
import theano


class QLearning_NN():
    def __init__(self,
                 n_row,
                 n_col):
        X = T.imatrix()
        y = T.ivector()
        model = []
        model['input'] = InputLayer(shape=(None, n_row * n_col), input_var=X)
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
    qnn = QLearning_NN(nb_row, nb_col)
    h = 0
    for i in xrange(n_epochs):
        # Init the game
        state = init_state()
        status = 1
        # While the game is not ended
        while status == 1:
            # Get a prediction from the Neural network
            prediction = qnn.predict_fn(state['map'].reshape(-1, ))

            if random.random() < epsilon:  # choose random action
                action = np.random.randint(0, nb_col)
            else:  # choose best action
                action = np.argmax(prediction)
            action = [action / 3, action % 3]  # transform action

            #  get new state of the game, and the immediate reward
            new_state = make_move(0, state, action)
            reward = get_reward(new_state)
            if abs(reward) != 10:  # if the game is not over
                new_state_ennemy = ia_move(new_state)
                reward_ennemy = get_reward(new_state_ennemy)
            else:
                new_state_ennemy = new_state
                if reward == 10:
                    reward_ennemy = -10
                elif reward == -10:
                    reward_ennemy = 10

            if len(replay) < buffer:  # If there is not enough example in the replay buffer
                replay.append((state, action, reward, new_state, reward_ennemy, new_state_ennemy))
            else:  # Else there are enough example
                if h < (buffer - 1):  # Replace oldest example
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
                    new_qval = qnn.predict_fn(new_state['map'].reshape(-1, ))
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
                print("Loss: ", qnn.train_fn(X_train, Y_train))
                state = new_state_ennemy
