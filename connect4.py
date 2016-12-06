import numpy as np

nb_col = 8
nb_row = 10


def init_state():
    """
    :return: [player_id, map, error]
    """
    map = np.full((nb_row, nb_col), -1)
    return {'player': 0,
            'map': map,
            'error': 0,
            'last_move': [-1, -1]}


def is_in(x, y):
    if x < 0 or x >= nb_row or y < 0 or y >= nb_col:
        return False
    return True

def nb_neighboors(state, x, y, player):
    max_count = -1
    for pair in [[-1, -1, 1, 1], [-1, 0, 1, 0], [0, -1, 0, 1]]:
        xup, yup, xdown, ydown = pair
        count = 1
        x_ac, y_ac = x + xup, y + yup
        while is_in(x_ac, y_ac) and state['map'][x_ac][y_ac] == player:
            x_ac += xup
            y_ac += yup
            count += 1
        x_ac, y_ac = x + xdown, y + ydown
        while is_in(x_ac, y_ac) and state['map'][x_ac][y_ac] == player:
            x_ac += xdown
            y_ac += ydown
            count += 1
        if max_count < count:
            max_count = count
    return max_count

def ia_move(state):
    best_score = -1
    for col in xrange(nb_col):
        row = -1
        while state['map'][row][col] != -1:
            row += 1
            if row == nb_row:
                break
        if row == nb_row:
            break
        score = nb_neighboors(state, row, col, 1)
        if score > best_score:
            best_row = row
            best_col = col
            best_score = score

    return make_move(1, state, [best_row, best_col])


def make_move(player, state, action):
    x, y = action[0], action[1]
    if not is_in(x, y):
        state['error'] = 1
    elif x - 1 >= 0 and state['map'][x - 1][y] == -1:
        state['error'] = 1
    else:
        if state['map'][x][y] != -1:
            state['error'] = 2
        else:
            state['map'][x][y] = player
            state['error'] = 0
    if state['error'] != 0:
        state['last_move'] = [-1, -1]
    else:
        state['last_move'] = [x, y]
    state['player'] += 1
    state['player'] %= 2
    return state


def get_reward(state):
    x = state['last_move'][0]
    y = state['last_move'][1]
    player = state['player'] - 1
    if player < 0:
        player = 1
    if x == -1:
        return -5
    else:
        has_won = (nb_neighboors(state, x, y, player) == 4)
    if has_won and player == 0:
        return 10
    elif has_won and player == 1:
        return -10
    if not has_won:
        return -5


def display_grid(state, debug=0):
    if debug == 0:
        print(state)
    else:
        print(state['map'][::-1])


if __name__ == '__main__':
    state = init_state()
    #  print(state)
    state = make_move(0, state, [0, 1])
    display_grid(state)
    print(get_reward(state))
    state = make_move(1, state, [0, 2])
    print(get_reward(state))
    state = make_move(0, state, [1, 2])
    print(get_reward(state))
    state = make_move(1, state, [0, 3])
    print(get_reward(state))
    state = make_move(0, state, [1, 3])
    print(get_reward(state))
    state = make_move(1, state, [0, 4])
    print(get_reward(state))
    state = make_move(0, state, [2, 3])
    print(get_reward(state))
    state = make_move(1, state, [1, 4])
    print(get_reward(state))
    state = make_move(0, state, [2, 4])
    print(get_reward(state))
    state = make_move(1, state, [0, 5])
    print(get_reward(state))
    state = make_move(1, state, [3, 4])
    print(get_reward(state))


