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
    def recursive_check(state, x, y, depth, player, visited):
        if depth == 3:
            return True
        if depth == 0:
            visited.append([x, y])
        won = False
        for i in xrange(-1, 2):
            for j in xrange(-1, 2):
                if is_in(x + i, y + j):
                    if (i != 0 or j != 0):
                        if not ([x + i, y + j] in visited):
                            # print(x + i, y + j, state['map'][x + i][y + j], player)
                            if state['map'][x + i][y + j] == player:
                                visited.append([x + i, y + j])
                                won = won or recursive_check(state, x + i, y + j, depth + 1, player, visited)
                                visited.pop()
        return won

    x = state['last_move'][0]
    y = state['last_move'][1]
    player = state['player'] - 1
    if player < 0:
        player = 1
    if x == -1:
        return -6
    else:
        has_won = recursive_check(state, x, y, 0, player, [])
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
    state = make_move(0, state, [1, 1])
    print(get_reward(state))
    state = make_move(1, state, [1, 2])
    print(get_reward(state))
    state = make_move(0, state, [2, 1])
    print(get_reward(state))
    state = make_move(1, state, [2, 2])
    print(get_reward(state))
    state = make_move(0, state, [0, 3])
    print(get_reward(state))
    state = make_move(1, state, [3, 2])
    print(get_reward(state))
