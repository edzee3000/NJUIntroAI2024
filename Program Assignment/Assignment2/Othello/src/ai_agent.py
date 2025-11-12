from othello_game import OthelloGame

 
def get_best_move(game, max_depth=6):
    """
    黑棋为1，白棋为-1，空为0，执黑棋的玩家为1，执白棋的玩家为-1    在游戏里面由于玩家先手，因此1为玩家，-1为AI
    Given the current game state, this function returns the best move for the AI player using the Alpha-Beta Pruning
    algorithm with a specified maximum search depth.

    Parameters:
        game (OthelloGame): The current game state.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    
    get_best_move函数
    这个函数是AI决策的入口点，它接受当前游戏状态和最大搜索深度作为参数。
    根据当前玩家是否为AI，选择适当的策略（Minimax、MTD(f)或Alpha-Beta剪枝）来决定最佳移动。
    """
    player = True  #player为True表示AI
    if game.current_player == -1:#-1表示是玩家
        player = False
    # _, best_move = minmax_decider(game, max_depth,player)#使用minmax算法
    # _, best_move = mtd_f(game, 0, max_depth)#使用mtd_f算法
    _, best_move = alphabeta_decider(game, max_depth, player)#使用alphabeta剪枝策略
    return best_move


def minmax_decider(
    game, 
    max_depth, 
    maximizing_player=True
):
    """
    MinMax Decider algorithm for selecting the best move for the AI player.

    Parameters:
        game (OthelloGame): The current game state.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.
        maximizing_player (bool): True if maximizing player (AI), False if minimizing player (opponent).

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    
    MinMax决策算法，用于为AI玩家选择最佳移动。
    参数：
        game (OthelloGame): 当前游戏状态。
        max_depth (int): Alpha-Beta算法的最大搜索深度。
        maximizing_player (bool): 如果是最大化玩家（AI），则为True；如果是最小化玩家（对手），则为False。

    返回值：
        tuple: 包含最佳移动的评估值和相应移动（行，列）的元组。

    minmax_decider函数:
    实现了Minimax算法，用于选择最佳移动。
    通过递归地探索所有可能的移动，选择使对手处于最差情况的移动。
    考虑了最大搜索深度和游戏结束条件。
    """
    if max_depth == 0 or game.is_game_over():
        return evaluate_game_state(game), None

    valid_moves = game.get_valid_moves()

    if maximizing_player:#如果是AI的话
        max_eval = float("-inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = minmax_decider(new_game, max_depth - 1, False)

            if eval > max_eval:
                max_eval = eval
                best_move = move

        return max_eval, best_move
    else:#如果是玩家的话
        min_eval = float("inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = minmax_decider(new_game, max_depth - 1, True)

            if eval < min_eval:
                min_eval = eval
                best_move = move

        return min_eval, best_move
    









def alphabeta_decider(
    game, max_depth, maximizing_player=True, alpha=float("-inf"), beta=float("inf")
):#需要再此处加入AlphaBeta剪枝
    """
    MinMax Decider algorithm for selecting the best move for the AI player.MinMax Decider算法用于为AI玩家选择最佳移动
    alphabeta_decider函数
    这个函数的框架已经搭建好，但实现细节尚未填充。
    它应该实现Alpha-Beta剪枝，这是一种优化的Minimax算法，可以减少搜索空间并提高效率。
    """
    # Your implementation for Alpha beta pruning
    if max_depth == 0 or game.is_game_over():
        return evaluate_game_state(game), None
    valid_moves = game.get_valid_moves()
    if maximizing_player:#如果是AI的话
        max_eval = float("-inf")
        best_move = None
        for move in valid_moves:
            #######################################################
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)
            #######################################################
            eval, _ = alphabeta_decider(new_game, max_depth - 1, False, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
                alpha=max_eval
            if beta<=alpha:
                #print("max_eval为：",min_eval)
                return max_eval, best_move
        return max_eval, best_move
    else:#如果是玩家的话
        min_eval = float("inf")
        best_move = None
        for move in valid_moves:
            #######################################################
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)
            #######################################################
            eval, _ = alphabeta_decider(new_game, max_depth - 1, True, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
                beta=min_eval
            if beta<=alpha:
                #print("min_eval为：",min_eval)
                return min_eval, best_move
        return min_eval, best_move







def mtd_f(game, guess, max_depth):
    """
    MTD(f) algorithm for selecting the best move for the AI player.
    
    Parameters:
        game (OthelloGame): The current game state.
        guess (float): The initial guess for the evaluation value.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
     MTD(f)算法用于为AI玩家选择最佳移动。
    
    参数：
        game (OthelloGame): 当前游戏状态。
        guess (float): 评估值的初始猜测。
        max_depth (int): Alpha-Beta算法的最大搜索深度。

    返回值：
        tuple: 包含最佳移动的评估值和相应移动（行，列）的元组。
    mtd_f函数
    实现了MTD(f)算法，它是一种结合了Minimax和Alpha-Beta剪枝的算法。
    它使用一个“猜测”值来缩小搜索范围，并使用Alpha-Beta剪枝来进一步优化搜索。

    """
    # Initialize alpha and beta bounds
    lower_bound = float("-inf")
    upper_bound = float("inf")
    
    g = guess
    
    while lower_bound < upper_bound:
        if g == lower_bound:
            beta = g + 1
        else:
            beta = g
        
        # Perform a zero-window search using alpha-beta pruning
        g, best_move = alphabeta_decider(game, max_depth, alpha=beta - 1, beta=beta, maximizing_player=True)
        
        # Update the bounds based on the result
        if g < beta:
            upper_bound = g
        else:
            lower_bound = g
    
    return g, best_move








def evaluate_game_state(game):#需要对其进行改进
    """
    Evaluates the current game state for the AI player.
    对于AI player评估当前游戏状态的好坏
    Parameters:
        game (OthelloGame): The current game state.
    参数是当前游戏状态
    Returns:
        float: The evaluation value representing the desirability of the game state for the AI player.
    评估AI玩家的当前游戏状态。

    参数：
        game (OthelloGame): 当前游戏状态。

    返回值：
        float: 表示AI玩家对游戏状态的期望值。
    evaluate_game_state函数
    用于评估当前游戏状态对AI玩家的吸引力。
    它考虑了多个因素，如硬币平衡（棋子数量差异）、机动性（有效移动数）、角位占用和边缘占用。
    新增了对游戏结束状态的判断，如果游戏结束，根据获胜方给予一个很大的正或负的常数，以表示游戏结果对评估值的影响。
    """
    # 黑-白
    # Evaluation weights for different factors
    coin_parity_weight = 1  #硬币平衡权重，用来评估棋子数量差异的重要性
    mobility_weight = 2 #机动性权重，用来评估有效移动数的重要性
    corner_occupancy_weight = 100 #角位占用权重，用来评估角位上棋子的重要性
    stability_weight = 1  #稳定性权重，用来评估稳定棋子（不会被对手翻转的棋子）的重要性
    edge_occupancy_weight = 8  #边缘占用权重，用来评估边缘上棋子的重要性

    # Coin parity (difference in disk count)
    black_disk_count = sum(row.count(game.current_player) for row in game.board)
    white_disk_count = sum(row.count(-game.current_player) for row in game.board)
    coin_parity = black_disk_count - white_disk_count #计算当前AI和对手的棋子数量差异

    # Mobility (number of valid moves for the current player)
    current_valid_moves = len(game.get_valid_moves())
    opponent_valid_moves = len(
        OthelloGame(player_mode=-game.current_player).get_valid_moves()
    )
    if (game.current_player == 1):
        black_valid_moves = current_valid_moves
        white_valid_moves = opponent_valid_moves
    else:
        black_valid_moves = opponent_valid_moves
        white_valid_moves = current_valid_moves
    mobility = black_valid_moves - white_valid_moves #计算当前玩家和对手的有效移动数，并计算两者的差异

    # Corner occupancy (number of player disks in the corners)计算棋盘四个角上当前玩家的棋子数量
    corner_occupancy = sum(
        game.board[i][j] for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)]
    )

    # Stability (number of stable disks)
    stability = calculate_stability(game)#调用 calculate_stability 函数计算稳定棋子的数量。
        
    # Edge occupancy (number of player disks on the edges) 计算棋盘边缘上当前玩家的棋子数量
    edge_occupancy = sum(game.board[i][j] for i in [0, 7] for j in range(1, 7)) + sum(
        game.board[i][j] for i in range(1, 7) for j in [0, 7]
    )

    # winner constant
    win_constant = 0
    if game.is_game_over():
        winner = game.get_winner()
        if (winner == 1):
            win_constant = 10000
        else:
            win_constant = -10000


    # Combine the factors with the corresponding weights to get the final evaluation value
    evaluation = (
        coin_parity * coin_parity_weight
        + mobility * mobility_weight
        + corner_occupancy * corner_occupancy_weight
        + stability * stability_weight
        + edge_occupancy * edge_occupancy_weight
        + win_constant
    )

    return evaluation










def calculate_stability(game):
    """
    Calculates the stability of the AI player's disks on the board.

    Parameters:
        game (OthelloGame): The current game state.

    Returns:
        int: The number of stable disks for the AI player.
    calculate_stability函数
    用于计算AI玩家的棋子稳定性。
    它检查每个棋子是否被对手的棋子包围，或者是否位于棋盘的边缘或角落。
    修改了函数，现在它计算黑白双方的稳定棋子数量，并返回两者的差值。
    """
 
    def neighbors(row, col):
        return [
            (row + dr, col + dc)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
        ]

    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    edges = [(i, j) for i in [0, 7] for j in range(1, 7)] + [
        (i, j) for i in range(1, 7) for j in [0, 7]
    ]
    inner_region = [(i, j) for i in range(2, 6) for j in range(2, 6)]
    regions = [corners, edges, inner_region]

    black_stable = 0
    white_stable = 0


    def is_stable_disk(row, col,player):
        return (
            all(game.board[r][c] == game.current_player for r, c in neighbors(row, col))
            or (row, col) in edges + corners
        )

    for region in regions:
        for row, col in region:
            if game.board[row][col] == 1:
                black_stable += 1 if is_stable_disk(row, col, 1) else 0
            if game.board[row][col] == 0:
                white_stable += 1 if is_stable_disk(row, col, -1) else 0

    return black_stable-white_stable
