"""Configuration for Chess RL training."""


class Config:
    # Input representation (8x8 planes with history)
    board_size = 8
    history_length = 4  # Number of past positions to include (current + N-1)
    include_ep_plane = True
    extra_planes = 7 + (1 if include_ep_plane else 0)  # castling(4) + side(1) + repetition(1) + move_count(1) + ep(1)
    input_planes = 12 * history_length + extra_planes
    input_shape = (board_size, board_size, input_planes)
    input_size = board_size * board_size * input_planes
    policy_size = 1924  # All possible chess moves (computed by MoveEncoder)

    # Network architecture (lightweight residual conv net)
    residual_blocks = 4
    residual_filters = 64
    value_hidden = 64

    # MCTS parameters
    num_simulations = 400  # Simulations per move
    c_puct = 1.5  # Exploration constant
    dirichlet_alpha = 0.2  # Noise for root exploration
    dirichlet_epsilon = 0.15  # Weight of noise
    dirichlet_moves = 12  # Add root noise only for first N moves

    # Self-play parameters
    games_per_iteration = 100
    temperature = 1.0  # Move selection temperature
    temp_threshold = 15  # Moves before temperature drops to near-zero
    max_moves = 512  # Maximum moves per game
    resign_threshold = -0.9  # Resign if value drops below this (speeds up self-play)
    resign_check_moves = 10  # Only check resign after this many moves
    value_target_mix = 0.3  # Mix MCTS root value into training targets

    # Training parameters
    batch_size = 512
    learning_rate = 0.0002  # Lower LR for stability (was 0.001)
    weight_decay = 1e-4
    use_lr_schedule = True
    lr_total_steps = 200000
    lr_final_alpha = 0.05
    buffer_size = 100000  # Replay buffer size
    training_steps = 500  # Training steps per iteration
    num_iterations = 100  # Total training iterations

    # Supervised learning (Phase 1)
    supervised_position_source = "stockfish"  # stockfish or random
    supervised_position_max_moves = 60
    supervised_position_temperature = 1.5
    supervised_multipv = 4
    supervised_policy_temperature = 1.0
    supervised_augment = True

    # Curriculum learning (Phase 2)
    curriculum_games_per_iteration = 40
    curriculum_training_steps = 200
    curriculum_multipv = 4
    curriculum_policy_temperature = 1.0
    curriculum_policy_weight = 0.7
    curriculum_value_weight = 1.0
    curriculum_limit_strength = True
    curriculum_elo = 800
    curriculum_skill_level = 0

    # Checkpointing
    checkpoint_interval = 5
    checkpoint_dir = "checkpoints"

    # Colab-specific settings
    colab_checkpoint_dir = "/content/drive/MyDrive/chess-rl/checkpoints"
    use_mixed_precision = True

    # Training schedule (for progressive training)
    warmup_iterations = 10
    warmup_simulations = 100
    warmup_games = 50
    main_simulations = 200
    main_games = 100
    refinement_simulations = 400
