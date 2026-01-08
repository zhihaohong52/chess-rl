"""Configuration for Chess RL training."""


class Config:
    # Input/Output dimensions
    input_size = 781  # 12 bitboards (768) + 4 castling + 8 en passant + 1 side to move
    policy_size = 1924  # All possible chess moves (computed by MoveEncoder)

    # Network architecture (lightweight for fast inference)
    hidden_layers = [512, 512, 256, 256]
    value_hidden = 64

    # MCTS parameters
    num_simulations = 400  # Simulations per move
    c_puct = 1.5  # Exploration constant
    dirichlet_alpha = 0.3  # Noise for root exploration
    dirichlet_epsilon = 0.25  # Weight of noise

    # Self-play parameters
    games_per_iteration = 100
    temperature = 1.0  # Move selection temperature
    temp_threshold = 15  # Moves before temperature drops to near-zero
    max_moves = 512  # Maximum moves per game
    resign_threshold = -0.9  # Resign if value drops below this (speeds up self-play)
    resign_check_moves = 10  # Only check resign after this many moves

    # Training parameters
    batch_size = 512
    learning_rate = 0.0002  # Lower LR for stability (was 0.001)
    weight_decay = 1e-4
    buffer_size = 100000  # Replay buffer size
    training_steps = 500  # Training steps per iteration
    num_iterations = 100  # Total training iterations

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
