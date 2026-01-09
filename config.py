"""Configuration for Chess RL training."""

import os


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

    # Network architecture - MEDIUM preset by default
    # Presets (use --model-size in train_full.py):
    #   small:  4 blocks x 64 filters  (~600K params) - fast training
    #   medium: 8 blocks x 128 filters (~3.5M params) - good balance (DEFAULT)
    #   large: 12 blocks x 192 filters (~12M params)  - strongest
    residual_blocks = 8
    residual_filters = 128
    value_hidden = 128

    # Squeeze-and-Excitation attention (improves quality, small speed cost)
    use_se_blocks = True
    se_ratio = 4  # Reduction ratio for SE bottleneck

    # Head architecture
    policy_channels = 8   # Channels in policy head (1x1 conv)
    value_channels = 8    # Channels in value head (1x1 conv)

    # MCTS parameters
    num_simulations = 400  # Simulations per move
    c_puct = 1.5  # Exploration constant
    dirichlet_alpha = 0.2  # Noise for root exploration
    dirichlet_epsilon = 0.15  # Weight of noise
    dirichlet_moves = 12  # Add root noise only for first N moves

    # Self-play parameters
    games_per_iteration = 100
    temperature = 1.0  # Move selection temperature (early game)
    temp_threshold = 15  # Moves before temperature drops
    late_temperature = 0.1  # Temperature after threshold (near-greedy)
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
    supervised_position_source = "random"  # random (fast) or stockfish (slower, guided)
    supervised_position_max_moves = 60
    supervised_position_temperature = 1.5
    supervised_multipv = 4
    supervised_policy_temperature = 1.0
    supervised_augment = True
    supervised_cache_enabled = True
    supervised_cache_dir = "cache/supervised"
    supervised_cache_size = 50000  # Number of positions to cache
    supervised_cache_dtype = "float16"
    supervised_cache_rebuild = False
    supervised_cache_workers = max(1, os.cpu_count() or 1)  # Auto-detect CPU cores
    supervised_cache_batch_size = 64  # Smaller batches for faster progress updates

    # Stockfish settings for parallel cache building
    stockfish_hash_mb = 16  # Hash table per worker (MB)
    stockfish_threads = 1   # Threads per worker (1 is best for multiprocessing)

    # Curriculum learning (Phase 2)
    curriculum_games_per_iteration = 40
    curriculum_training_steps = 200
    curriculum_multipv = 4
    curriculum_policy_temperature = 1.0
    curriculum_policy_weight = 0.7
    curriculum_value_weight = 1.0
    curriculum_initial_depth = 1      # Starting Stockfish depth
    curriculum_max_depth = 12          # Target depth (12+ is very strong, ~2500+ Elo)
    curriculum_limit_strength = False  # Use depth for difficulty, not Elo limiting
    curriculum_elo = 1320              # Only used if limit_strength=True
    curriculum_skill_level = 0
    curriculum_num_parallel = max(1, os.cpu_count() or 4)  # Auto-detect CPU cores
    curriculum_num_simulations = 50  # Lower sims OK with Stockfish teacher

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
