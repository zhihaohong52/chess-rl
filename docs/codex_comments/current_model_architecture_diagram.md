# Current Model Architecture Diagram

This diagram describes the current `ChessTransformer` path implemented in `src/model/transformer.py`, with heads from `src/model/heads.py` and default dimensions from `config.py`.

## High-Level Flow

```mermaid
flowchart TD
    A[chess.Board] --> B[Canonical orientation<br/>side to move becomes White]
    B --> C[Token encoder]

    C --> D1[Square tokens<br/>shape: B x 64<br/>piece ids: 0..12]
    C --> D2[State features<br/>shape: B x 18<br/>castling, clocks, repetition, etc.]

    D1 --> E1[Piece embedding<br/>13 -> d_model]
    E1 --> E2[Add learned square position embedding<br/>64 x d_model]

    D2 --> F1[State MLP<br/>18 -> d_model]
    F1 --> F2[Broadcast global conditioning<br/>B x 1 x d_model]
    F2 --> E3[Add to all 64 square embeddings]
    E2 --> E3

    E3 --> G[Square representation<br/>B x 64 x 256]
    G --> H[Prepend learned CLS token<br/>B x 65 x 256]

    H --> I[8 x EncoderLayer]
    I --> J[Final LayerNorm]

    J --> K1[Square outputs<br/>B x 64 x 256]
    J --> K2[CLS output<br/>B x 256]

    K1 --> L1[PolicyHead]
    L1 --> M1[Policy logits<br/>B x 1924]

    K2 --> L2{Value head type}
    L2 -->|default: wdl| M2[WDL logits<br/>B x 3]
    L2 -->|optional: hlgauss| M3[Distributional value logits<br/>B x value_buckets]

    K2 --> L3[MovesLeftHead]
    L3 --> M4[Moves-left prediction<br/>B x 1]
```

## Encoder Layer Detail

Each transformer block uses side-to-move canonical square embeddings and a Smolgen-style learned attention bias.

```mermaid
flowchart TD
    A[Layer input<br/>B x 65 x d_model] --> B[Split]

    B --> C[Square tokens only<br/>B x 64 x d_model]
    C --> D[Smolgen]
    D --> E[Per-head attention bias<br/>B x n_heads x 64 x 64]

    B --> F[LayerNorm]
    F --> G[Biased multi-head self-attention]
    E --> G
    G --> H[Residual add]

    H --> I[LayerNorm]
    I --> J[Feed-forward network<br/>MLP or configured FFN]
    J --> K[Residual add]
    K --> L[Layer output]
```

Note: Smolgen is generated from the 64 square tokens. The model sequence also contains a CLS token, so implementations must be careful about how the 64 x 64 bias is applied relative to the 65-token attention sequence.

## Policy Head Shape

The policy head maps square representations to the fixed move-index vocabulary used by `MoveEncoder`.

```mermaid
flowchart LR
    A[Square outputs<br/>B x 64 x d_model] --> B[From-square features]
    A --> C[To-square features]
    A --> D[Promotion class features]

    B --> E[Move-index scorer]
    C --> E
    D --> E

    E --> F[1924 policy logits]
    F --> G[Evaluation masks to legal moves]
    G --> H[Softmax over legal moves for MCTS priors]
```

The current interface emits all 1924 logits. During evaluation, illegal moves are filtered by the legal move list before normalization.

## Search Integration

```mermaid
flowchart TD
    A[Current board] --> B[TransformerEvaluator]
    B --> C[Encode board batch]
    C --> D[ChessTransformer]

    D --> E[Policy logits]
    D --> F[WDL or distributional value]

    E --> G[Legal move mask + softmax]
    G --> H[MCTS priors P(s,a)]

    F --> I[Expected value]
    I --> J[MCTS leaf value]

    H --> K[Batched PUCT MCTS]
    J --> K
    K --> L[Move selection]
```

## Default Architecture Constants

```text
d_model:             256
n_layers:            8
n_heads:             8
d_ff:                1024
state_dim:           18
policy_size:         1924
value_head_type:     wdl
value_buckets:       64 when using hlgauss
smolgen_compress:    32
smolgen_hidden:      128
smolgen_gen:         128
transformer_dropout: 0.0
baseline params:     about 10.3M
```

## Architectural Interpretation

The current model is best understood as:

```text
canonical board tokenizer
-> square-token transformer with learned relational attention bias
-> flat move-policy head
-> global value head from CLS
-> moves-left auxiliary head
-> MCTS evaluator that masks illegal moves and consumes policy/value outputs
```

It is a strong prototype architecture for policy/value distillation, but it is still missing richer history representation, explicit legal-move feature scoring, and search-trained policy/value targets for a Stockfish-beating trajectory.
