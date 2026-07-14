use std::sync::OnceLock;

pub struct Zobrist {
    pub piece: [[u64; 64]; 12],
    pub side: u64,
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
}

static Z: OnceLock<Zobrist> = OnceLock::new();

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn z() -> &'static Zobrist {
    Z.get_or_init(|| {
        let mut s = 0x00C0_FFEE_u64;
        let mut zb = Zobrist {
            piece: [[0; 64]; 12], side: 0, castling: [0; 16], ep_file: [0; 8],
        };
        for p in 0..12 { for q in 0..64 { zb.piece[p][q] = splitmix(&mut s); } }
        zb.side = splitmix(&mut s);
        for c in 0..16 { zb.castling[c] = splitmix(&mut s); }
        for f in 0..8 { zb.ep_file[f] = splitmix(&mut s); }
        zb
    })
}
