//! Self-play data generation (M6). Plays fixed-node games with the v0.4.0 engine,
//! filters to quiet positions, and streams White-relative `FEN | score | wdl` text
//! for bullet training. Reuses search/board/eval unchanged.

/// Deterministic, seedable xorshift64. Used ONLY for opening-move selection, so
/// datagen is reproducible per seed and decorrelated across worker seeds.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        // xorshift cannot start from 0; substitute a fixed nonzero constant.
        Self { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform-ish integer in `0..n` (modulo bias is negligible for the small `n`
    /// used in opening selection). Panics if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prng_is_deterministic_per_seed_and_varies_across_seeds() {
        let mut a = XorShift64::new(12345);
        let seq_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        // Same seed reproduces the exact sequence.
        let mut a2 = XorShift64::new(12345);
        let seq_a2: Vec<u64> = (0..8).map(|_| a2.next_u64()).collect();
        assert_eq!(seq_a, seq_a2, "same seed must reproduce the sequence");
        // Different seed diverges.
        let mut b = XorShift64::new(67890);
        let seq_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b, "different seeds must diverge");
        // No zero-lock, values vary.
        assert!(seq_a.iter().all(|&x| x != 0));
        assert!(seq_a[0] != seq_a[1]);
    }

    #[test]
    fn below_stays_in_range() {
        let mut r = XorShift64::new(1);
        for _ in 0..1000 {
            assert!(r.below(20) < 20);
            assert!(r.below(1) == 0);
        }
    }
}
