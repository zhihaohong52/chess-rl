use crate::moves::Move;
pub const BOUND_EXACT: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_UPPER: u8 = 3;
#[derive(Clone, Copy)] pub struct Entry { pub key: u64, pub mv: Move, pub score: i32, pub depth: i8, pub bound: u8 }
pub struct Tt { entries: Vec<Entry>, mask: usize }
impl Tt {
    pub fn new(mb: usize) -> Self {
        let entries = ((mb.max(1) * 1024 * 1024 / std::mem::size_of::<Entry>()).next_power_of_two() / 2).max(1);
        Self { entries: vec![Entry { key: 0, mv: Move::NONE, score: 0, depth: -1, bound: 0 }; entries], mask: entries - 1 }
    }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn clear(&mut self) { self.entries.fill(Entry { key: 0, mv: Move::NONE, score: 0, depth: -1, bound: 0 }); }
    pub fn probe(&self, key: u64) -> Option<Entry> { let e = self.entries[key as usize & self.mask]; (e.bound != 0 && e.key == key).then_some(e) }
    pub fn store(&mut self, key: u64, mv: Move, score: i32, depth: i8, bound: u8) { self.entries[key as usize & self.mask] = Entry { key, mv, score, depth, bound }; }
}
#[cfg(test)] mod tests { use super::*; #[test] fn store_probe_roundtrip() { let mut tt=Tt::new(1); tt.store(0xdead_beef, Move::new(12,28,0),42,7,BOUND_EXACT); let e=tt.probe(0xdead_beef).unwrap(); assert_eq!(e.score,42); assert_eq!(e.depth,7); assert_eq!(e.mv,Move::new(12,28,0)); assert!(tt.probe(0xdead_bee0).is_none()); } #[test] fn size_is_power_of_two(){ assert!(Tt::new(16).len().is_power_of_two()); } }
