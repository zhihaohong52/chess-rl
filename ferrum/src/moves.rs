use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move(pub u16);

pub const FLAG_QUIET: u16 = 0;
pub const FLAG_DPP: u16 = 1;
pub const FLAG_OO: u16 = 2;
pub const FLAG_OOO: u16 = 3;
pub const FLAG_CAP: u16 = 4;
pub const FLAG_EP: u16 = 5;
pub const FLAG_PROMO: u16 = 8; // 8..=11 quiet promo N/B/R/Q; 12..=15 capture promo

impl Move {
    pub const NONE: Move = Move(0);
    pub fn new(from: u8, to: u8, flags: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (flags << 12))
    }
    pub fn from(self) -> u8 { (self.0 & 63) as u8 }
    pub fn to(self) -> u8 { ((self.0 >> 6) & 63) as u8 }
    pub fn flags(self) -> u16 { self.0 >> 12 }
    pub fn is_capture(self) -> bool { self.flags() & FLAG_CAP != 0 }
    pub fn is_promo(self) -> bool { self.flags() & FLAG_PROMO != 0 }
    pub fn promo_pt(self) -> usize { KNIGHT + (self.flags() & 3) as usize }
    pub fn uci(self) -> String {
        let mut s = format!("{}{}", sq_name(self.from()), sq_name(self.to()));
        if self.is_promo() { s.push(b"nbrq"[(self.flags() & 3) as usize] as char); }
        s
    }
}
