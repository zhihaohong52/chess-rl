use crate::{board::Board, search::{Limits, Searcher}};
const FENS: [&str; 10] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "8/8/1p1k4/p1p2p1p/P1P2P1P/1P1K4/8/8 w - - 0 1",
    "2rq1rk1/pb1nbppp/1p2pn2/2pp4/3P1B2/2NBPN2/PPQ2PPP/R4RK1 w - - 0 11",
];
pub fn bench(){let start=std::time::Instant::now();let mut nodes=0;for fen in FENS{let mut board=Board::from_fen(fen).unwrap();let mut s=Searcher::new(16);s.think(&mut board,&Limits{depth:Some(6),..Default::default()},&[]);nodes+=s.node_count()}println!("bench: {nodes} nodes {:.0} nps",nodes as f64/start.elapsed().as_secs_f64())}
