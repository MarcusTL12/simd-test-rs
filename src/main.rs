#![feature(slice_as_chunks)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::time::Instant;

// mod matrix;
// pub use matrix::Matrix;

mod colatz;
pub use colatz::*;

fn main() {
    let mut args = std::env::args();
    args.next();
    let n_nums = args.next().unwrap().parse().unwrap();
    let n_iter = args.next().unwrap().parse().unwrap();

    let mut nums: Vec<_> = (1..=n_nums as T).collect();
    let mut maxs = vec![0; n_nums];

    let t = Instant::now();

    let amt_done = do_n_colatz(&mut nums, &mut maxs, n_iter);

    let t = t.elapsed();

    println!("Amt done: {amt_done}");
    println!("Took {t:?}");
}
