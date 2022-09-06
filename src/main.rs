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

    let threads = if let Some(t) = args.next().and_then(|x| x.parse().ok()) {
        t
    } else {
        1
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or(());

    let mut nums: Vec<_> = (1..=n_nums as T).collect();
    let mut maxs = vec![0; n_nums];

    let t = Instant::now();

    // do_n_colatz(&mut nums, &mut maxs, n_iter);
    if threads == 1 {
        do_n_colatz_chunked(&mut nums, &mut maxs, n_iter);
    } else {
        do_n_colatz_threaded(&mut nums, &mut maxs, n_iter);
    }

    let amt_done = get_amt_done(&nums);

    let t = t.elapsed();

    println!("Amt done: {amt_done}");
    let max_reached = maxs.iter().max().unwrap();
    println!("Max reached: {max_reached}");
    println!("Took {t:?}");
}
