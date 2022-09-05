use std::cmp::max;

use rayon::prelude::*;

pub type T = i64;

fn do_colatz(is: &mut [T]) {
    for i in is {
        *i = if *i % 2 == 0 { *i } else { 3 * *i + 1 } / 2;
    }
}

fn update_max(is: &[T], maxs: &mut [T]) {
    for (i, maxx) in is.iter().zip(maxs) {
        *maxx = max(*i, *maxx);
    }
}

fn amt_done(is: &[T]) -> usize {
    is.iter().filter(|&&i| i <= 4).count()
}

pub fn do_n_colatz(is: &mut [T], maxs: &mut [T], n: usize) -> usize {
    for _ in 0..n {
        do_colatz(is);
        update_max(is, maxs);
    }

    amt_done(is)
}

pub fn do_n_colatz_threaded(
    is: &mut [T],
    maxs: &mut [T],
    n: usize,
) -> usize {
    // let chunk_size = is.len() / nth;
    const CHUNK_SIZE: usize = 1024;

    let (is_c, is_r) = is.as_chunks_mut::<CHUNK_SIZE>();
    let (maxs_c, maxs_r) = maxs.as_chunks_mut::<CHUNK_SIZE>();

    is_c.par_iter_mut().zip_eq(maxs_c.par_iter_mut()).for_each(
        |(is_c, maxs_c)| {
            do_n_colatz(is_c, maxs_c, n);
        },
    );

    do_n_colatz(is_r, maxs_r, n);

    amt_done(is)
}
