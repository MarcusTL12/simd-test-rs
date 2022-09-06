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

pub fn get_amt_done(is: &[T]) -> usize {
    is.iter().filter(|&&i| i <= 4).count()
}

pub fn do_n_colatz(is: &mut [T], maxs: &mut [T], n: usize) {
    for _ in 0..n {
        do_colatz(is);
        update_max(is, maxs);
    }
}

fn do_colatz_const<const N: usize>(is: &mut [T; N]) {
    for i in is {
        *i = if *i % 2 == 0 { *i } else { 3 * *i + 1 } / 2;
    }
}

fn update_max_const<const N: usize>(is: &[T; N], maxs: &mut [T; N]) {
    for (i, maxx) in is.iter().zip(maxs) {
        *maxx = max(*i, *maxx);
    }
}

fn do_n_colatz_const_chunk<const N: usize>(
    is: &mut [T; N],
    maxs: &mut [T; N],
    n: usize,
) {
    for _ in 0..n {
        do_colatz_const(is);
        update_max_const(is, maxs);
    }
}

pub fn do_n_colatz_chunked(is: &mut [T], maxs: &mut [T], n: usize) {
    const CHUNK_SIZE: usize = 64;

    let is_c = is.array_chunks_mut::<CHUNK_SIZE>();
    let maxs_c = maxs.array_chunks_mut::<CHUNK_SIZE>();

    for (is, maxs) in is_c.zip(maxs_c) {
        do_n_colatz_const_chunk(is, maxs, n);
    }

    let is_r = is.array_chunks_mut::<CHUNK_SIZE>().into_remainder();
    let maxs_r = maxs.array_chunks_mut::<CHUNK_SIZE>().into_remainder();

    do_n_colatz(is_r, maxs_r, n);
}

pub fn do_n_colatz_threaded(is: &mut [T], maxs: &mut [T], n: usize) {
    const CHUNK_SIZE: usize = 64;

    let (is_c, is_r) = is.as_chunks_mut::<CHUNK_SIZE>();
    let (maxs_c, maxs_r) = maxs.as_chunks_mut::<CHUNK_SIZE>();

    is_c.par_iter_mut().zip_eq(maxs_c.par_iter_mut()).for_each(
        |(is_c, maxs_c)| {
            do_n_colatz_const_chunk(is_c, maxs_c, n);
        },
    );

    do_n_colatz(is_r, maxs_r, n);
}
