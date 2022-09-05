use std::cmp::max;

pub type T = i64;

fn do_colatz(is: &mut [T]) {
    for i in is {
        *i = if *i % 2 == 0 { *i / 2 } else { 3 * *i + 1 };
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
