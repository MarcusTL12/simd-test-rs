#![feature(slice_as_chunks)]
#![feature(portable_simd)]
use std::simd::Simd;

use std::time::Instant;

type T = f32;

pub fn myfunc(d: &mut [T], u: &[T], v: &[T]) {
    for (c, (a, b)) in d.iter_mut().zip(u.iter().zip(v.iter())) {
        *c = a + b;
    }
}

fn fillvec(v: &mut [T]) {
    const N: usize = 17;

    let mut buf = [0.0; N];
    for (i, x) in buf.iter_mut().enumerate() {
        *x = (i as T) - ((N - 1) as T) / 2.0;
    }

    let (cs, r) = v.as_chunks_mut();
    for c in cs {
        *c = buf;
    }

    for (y, x) in r.iter_mut().zip(buf) {
        *y = x;
    }
}

pub fn create_vec(n: usize) -> Vec<T> {
    let mut v = vec![0.0; n];

    fillvec(&mut v);

    v
}

pub fn dot_naive(u: &[T], v: &[T]) -> T {
    u.iter().zip(v).map(|(a, b)| a * b).sum()
}

pub fn dot_simd(u: &[T], v: &[T]) -> T {
    const N: usize = 16;

    let (ucs, ur) = u.as_chunks::<N>();
    let (vcs, vr) = v.as_chunks::<N>();

    let s: Simd<_, N> = ucs
        .iter()
        .zip(vcs)
        .map(|(uc, vc)| {
            let uc = Simd::<_, N>::from_slice(uc);
            let vc = Simd::<_, N>::from_slice(vc);

            uc * vc
        })
        .sum();

    s.to_array().iter().sum::<T>()
        + ur.iter().zip(vr).map(|(a, b)| a * b).sum::<T>()
}

fn main() {
    const N: usize = 100_000_000;
    const N_RUNS: usize = 10;

    let u = create_vec(N);
    let mut v = create_vec(N);
    v.reverse();

    let t = Instant::now();
    let x: T = (0..N_RUNS).map(|_| dot_naive(&u, &v)).sum();
    let t = t.elapsed();
    println!("x: {x}\nt: {t:?}\n");

    let t = Instant::now();
    let x: T = (0..N_RUNS).map(|_| dot_simd(&u, &v)).sum();
    let t = t.elapsed();
    println!("x: {x}\nt: {t:?}\n");
}
