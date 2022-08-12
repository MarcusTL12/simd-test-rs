#![feature(slice_as_chunks)]
// #![feature(portable_simd)]
// use std::simd::Simd;

use std::time::Instant;

mod matrix;
pub use matrix::Matrix;

fn main() {
    const N: usize = 50_000;

    let mut m1 = Matrix::<f32>::new_rand(N, N);
    let m2 = Matrix::new_rand(N, N);

    let t = Instant::now();
    m1 += &m2;
    let t = t.elapsed();

    println!("{t:?}");
}
