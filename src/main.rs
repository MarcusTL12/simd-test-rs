#![feature(slice_as_chunks)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::{env::Args, time::Instant};

use rand::Rng;

pub mod linalg;

pub mod colatz;

pub mod lennard_jones;
pub mod lennard_jones_t;

pub mod transpose_u8;

fn set_threads(args: &mut Args) -> usize {
    let threads = if let Some(t) = args.next().and_then(|x| x.parse().ok()) {
        t
    } else {
        1
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or(());

    threads
}

fn main() {
    let mut args = std::env::args();
    args.next();

    match args.next().unwrap().as_str() {
        "colatz" => {
            use colatz::*;

            let n_nums = args.next().unwrap().parse().unwrap();
            let n_iter = args.next().unwrap().parse().unwrap();

            let threads = set_threads(&mut args);

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
        "dot" => {
            use linalg::vector::*;

            let n_nums = args.next().unwrap().parse().unwrap();

            let t = Instant::now();
            let v = rand_vec::<1000003, _, _>(n_nums, -100f64..100.0);
            let w = rand_vec::<1000033, _, _>(n_nums, -100f64..100.0);
            let t = t.elapsed();

            println!("Took {t:?} to create rand vectors.");

            {
                let t = Instant::now();
                let d = dot_naive(&v, &w);
                let t = t.elapsed();

                println!(" Naive: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<2, _>(&v, &w);
                let t = t.elapsed();

                println!("     2: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<4, _>(&v, &w);
                let t = t.elapsed();

                println!("     4: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<8, _>(&v, &w);
                let t = t.elapsed();

                println!("     8: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<16, _>(&v, &w);
                let t = t.elapsed();

                println!("    16: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<32, _>(&v, &w);
                let t = t.elapsed();

                println!("    32: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<64, _>(&v, &w);
                let t = t.elapsed();

                println!("    64: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<128, _>(&v, &w);
                let t = t.elapsed();

                println!("   128: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<128, _>(&v, &w);
                let t = t.elapsed();

                println!("   128: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<256, _>(&v, &w);
                let t = t.elapsed();

                println!("   256: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<1024, _>(&v, &w);
                let t = t.elapsed();

                println!("  1024: {d} took {t:?}");
            }

            {
                let t = Instant::now();
                let d = dot_chunked::<8192, _>(&v, &w);
                let t = t.elapsed();

                println!("  8192: {d} took {t:?}");
            }
        }
        "matmul" => {
            use linalg::matrix::Matrix;

            let n = args.next().unwrap().parse().unwrap();

            let t = Instant::now();
            let a = Matrix::new_rand::<1000003, _>(n, n, -1.0..1.0);
            let b = Matrix::new_rand::<1000033, _>(n, n, -1.0..1.0);
            let t = t.elapsed();

            let mut c = Matrix::zeros(n, n);

            println!("Took {t:?} to create rand matrices.");

            let t = Instant::now();
            a.mul_naive(&b, &mut c);
            let t = t.elapsed();

            println!(" Naive: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd::<2>(&b, &mut c);
            let t = t.elapsed();

            println!("     2: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd::<4>(&b, &mut c);
            let t = t.elapsed();

            println!("     4: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd::<8>(&b, &mut c);
            let t = t.elapsed();

            println!("     8: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd::<16>(&b, &mut c);
            let t = t.elapsed();

            println!("    16: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd::<32>(&b, &mut c);
            let t = t.elapsed();

            println!("    32: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd::<64>(&b, &mut c);
            let t = t.elapsed();

            println!("    64: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);
        }
        "matmul-par" => {
            use linalg::matrix::Matrix;

            let n = args.next().unwrap().parse().unwrap();

            set_threads(&mut args);

            let t = Instant::now();
            let a = Matrix::new_rand::<1000003, _>(n, n, -1.0..1.0);
            let b = Matrix::new_rand::<1000033, _>(n, n, -1.0..1.0);
            let t = t.elapsed();

            let mut c = Matrix::zeros(n, n);

            println!("Took {t:?} to create rand matrices.");

            let t = Instant::now();
            a.mul_simd_par::<1>(&b, &mut c);
            let t = t.elapsed();

            println!("     1: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd_par::<2>(&b, &mut c);
            let t = t.elapsed();

            println!("     2: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd_par::<4>(&b, &mut c);
            let t = t.elapsed();

            println!("     4: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd_par::<8>(&b, &mut c);
            let t = t.elapsed();

            println!("     8: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd_par::<16>(&b, &mut c);
            let t = t.elapsed();

            println!("    16: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd_par::<32>(&b, &mut c);
            let t = t.elapsed();

            println!("    32: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);

            let t = Instant::now();
            a.mul_simd_par::<64>(&b, &mut c);
            let t = t.elapsed();

            println!("    64: {:8.4?} took {t:?}", c[(n / 2, n / 3)]);
        }
        "lennard-jones" => {
            use lennard_jones::*;

            let n = args.next().unwrap().parse().unwrap();

            let r = setup_cubic_lattice(n, 1.0);

            let t = Instant::now();
            let e = lennard_jones_naive(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("Naive: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<4, _>(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("    4: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<8, _>(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("    8: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<16, _>(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("   16: {e} \t\t took {t:?}");
        }
        "lennard-jones-par" => {
            use lennard_jones::*;

            let n = args.next().unwrap().parse().unwrap();

            set_threads(&mut args);

            let r = setup_cubic_lattice(n, 1.0);

            let t = Instant::now();
            let e = lennard_jones::<8, _>(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("  Serial: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones_par::<8, _>(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("Parallel: {e} \t\t took {t:?}");
        }
        "lennard-jones-par2" => {
            use lennard_jones::*;

            let n = args.next().unwrap().parse().unwrap();

            set_threads(&mut args);

            let r = setup_cubic_lattice(n, 1.0);

            let t = Instant::now();
            let e = lennard_jones_par::<4, _>(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("  4: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones_par::<8, _>(1.0, 1.0, &r);
            let t = t.elapsed();

            println!("  8: {e} \t\t took {t:?}");
        }
        "lennard-jones-grad" => {
            use lennard_jones::*;

            let n = args.next().unwrap().parse().unwrap();

            let r = setup_cubic_lattice(n, 1.0);
            let mut g = vec![[0.0; 3]; n.pow(3)];

            let t = Instant::now();
            lennard_jones_grad_naive(1.0, 1.0, &mut g, &r);
            let t = t.elapsed();

            println!("Naive: {:?} \t\t took {t:?}", g[0]);

            let t = Instant::now();
            lennard_jones_grad::<2, _>(1.0, 1.0, &mut g, &r);
            let t = t.elapsed();

            println!("    4: {:?} \t\t took {t:?}", g[0]);
        }
        "lennard-jones-T" => {
            use lennard_jones_t::*;

            let n = args.next().unwrap().parse().unwrap();

            let [x, y, z] = setup_cubic_lattice(n, 1.0);

            let t = Instant::now();
            let e = lennard_jones::<1, _>(1.0, 1.0, &x, &y, &z);
            let t = t.elapsed();

            println!("    1: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<2, _>(1.0, 1.0, &x, &y, &z);
            let t = t.elapsed();

            println!("    2: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<4, _>(1.0, 1.0, &x, &y, &z);
            let t = t.elapsed();

            println!("    4: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<8, _>(1.0, 1.0, &x, &y, &z);
            let t = t.elapsed();

            println!("    8: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<16, _>(1.0, 1.0, &x, &y, &z);
            let t = t.elapsed();

            println!("   16: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<32, _>(1.0, 1.0, &x, &y, &z);
            let t = t.elapsed();

            println!("   32: {e} \t\t took {t:?}");

            let t = Instant::now();
            let e = lennard_jones::<64, _>(1.0, 1.0, &x, &y, &z);
            let t = t.elapsed();

            println!("   64: {e} \t\t took {t:?}");
        }
        "lennard-jones-T-grad" => {
            use lennard_jones_t::*;

            let n = args.next().unwrap().parse().unwrap();

            let [x, y, z] = setup_cubic_lattice(n, 1.0);
            let mut gx = vec![0.0; n.pow(3)];
            let mut gy = vec![0.0; n.pow(3)];
            let mut gz = vec![0.0; n.pow(3)];

            let t = Instant::now();
            let e = lennard_jones_grad::<1, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz,
            );
            let t = t.elapsed();

            println!("    1: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad::<2, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz,
            );
            let t = t.elapsed();

            println!("    2: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad::<4, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz,
            );
            let t = t.elapsed();

            println!("    4: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad::<8, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz,
            );
            let t = t.elapsed();

            println!("    8: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad::<16, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz,
            );
            let t = t.elapsed();

            println!("   16: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad::<32, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz,
            );
            let t = t.elapsed();

            println!("   32: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad::<64, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz,
            );
            let t = t.elapsed();

            println!("   64: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);
        }
        "lennard-jones-T-grad-par" => {
            use lennard_jones_t::*;

            let n = args.next().unwrap().parse().unwrap();

            set_threads(&mut args);

            let [x, y, z] = setup_cubic_lattice(n, 1.0);
            let mut gx = vec![0.0; n.pow(3)];
            let mut gy = vec![0.0; n.pow(3)];
            let mut gz = vec![0.0; n.pow(3)];
            let mut buf = Vec::new();

            let t = Instant::now();
            let e = lennard_jones_grad_par::<1, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz, &mut buf,
            );
            let t = t.elapsed();

            println!("    1: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad_par::<2, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz, &mut buf,
            );
            let t = t.elapsed();

            println!("    2: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad_par::<4, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz, &mut buf,
            );
            let t = t.elapsed();

            println!("    4: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad_par::<8, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz, &mut buf,
            );
            let t = t.elapsed();

            println!("    8: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad_par::<16, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz, &mut buf,
            );
            let t = t.elapsed();

            println!("   16: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad_par::<32, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz, &mut buf,
            );
            let t = t.elapsed();

            println!("   32: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);

            let t = Instant::now();
            let e = lennard_jones_grad_par::<64, _>(
                1.0, 1.0, &x, &y, &z, &mut gx, &mut gy, &mut gz, &mut buf,
            );
            let t = t.elapsed();

            println!("   64: {e} \t\t took {t:?}");
            println!("gx: {:8.4?}", &gx[0..4]);
        }
        "transpose-u8-8" => {
            use transpose_u8::{naive, transpose_64x8_u8};
            use rand::thread_rng;
            
            let n_reps: usize = args.next().unwrap().parse().unwrap();

            const M: usize = 8;
            const N: usize = 64;

            let mut a = [[0u8; M]; N];
            let mut b_naive = [[0u8; N]; M];

            let mut rng = thread_rng();
            for r in &mut a {
                for x in r {
                    *x = rng.gen();
                }
            }

            let t = Instant::now();
            for _ in 0..n_reps {
                naive::trans(&a, &mut b_naive);
            }
            let t = t.elapsed();
            println!("Naive took {t:?}");

            let mut b_simd = [[0u8; N]; M];

            let t = Instant::now();
            for _ in 0..n_reps {
                transpose_64x8_u8::trans(&a, &mut b_simd);
            }
            let t = t.elapsed();
            println!(" Simd took {t:?}");

            assert_eq!(b_naive, b_simd);
        }
        "transpose-u8-16" => {
            use transpose_u8::{naive, transpose_64x16_u8};
            use rand::thread_rng;
            
            let n_reps: usize = args.next().unwrap().parse().unwrap();

            const M: usize = 16;
            const N: usize = 64;

            let mut a = [[0u8; M]; N];
            let mut b_naive = [[0u8; N]; M];

            let mut rng = thread_rng();
            for r in &mut a {
                for x in r {
                    *x = rng.gen();
                }
            }

            let t = Instant::now();
            for _ in 0..n_reps {
                naive::trans(&a, &mut b_naive);
            }
            let t = t.elapsed();
            println!("Naive took {t:?}");

            let mut b_simd = [[0u8; N]; M];

            let t = Instant::now();
            for _ in 0..n_reps {
                transpose_64x16_u8::trans(&a, &mut b_simd);
            }
            let t = t.elapsed();
            println!(" Simd took {t:?}");

            assert_eq!(b_naive, b_simd);
        }
        _ => {}
    }
}
