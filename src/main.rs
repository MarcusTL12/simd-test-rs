#![feature(slice_as_chunks)]
#![feature(portable_simd)]
#![feature(array_chunks)]

use std::{env::Args, time::Instant};

pub mod linalg;

pub mod colatz;

pub mod lennard_jones;
pub mod lennard_jones_t;

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
        _ => {}
    }
}
