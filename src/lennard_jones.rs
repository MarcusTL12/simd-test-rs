use std::{
    cell::RefCell,
    iter::Sum,
    ops::{AddAssign, SubAssign},
};

use num_traits::Float;
use thread_local::ThreadLocal;

pub fn setup_cubic_lattice<T: Float + Sum + AddAssign>(
    n: usize,
    r: T,
) -> Vec<[T; 3]> {
    let mut rs = Vec::new();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                rs.push([
                    T::from(i).unwrap() * r,
                    T::from(j).unwrap() * r,
                    T::from(k).unwrap() * r,
                ]);
            }
        }
    }

    rs
}

pub fn lennard_jones_naive<T: Float + Sum + AddAssign>(
    r_eq: T,
    e_b: T,
    r: &[[T; 3]],
) -> T {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;

    let s2 = two.powf(-one / three) * r_eq.powi(2);

    let mut e = T::zero();
    for (i, ri) in r.iter().enumerate() {
        for rj in r.iter().take(i) {
            let r2: T = ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
            let sr2 = s2 / r2;
            let sr6 = sr2.powi(3);
            let sr12 = sr6.powi(2);

            e += sr12 - sr6;
        }
    }

    e * four * e_b
}

pub fn lennard_jones<const N: usize, T: Float + Sum + AddAssign>(
    r_eq: T,
    e_b: T,
    r: &[[T; 3]],
) -> T {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;

    let s2 = two.powf(-one / three) * r_eq.powi(2);

    let mut es = [T::zero(); N];
    for (i, ri) in r.iter().enumerate() {
        let (rcs, rr): (&[[_; N]], _) = r[0..i].as_chunks();

        for rc in rcs {
            for (j, rj) in rc.iter().enumerate() {
                let r2: T =
                    ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
                let sr2 = s2 / r2;
                let sr6 = sr2.powi(3);
                let sr12 = sr6.powi(2);
                es[j] += sr12 - sr6;
            }
        }

        for (j, rj) in rr.iter().enumerate() {
            let r2: T = ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
            let sr2 = s2 / r2;
            let sr6 = sr2.powi(3);
            let sr12 = sr6.powi(2);

            es[j] += sr12 - sr6;
        }
    }

    es.into_iter().sum::<T>() * four * e_b
}

pub fn lennard_jones_par<
    const N: usize,
    T: Float + Sum + AddAssign + Send + Sync,
>(
    r_eq: T,
    e_b: T,
    r: &[[T; 3]],
) -> T {
    use rayon::prelude::*;

    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;

    let s2 = two.powf(-one / three) * r_eq.powi(2);

    let tl_es = ThreadLocal::new();
    r.into_par_iter().enumerate().for_each(|(i, ri)| {
        let mut es = unsafe {
            tl_es
                .get_or(|| RefCell::new(Some([T::zero(); N])))
                .take()
                .unwrap_unchecked()
        };

        let (rcs, rr): (&[[_; N]], _) = r[0..i].as_chunks();

        for rc in rcs {
            for (j, rj) in rc.iter().enumerate() {
                let r2: T =
                    ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
                let sr2 = s2 / r2;
                let sr6 = sr2.powi(3);
                let sr12 = sr6.powi(2);
                es[j] += sr12 - sr6;
            }
        }

        for (j, rj) in rr.iter().enumerate() {
            let r2: T = ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
            let sr2 = s2 / r2;
            let sr6 = sr2.powi(3);
            let sr12 = sr6.powi(2);

            es[j] += sr12 - sr6;
        }

        unsafe { tl_es.get().unwrap_unchecked() }.swap(&RefCell::new(Some(es)));
    });

    tl_es
        .into_iter()
        .flat_map(|es| unsafe { es.into_inner().unwrap_unchecked() })
        .sum::<T>()
        * four
        * e_b
}

pub fn lennard_jones_grad_naive<T: Float + Sum + AddAssign + SubAssign>(
    r_eq: T,
    e_b: T,
    g: &mut [[T; 3]],
    r: &[[T; 3]],
) {
    assert_eq!(r.len(), g.len());

    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let twentyfour = two * three * four;

    let s2 = two.powf(-one / three) * r_eq.powi(2);

    for gc in g.iter_mut() {
        *gc = [T::zero(); 3];
    }

    for (i, ri) in r.iter().enumerate() {
        for (j, rj) in r.iter().enumerate().take(i) {
            let r2: T = ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
            let a = s2 / r2;
            let a3 = a.powi(3);
            let s = -twentyfour * e_b * a3 / r2 * (two * a3 - one);
            for q in 0..3 {
                let gq = (rj[q] - ri[q]) * s;
                g[i][q] -= gq;
                g[j][q] += gq;
            }
        }
    }
}

pub fn lennard_jones_grad<
    const N: usize,
    T: Float + Sum + AddAssign + SubAssign,
>(
    r_eq: T,
    e_b: T,
    g: &mut [[T; 3]],
    r: &[[T; 3]],
) {
    assert_eq!(r.len(), g.len());

    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let four = two + two;
    let twentyfour = two * three * four;

    let s2 = two.powf(-one / three) * r_eq.powi(2);

    for gc in g.iter_mut() {
        *gc = [zero; 3];
    }

    let mut bufs = [[zero; 3]; N];

    for (i, ri) in r.iter().enumerate() {
        let (rcs, rr): (&[[_; N]], _) = r[0..i].as_chunks();

        let mut bufs2 = [[zero; 3]; N];

        for (c, rc) in rcs.iter().enumerate() {
            for (rj, buf) in rc.iter().zip(&mut bufs) {
                let r2: T =
                    ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
                let a = s2 / r2;
                let a3 = a.powi(3);
                let s = -twentyfour * e_b * a3 / r2 * (two * a3 - one);
                for (b, (&ra, &rb)) in buf.iter_mut().zip(ri.iter().zip(rj)) {
                    *b = (rb - ra) * s;
                }
            }

            for (gc, b) in g[c * N..(c + 1) * N].iter_mut().zip(bufs) {
                for (x, y) in gc.iter_mut().zip(b) {
                    *x += y;
                }
            }

            for (b2, b) in bufs2.iter_mut().zip(bufs) {
                for (x, y) in b2.iter_mut().zip(b) {
                    *x -= y;
                }
            }
        }

        for b in bufs2 {
            for (x, y) in g[i].iter_mut().zip(b) {
                *x += y;
            }
        }

        // for (rj, buf) in rr.iter().zip(&mut bufs) {
        //     let r2: T = ri.iter().zip(rj).map(|(x, y)| (*x - *y).powi(2)).sum();
        //     let a = s2 / r2;
        //     let a3 = a.powi(3);
        //     let s = -twentyfour * e_b * a3 / r2 * (two * a3 - one);
        //     for (b, (&ra, &rb)) in buf.iter_mut().zip(ri.iter().zip(rj)) {
        //         *b = (rb - ra) * s;
        //     }
        // }
    }
}
