use std::{cell::RefCell, iter::Sum, ops::AddAssign};

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
