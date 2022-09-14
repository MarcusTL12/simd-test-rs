use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Sub},
    simd::{LaneCount, Simd, SimdElement, SupportedLaneCount},
};

use num_traits::Float;

pub fn setup_cubic_lattice<T: Float>(n: usize, r: T) -> [Vec<T>; 3] {
    let mut xs = Vec::with_capacity(n.pow(3));
    let mut ys = Vec::with_capacity(n.pow(3));
    let mut zs = Vec::with_capacity(n.pow(3));

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                xs.push(T::from(i).unwrap() * r);
                ys.push(T::from(j).unwrap() * r);
                zs.push(T::from(k).unwrap() * r);
            }
        }
    }

    [xs, ys, zs]
}

fn lennard_jones_chunk<
    const N: usize,
    T: Float + SimdElement + Sum + AddAssign,
>(
    s2: T,
    x: &[T; N],
    y: &[T; N],
    z: &[T; N],
) -> T
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Output = Simd<T, N>>
        + Div<Output = Simd<T, N>>,
{
    let xi = Simd::from(*x);
    let yi = Simd::from(*y);
    let zi = Simd::from(*z);

    let mut e = T::zero();
    for (i, ((xj, yj), zj)) in x.iter().zip(y).zip(z).enumerate() {
        let xj = Simd::from([*xj; N]);
        let yj = Simd::from([*yj; N]);
        let zj = Simd::from([*zj; N]);

        let dx = xj - xi;
        let dy = yj - yi;
        let dz = zj - zi;

        let r2 = dx * dx + dy * dy + dz * dz;
        let s2s = Simd::from([s2; N]);
        let sr2 = s2s / r2;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;

        let es = sr12 - sr6;
        e += es.as_array().iter().cloned().take(i).sum::<T>();
    }
    e
}

fn lennard_jones_rest<T: Float + AddAssign>(
    s2: T,
    x: &[T],
    y: &[T],
    z: &[T],
) -> T {
    let mut e = T::zero();
    for (i, ((xi, yi), zi)) in x.iter().zip(y).zip(z).enumerate() {
        for ((xj, yj), zj) in x.iter().zip(y).zip(z).take(i) {
            let dx = *xj - *xi;
            let dy = *yj - *yi;
            let dz = *zj - *zi;

            let r2 = dx * dx + dy * dy + dz * dz;
            let sr2 = s2 / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;

            e += sr12 - sr6;
        }
    }
    e
}

pub fn lennard_jones<const N: usize, T: Float + SimdElement + Sum + AddAssign>(
    r_eq: T,
    e_b: T,
    x: &[T],
    y: &[T],
    z: &[T],
) -> T
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Output = Simd<T, N>>
        + Div<Output = Simd<T, N>>,
{
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let three = T::from(3.0).unwrap();
    let four = T::from(4.0).unwrap();

    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());

    let (xcs, xr): (&[[_; N]], _) = x.as_chunks();
    let (ycs, yr): (&[[_; N]], _) = y.as_chunks();
    let (zcs, zr): (&[[_; N]], _) = z.as_chunks();

    let s2 = two.powf(-one / three) * r_eq.powi(2);

    let mut es = Simd::from([T::zero(); N]);
    for (i, ((xc, yc), zc)) in xcs.iter().zip(ycs).zip(zcs).enumerate() {
        let xi = Simd::from(*xc);
        let yi = Simd::from(*yc);
        let zi = Simd::from(*zc);

        for ((xj, yj), zj) in x.iter().zip(y).zip(z).take(N * i) {
            let xj = Simd::from([*xj; N]);
            let yj = Simd::from([*yj; N]);
            let zj = Simd::from([*zj; N]);

            let dx = xj - xi;
            let dy = yj - yi;
            let dz = zj - zi;

            let r2 = dx * dx + dy * dy + dz * dz;
            let s2s = Simd::from([s2; N]);
            let sr2 = s2s / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;

            es += sr12 - sr6;
        }
    }

    let mut e: T = es.as_array().iter().cloned().sum();

    for ((xc, yc), zc) in xcs.iter().zip(ycs).zip(zcs) {
        e += lennard_jones_chunk(s2, xc, yc, zc);
    }

    e += lennard_jones_rest(s2, xr, yr, zr);

    e * four * e_b
}
