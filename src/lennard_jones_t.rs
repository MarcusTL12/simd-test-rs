use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
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
    let s2s = Simd::from([s2; N]);

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

unsafe fn get_const_slice<'a, const N: usize, T>(
    v: &'a [T],
    i: usize,
) -> &'a [T; N] {
    v.as_chunks_unchecked().get_unchecked(i)
}

unsafe fn get_const_slice_mut<'a, const N: usize, T>(
    v: &'a mut [T],
    i: usize,
) -> &'a mut [T; N] {
    v.as_chunks_unchecked_mut().get_unchecked_mut(i)
}

fn lennard_jones_grad_rest<T: Float + AddAssign + SubAssign>(
    s2: T,
    e_b: T,
    x: &[T],
    y: &[T],
    z: &[T],
    gx: &mut [T],
    gy: &mut [T],
    gz: &mut [T],
) -> T {
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let twentyfour = T::from(24.0).unwrap();

    let mut e = T::zero();
    for (i, ((xi, yi), zi)) in x.iter().zip(y).zip(z).enumerate() {
        let mut gxi = gx[i];
        let mut gyi = gy[i];
        let mut gzi = gz[i];

        for (j, ((xj, yj), zj)) in x.iter().zip(y).zip(z).enumerate().take(i) {
            let dx = *xj - *xi;
            let dy = *yj - *yi;
            let dz = *zj - *zi;

            let r2 = dx * dx + dy * dy + dz * dz;
            let sr2 = s2 / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;

            e += sr12 - sr6;

            let gs = -twentyfour * e_b * sr6 / r2 * (two * sr6 - one);

            let gxs = gs * dx;
            let gys = gs * dy;
            let gzs = gs * dz;

            gxi -= gxs;
            gyi -= gys;
            gzi -= gzs;

            gx[j] += gxs;
            gy[j] += gys;
            gz[j] += gzs;
        }

        gx[i] += gxi;
        gy[i] += gyi;
        gz[i] += gzi;
    }
    e
}

pub fn lennard_jones_grad<
    const N: usize,
    T: Float + SimdElement + Sum + AddAssign + SubAssign,
>(
    r_eq: T,
    e_b: T,
    x: &[T],
    y: &[T],
    z: &[T],
    gx: &mut [T],
    gy: &mut [T],
    gz: &mut [T],
) -> T
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Output = Simd<T, N>>
        + Div<Output = Simd<T, N>>
        + Neg<Output = Simd<T, N>>,
{
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let three = T::from(3.0).unwrap();
    let four = T::from(4.0).unwrap();
    let twentyfour = T::from(24.0).unwrap();

    let one_s = Simd::from([one; N]);
    let two_s = Simd::from([two; N]);
    let twentyfour_s = Simd::from([twentyfour; N]);

    let e_b_s = Simd::from([e_b; N]);

    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());

    assert_eq!(x.len(), gx.len());
    assert_eq!(x.len(), gy.len());
    assert_eq!(x.len(), gz.len());

    gx.fill(T::zero());
    gy.fill(T::zero());
    gz.fill(T::zero());

    let s2 = two.powf(-one / three) * r_eq.powi(2);
    let s2s = Simd::from([s2; N]);

    let (xcs, xr): (&[[_; N]], _) = x.as_chunks();
    let (ycs, yr): (&[[_; N]], _) = y.as_chunks();
    let (zcs, zr): (&[[_; N]], _) = z.as_chunks();

    let mut e = T::zero();
    let mut es = Simd::from([T::zero(); N]);
    for (i, ((xc, yc), zc)) in xcs.iter().zip(ycs).zip(zcs).enumerate() {
        e += lennard_jones_grad_rest(
            s2,
            e_b,
            xc,
            yc,
            zc,
            unsafe { get_const_slice_mut::<N, _>(gx, i) },
            unsafe { get_const_slice_mut::<N, _>(gy, i) },
            unsafe { get_const_slice_mut::<N, _>(gz, i) },
        );

        let xi = Simd::from(*xc);
        let yi = Simd::from(*yc);
        let zi = Simd::from(*zc);

        let mut gxi = Simd::from(*unsafe { get_const_slice(gx, i) });
        let mut gyi = Simd::from(*unsafe { get_const_slice(gy, i) });
        let mut gzi = Simd::from(*unsafe { get_const_slice(gz, i) });

        for (j, ((xj, yj), zj)) in
            x.iter().zip(y).zip(z).enumerate().take(N * i)
        {
            let xj = Simd::from([*xj; N]);
            let yj = Simd::from([*yj; N]);
            let zj = Simd::from([*zj; N]);

            let dx = xj - xi;
            let dy = yj - yi;
            let dz = zj - zi;

            let r2 = dx * dx + dy * dy + dz * dz;
            let sr2 = s2s / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;

            es += sr12 - sr6;

            let gs = -twentyfour_s * e_b_s * sr6 / r2 * (two_s * sr6 - one_s);

            let gxs = gs * dx;
            let gys = gs * dy;
            let gzs = gs * dz;

            gxi -= gxs;
            gyi -= gys;
            gzi -= gzs;

            gx[j] += gxs.as_array().iter().cloned().sum();
            gy[j] += gys.as_array().iter().cloned().sum();
            gz[j] += gzs.as_array().iter().cloned().sum();
        }

        *unsafe { get_const_slice_mut(gx, i) } = *gxi.as_array();
        *unsafe { get_const_slice_mut(gy, i) } = *gyi.as_array();
        *unsafe { get_const_slice_mut(gz, i) } = *gzi.as_array();
    }

    e += es.as_array().iter().cloned().sum();

    e += lennard_jones_grad_rest(
        s2,
        e_b,
        xr,
        yr,
        zr,
        gx.as_chunks_mut::<N>().1,
        gy.as_chunks_mut::<N>().1,
        gz.as_chunks_mut::<N>().1,
    );

    e * four * e_b
}
