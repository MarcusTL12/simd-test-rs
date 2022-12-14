use std::{
    cell::RefCell,
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
    simd::{LaneCount, Simd, SimdElement, SimdFloat, SupportedLaneCount},
};

use num_traits::Float;
use thread_local::ThreadLocal;

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
        let xj = Simd::splat(*xj);
        let yj = Simd::splat(*yj);
        let zj = Simd::splat(*zj);

        let dx = xj - xi;
        let dy = yj - yi;
        let dz = zj - zi;

        let r2 = dx * dx + dy * dy + dz * dz;
        let s2s = Simd::splat(s2);
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
        + Div<Output = Simd<T, N>>
        + SimdFloat<Scalar = T>,
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
    let s2s = Simd::splat(s2);

    let mut es = Simd::splat(T::zero());
    for (i, ((xc, yc), zc)) in xcs.iter().zip(ycs).zip(zcs).enumerate() {
        let xi = Simd::from(*xc);
        let yi = Simd::from(*yc);
        let zi = Simd::from(*zc);

        for ((xj, yj), zj) in x.iter().zip(y).zip(z).take(N * i) {
            let xj = Simd::splat(*xj);
            let yj = Simd::splat(*yj);
            let zj = Simd::splat(*zj);

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

    let mut e = es.reduce_sum();

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
        + Neg<Output = Simd<T, N>>
        + SimdFloat<Scalar = T>,
{
    let one = T::one();
    let two = T::from(2.0).unwrap();
    let three = T::from(3.0).unwrap();
    let four = T::from(4.0).unwrap();
    let twentyfour = T::from(24.0).unwrap();

    let one_s = Simd::splat(one);
    let two_s = Simd::splat(two);
    let twentyfour_s = Simd::splat(twentyfour);

    let e_b_s = Simd::splat(e_b);

    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());

    assert_eq!(x.len(), gx.len());
    assert_eq!(x.len(), gy.len());
    assert_eq!(x.len(), gz.len());

    gx.fill(T::zero());
    gy.fill(T::zero());
    gz.fill(T::zero());

    let s2 = two.powf(-one / three) * r_eq.powi(2);
    let s2s = Simd::splat(s2);

    let (xcs, xr): (&[[_; N]], _) = x.as_chunks();
    let (ycs, yr): (&[[_; N]], _) = y.as_chunks();
    let (zcs, zr): (&[[_; N]], _) = z.as_chunks();

    let mut e = T::zero();
    let mut es = Simd::splat(T::zero());
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
            let xj = Simd::splat(*xj);
            let yj = Simd::splat(*yj);
            let zj = Simd::splat(*zj);

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

            gx[j] += gxs.reduce_sum();
            gy[j] += gys.reduce_sum();
            gz[j] += gzs.reduce_sum();
        }

        *unsafe { get_const_slice_mut(gx, i) } = *gxi.as_array();
        *unsafe { get_const_slice_mut(gy, i) } = *gyi.as_array();
        *unsafe { get_const_slice_mut(gz, i) } = *gzi.as_array();
    }

    e += es.reduce_sum();

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

pub fn lennard_jones_grad_par<
    'a,
    const N: usize,
    T: Float + SimdElement + Sum + AddAssign + SubAssign + Send + Sync,
>(
    r_eq: T,
    e_b: T,
    x: &[T],
    y: &[T],
    z: &[T],
    gx: &mut [T],
    gy: &mut [T],
    gz: &mut [T],
    buf: &mut Vec<Vec<T>>,
) -> T
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: Add<Output = Simd<T, N>>
        + Sub<Output = Simd<T, N>>
        + Mul<Output = Simd<T, N>>
        + Div<Output = Simd<T, N>>
        + Neg<Output = Simd<T, N>>
        + SimdFloat<Scalar = T>,
{
    use rayon::prelude::*;

    let one = T::one();
    let two = T::from(2.0).unwrap();
    let three = T::from(3.0).unwrap();
    let four = T::from(4.0).unwrap();
    let twentyfour = T::from(24.0).unwrap();

    let one_s = Simd::splat(one);
    let two_s = Simd::splat(two);
    let twentyfour_s = Simd::splat(twentyfour);

    let e_b_s = Simd::splat(e_b);

    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());

    assert_eq!(x.len(), gx.len());
    assert_eq!(x.len(), gy.len());
    assert_eq!(x.len(), gz.len());

    let (buf_s, buf_r) = crossbeam_channel::unbounded();

    for _ in 0..rayon::current_num_threads() * 3 {
        buf_s
            .send(buf.pop().unwrap_or(vec![T::zero(); x.len()]))
            .unwrap();
    }

    gx.fill(T::zero());
    gy.fill(T::zero());
    gz.fill(T::zero());

    let s2 = two.powf(-one / three) * r_eq.powi(2);
    let s2s = Simd::splat(s2);

    let (xcs, xr): (&[[_; N]], _) = x.as_chunks();
    let (ycs, yr): (&[[_; N]], _) = y.as_chunks();
    let (zcs, zr): (&[[_; N]], _) = z.as_chunks();

    let (gxcs, gxr): (&mut [[_; N]], _) = gx.as_chunks_mut();
    let (gycs, gyr): (&mut [[_; N]], _) = gy.as_chunks_mut();
    let (gzcs, gzr): (&mut [[_; N]], _) = gz.as_chunks_mut();

    let parit = xcs
        .into_par_iter()
        .zip_eq(ycs.into_par_iter())
        .zip_eq(zcs.into_par_iter())
        .zip_eq(gxcs.into_par_iter())
        .zip_eq(gycs.into_par_iter())
        .zip_eq(gzcs.into_par_iter())
        .enumerate();

    let tls = ThreadLocal::new();

    parit.for_each_with(
        buf_r,
        |buf_r, (i, (((((xc, yc), zc), gxc), gyc), gzc))| {
            let (mut e, mut gx_buf, mut gy_buf, mut gz_buf) = unsafe {
                tls.get_or(|| {
                    let mut gx = buf_r.recv().unwrap();
                    let mut gy = buf_r.recv().unwrap();
                    let mut gz = buf_r.recv().unwrap();
                    gx.fill(T::zero());
                    gy.fill(T::zero());
                    gz.fill(T::zero());
                    RefCell::new(Some((T::zero(), gx, gy, gz)))
                })
                .take()
                .unwrap_unchecked()
            };

            e += lennard_jones_grad_rest(s2, e_b, xc, yc, zc, gxc, gyc, gzc);

            let mut es = Simd::splat(T::zero());

            let xi = Simd::from(*xc);
            let yi = Simd::from(*yc);
            let zi = Simd::from(*zc);

            let mut gxi = Simd::from(*gxc);
            let mut gyi = Simd::from(*gyc);
            let mut gzi = Simd::from(*gzc);

            for (((((xj, yj), zj), gxj), gyj), gzj) in x
                .iter()
                .zip(y)
                .zip(z)
                .zip(&mut gx_buf)
                .zip(&mut gy_buf)
                .zip(&mut gz_buf)
                .take(N * i)
            {
                let xj = Simd::splat(*xj);
                let yj = Simd::splat(*yj);
                let zj = Simd::splat(*zj);

                let dx = xj - xi;
                let dy = yj - yi;
                let dz = zj - zi;

                let r2 = dx * dx + dy * dy + dz * dz;
                let sr2 = s2s / r2;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;

                es += sr12 - sr6;

                let gs =
                    -twentyfour_s * e_b_s * sr6 / r2 * (two_s * sr6 - one_s);

                let gxs = gs * dx;
                let gys = gs * dy;
                let gzs = gs * dz;

                gxi -= gxs;
                gyi -= gys;
                gzi -= gzs;

                *gxj += gxs.reduce_sum();
                *gyj += gys.reduce_sum();
                *gzj += gzs.reduce_sum();
            }

            e += es.reduce_sum();

            *gxc = *gxi.as_array();
            *gyc = *gyi.as_array();
            *gzc = *gzi.as_array();

            unsafe { tls.get().unwrap_unchecked() }
                .swap(&RefCell::new(Some((e, gx_buf, gy_buf, gz_buf))));
        },
    );

    let mut e = lennard_jones_grad_rest(s2, e_b, xr, yr, zr, gxr, gyr, gzr);

    for (tl_e, tl_gx, tl_gy, tl_gz) in
        tls.into_iter().map(|x| x.into_inner().unwrap())
    {
        e += tl_e;
        for (g, tl_g) in [&mut *gx, gy, gz].into_iter().zip([tl_gx, tl_gy, tl_gz]) {
            for (a, b) in g.iter_mut().zip(&tl_g) {
                *a += *b;
            }
        }
    }

    e * four * e_b
}
