use super::primtrait::PrimNum;

use rand::{distributions::uniform::SampleRange, thread_rng, Rng};

pub fn dot_naive<T: PrimNum>(v: &[T], w: &[T]) -> T {
    v.iter().zip(w).map(|(x, y)| *x * *y).sum()
}

pub fn dot_chunked<const N: usize, T: PrimNum>(v: &[T], w: &[T]) -> T {
    let (vc, vr) = v.as_chunks::<N>();
    let (wc, wr) = w.as_chunks::<N>();

    let mut acc = [T::zero(); N];

    for (xc, yc) in vc.iter().zip(wc) {
        for ((a, x), y) in acc.iter_mut().zip(xc).zip(yc) {
            *a += *x * *y;
        }
    }

    acc.into_iter().sum::<T>()
        + vr.iter().zip(wr).map(|(x, y)| *x * *y).sum::<T>()
}

unsafe fn to_const_slice<'a, const N: usize, T>(v: &'a [T]) -> &'a [T; N] {
    unsafe { v.as_chunks_unchecked().get_unchecked(0) }
}

pub fn rand_vec<const N: usize, T: PrimNum, R: SampleRange<T> + Clone>(
    n: usize,
    r: R,
) -> Vec<T> {
    let mut rng = thread_rng();

    let mut v = vec![T::zero(); n];

    let randbuf: Vec<T> = (0..N).map(|_| rng.gen_range(r.clone())).collect();
    let randbuf: &[T; N] = unsafe { to_const_slice(&randbuf) };

    let (vc, vr) = v.as_chunks_mut();

    for c in vc {
        *c = *randbuf;
    }

    for (x, y) in vr.iter_mut().zip(randbuf) {
        *x = *y;
    }

    v
}
