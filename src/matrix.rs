use num_traits::Float;
use rand::Rng;

pub struct Matrix<T: Float> {
    data: Vec<T>,
    w: usize,
    h: usize,
}

impl<T: Float> Matrix<T> {
    pub fn new_rand(w: usize, h: usize) -> Self {
        const N: usize = 1000_003;

        let mut data = vec![T::zero(); w * h];

        let mut rng = rand::thread_rng();

        let rand_buf: Vec<_> = (0..N).map(|_| rng.gen()).collect();

        let (cs, r) = data.as_chunks_mut();

        for c in cs {
            for (y, x) in c.iter_mut().zip(&rand_buf) {
                *y = *x;
            }
        }

        for (y, x) in r.iter_mut().zip(&rand_buf) {
            *y = *x;
        }

        Self { data, w, h }
    }
}
