use std::{fmt::Debug, ops::AddAssign};

use num_traits::Float;
use rand::{distributions::Standard, prelude::Distribution, Rng};

pub struct Matrix<T: Float> {
    pub data: Vec<T>,
    pub w: usize,
    pub h: usize,
}

impl<T: Float> Matrix<T>
where
    Standard: Distribution<T>,
{
    pub fn new_rand(w: usize, h: usize) -> Self {
        const N: usize = 1000_003;

        let mut data = vec![T::zero(); w * h];

        let mut rng = rand::thread_rng();

        let half = T::from(0.5).unwrap();

        let rand_buf: Vec<_> = (0..N).map(|_| rng.gen() - half).collect();

        let (cs, r) = data.as_chunks_mut::<N>();

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

impl<T: Float + Debug> Debug for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.data.chunks_exact(self.w) {
            write!(f, "[")?;
            let mut row = row.iter();
            write!(f, "{:>9.4?}", row.next().ok_or(std::fmt::Error)?)?;
            for x in row {
                write!(f, " {:>9.4?}", x)?;
            }
            writeln!(f, "]")?;
        }

        Ok(())
    }
}

impl<T: Float> Matrix<T> {
    pub fn zeros(w: usize, h: usize) -> Self {
        let data = vec![T::zero(); w * h];

        Self { data, w, h }
    }
}

impl<T: Float + AddAssign> Matrix<T> {
    pub fn add_assign_naive(&mut self, rhs: &Self) {
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a += *b;
        }
    }
}

impl<T: Float + AddAssign> AddAssign<&Self> for Matrix<T> {
    fn add_assign(&mut self, rhs: &Self) {
        self.add_assign_naive(rhs);
    }
}
