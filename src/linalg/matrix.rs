use super::{primtrait::PrimNum, vector::rand_vec};

use std::{
    fmt::Debug,
    ops::{AddAssign, Index, IndexMut},
};

use rand::distributions::uniform::SampleRange;

pub struct Matrix<T: PrimNum> {
    pub data: Vec<T>,
    pub w: usize,
    pub h: usize,
}

impl<T: PrimNum> Matrix<T> {
    pub fn new_rand<const N: usize, R: SampleRange<T> + Clone>(
        w: usize,
        h: usize,
        r: R,
    ) -> Self {
        Self {
            data: rand_vec::<N, _, _>(w * h, r),
            w,
            h,
        }
    }
}

impl<T: PrimNum + Debug> Debug for Matrix<T> {
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

impl<T: PrimNum> Matrix<T> {
    pub fn zeros(w: usize, h: usize) -> Self {
        let data = vec![T::zero(); w * h];

        Self { data, w, h }
    }
}

impl<T: PrimNum> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i * self.w + j]
    }
}

impl<T: PrimNum> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i * self.w + j]
    }
}

impl<T: PrimNum> Matrix<T> {
    pub fn add_assign_naive(&mut self, rhs: &Self) {
        assert_eq!((self.w, self.h), (rhs.w, rhs.h));

        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a += *b;
        }
    }
}

impl<T: PrimNum> AddAssign<&Self> for Matrix<T> {
    fn add_assign(&mut self, rhs: &Self) {
        self.add_assign_naive(rhs);
    }
}

impl<T: PrimNum> Matrix<T> {
    pub fn mul_naive(&self, rhs: &Self, dest: &mut Self) {
        assert_eq!((self.w, self.h, dest.w), (rhs.h, dest.h, rhs.w));

        dest.data.fill(T::zero());

        for i in 0..dest.h {
            for j in 0..dest.w {
                for k in 0..self.w {
                    dest[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
    }

    pub fn mul_t_naive(&self, rhs: &Self, dest: &mut Self) {
        assert_eq!((self.w, self.h, dest.w), (rhs.w, dest.h, rhs.h));

        dest.data.fill(T::zero());

        for i in 0..dest.h {
            for j in 0..dest.w {
                for k in 0..self.w {
                    dest[(i, j)] += self[(i, k)] * rhs[(j, k)];
                }
            }
        }
    }
}
