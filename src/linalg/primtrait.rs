use std::{iter::Sum, ops::AddAssign, simd::SimdElement};

use num_traits::Num;
use rand::distributions::uniform::SampleUniform;

pub trait PrimNum:
    Num + Copy + Sum + AddAssign + SampleUniform + SimdElement
{
}

impl<T: Num + Copy + Sum + AddAssign + SampleUniform + SimdElement> PrimNum
    for T
{
}
