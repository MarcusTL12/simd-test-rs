use std::{iter::Sum, ops::AddAssign};

use num_traits::Num;
use rand::distributions::uniform::SampleUniform;

pub trait PrimNum: Num + Copy + Sum + AddAssign + SampleUniform {}

impl<T: Num + Copy + Sum + AddAssign + SampleUniform> PrimNum for T {}
