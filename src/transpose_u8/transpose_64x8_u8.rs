use std::simd::{simd_swizzle, Simd};

use super::interleave_u8::*;

pub fn trans(a: &[[u8; 8]; N], b: &mut [[u8; N]; 8]) {
    let a: &[[u8; N]; 8] = unsafe { std::mem::transmute(a) };

    let mut a = a.map(|v| {
        simd_swizzle!(
            Simd::from(v),
            [
                0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2,
                10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4,
                12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6,
                14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
            ]
        )
    });

    for i in 0..4 {
        let (a1, a2) = int8(a[2 * i], a[2 * i + 1]);
        a[2 * i] = a1;
        a[2 * i + 1] = a2;
    }

    for i in 0..2 {
        for j in 0..2 {
            let (a1, a2) = int16(a[4 * i + j], a[4 * i + j + 2]);
            a[4 * i + j] = a1;
            a[4 * i + j + 2] = a2;
        }
    }

    for i in 0..4 {
        let (a1, a2) = int32(a[i], a[i + 4]);
        a[i] = a1;
        a[i + 4] = a2;
    }

    *b = unsafe {
        std::mem::transmute([a[0], a[4], a[2], a[6], a[1], a[5], a[3], a[7]])
    };
}
