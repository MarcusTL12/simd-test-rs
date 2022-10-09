use std::simd::{simd_swizzle, Simd};

use super::interleave_u8::*;

pub fn trans(a: &[[u8; 16]; N], b: &mut [[u8; N]; 16]) {
    let a: &[[u8; N]; 16] = unsafe { std::mem::transmute(a) };

    let mut a = a.map(|v| {
        simd_swizzle!(
            Simd::from(v),
            [
                0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4,
                20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24,
                40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28,
                44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
            ]
        )
    });

    for i in 0..8 {
        let (a1, a2) = int4(a[2 * i], a[2 * i + 1]);
        a[2 * i] = a1;
        a[2 * i + 1] = a2;
    }

    for i in 0..4 {
        for j in 0..2 {
            let (a1, a2) = int8(a[4 * i + j], a[4 * i + j + 2]);
            a[4 * i + j] = a1;
            a[4 * i + j + 2] = a2;
        }
    }

    for i in 0..2 {
        for j in 0..4 {
            let (a1, a2) = int16(a[8 * i + j], a[8 * i + j + 4]);
            a[8 * i + j] = a1;
            a[8 * i + j + 4] = a2;
        }
    }

    for i in 0..8 {
        let (a1, a2) = int32(a[i], a[i + 8]);
        a[i] = a1;
        a[i + 8] = a2;
    }

    *b = unsafe {
        std::mem::transmute([
            a[0], a[8], a[4], a[12], a[2], a[10], a[6], a[14], a[1], a[9],
            a[5], a[13], a[3], a[11], a[7], a[15],
        ])
    }
}
