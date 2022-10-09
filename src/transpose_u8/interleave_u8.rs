use std::simd::{
    simd_swizzle, Simd,
    Which::{First, Second},
};

pub const N: usize = 64;

#[inline(always)]
pub fn int4(a: Simd<u8, N>, b: Simd<u8, N>) -> (Simd<u8, N>, Simd<u8, N>) {
    let a0_u32: Simd<u32, 16> = unsafe { std::mem::transmute(a) };
    let b0_u32: Simd<u32, 16> = unsafe { std::mem::transmute(b) };

    let (a1_u32, b1_u32) = a0_u32.interleave(b0_u32);

    (unsafe { std::mem::transmute(a1_u32) }, unsafe {
        std::mem::transmute(b1_u32)
    })
}

#[inline(always)]
pub fn int8(a: Simd<u8, N>, b: Simd<u8, N>) -> (Simd<u8, N>, Simd<u8, N>) {
    let a0_u64: Simd<u64, 8> = unsafe { std::mem::transmute(a) };
    let b0_u64: Simd<u64, 8> = unsafe { std::mem::transmute(b) };

    let (a1_u64, b1_u64) = a0_u64.interleave(b0_u64);

    (unsafe { std::mem::transmute(a1_u64) }, unsafe {
        std::mem::transmute(b1_u64)
    })
}

#[inline(always)]
pub fn int16(a: Simd<u8, N>, b: Simd<u8, N>) -> (Simd<u8, N>, Simd<u8, N>) {
    let a0_u64: Simd<u64, 8> = unsafe { std::mem::transmute(a) };
    let b0_u64: Simd<u64, 8> = unsafe { std::mem::transmute(b) };

    let a1_u64 = simd_swizzle!(
        a0_u64,
        b0_u64,
        [
            First(0),
            First(1),
            Second(0),
            Second(1),
            First(2),
            First(3),
            Second(2),
            Second(3)
        ]
    );

    let b1_u64 = simd_swizzle!(
        a0_u64,
        b0_u64,
        [
            First(4),
            First(5),
            Second(4),
            Second(5),
            First(6),
            First(7),
            Second(6),
            Second(7)
        ]
    );

    (unsafe { std::mem::transmute(a1_u64) }, unsafe {
        std::mem::transmute(b1_u64)
    })
}

#[inline(always)]
pub fn int32(a: Simd<u8, N>, b: Simd<u8, N>) -> (Simd<u8, N>, Simd<u8, N>) {
    let a0_u64: Simd<u64, 8> = unsafe { std::mem::transmute(a) };
    let b0_u64: Simd<u64, 8> = unsafe { std::mem::transmute(b) };

    let a1_u64 = simd_swizzle!(
        a0_u64,
        b0_u64,
        [
            First(0),
            First(1),
            First(2),
            First(3),
            Second(0),
            Second(1),
            Second(2),
            Second(3)
        ]
    );

    let b1_u64 = simd_swizzle!(
        a0_u64,
        b0_u64,
        [
            First(4),
            First(5),
            First(6),
            First(7),
            Second(4),
            Second(5),
            Second(6),
            Second(7)
        ]
    );

    (unsafe { std::mem::transmute(a1_u64) }, unsafe {
        std::mem::transmute(b1_u64)
    })
}