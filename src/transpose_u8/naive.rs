pub fn trans<const M: usize, const N: usize>(
    a: &[[u8; M]; N],
    b: &mut [[u8; N]; M],
) {
    for i in 0..N {
        for j in 0..M {
            b[j][i] = a[i][j];
        }
    }
}
