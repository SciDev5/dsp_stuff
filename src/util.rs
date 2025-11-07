pub struct HeaplessContinuousPushBuffer<T, const N: usize> {
    i: usize,
    buffer: [T; N],
}
impl<T: Copy, const N: usize> HeaplessContinuousPushBuffer<T, N> {
    pub fn new(init: T) -> Self {
        Self {
            i: 0,
            buffer: [init; N],
        }
    }
    pub fn push(&mut self, v: T) {
        let i = (self.i + 1) % (N / 2);
        self.i = i;
        self.buffer[i] = v;
        self.buffer[i + N / 2] = v;
    }
    #[inline]
    pub fn slice(&self) -> &[T] {
        let i = self.i;
        &self.buffer[i..i + N / 2]
    }
}
