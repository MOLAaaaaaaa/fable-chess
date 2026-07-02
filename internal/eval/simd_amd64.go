//go:build amd64

package eval

// AVX2 kernels implemented in simd_amd64.s. All vector lengths are
// guaranteed to be multiples of 32 (ValidHidden), so the assembly loops
// need no tail handling.

//go:noescape
func detectAVX2() bool

//go:noescape
func avx2Add(dst, add *int16, n int)

//go:noescape
func avx2AddSub(dst, src, add, sub *int16, n int)

//go:noescape
func avx2AddSubSub(dst, src, add, sub1, sub2 *int16, n int)

//go:noescape
func avx2AddAddSubSub(dst, src, add1, add2, sub1, sub2 *int16, n int)

//go:noescape
func avx2Output(us, them, w *int16, n int, out *int32)

// HasAVX2 reports whether the AVX2 fast path is active (exported for
// diagnostics / info strings).
var HasAVX2 = detectAVX2()

func addVec(dst, add []int16) {
	if HasAVX2 {
		avx2Add(&dst[0], &add[0], len(dst))
		return
	}
	addGeneric(dst, add)
}

func addSubVec(dst, src, add, sub []int16) {
	if HasAVX2 {
		avx2AddSub(&dst[0], &src[0], &add[0], &sub[0], len(dst))
		return
	}
	addSubGeneric(dst, src, add, sub)
}

func addSubSubVec(dst, src, add, sub1, sub2 []int16) {
	if HasAVX2 {
		avx2AddSubSub(&dst[0], &src[0], &add[0], &sub1[0], &sub2[0], len(dst))
		return
	}
	addSubSubGeneric(dst, src, add, sub1, sub2)
}

func addAddSubSubVec(dst, src, add1, add2, sub1, sub2 []int16) {
	if HasAVX2 {
		avx2AddAddSubSub(&dst[0], &src[0], &add1[0], &add2[0], &sub1[0], &sub2[0], len(dst))
		return
	}
	addAddSubSubGeneric(dst, src, add1, add2, sub1, sub2)
}

func outputVec(us, them, w []int16) int64 {
	if HasAVX2 {
		var out [8]int32
		avx2Output(&us[0], &them[0], &w[0], len(us), &out[0])
		var sum int64
		for _, v := range out {
			sum += int64(v)
		}
		return sum
	}
	return outputGeneric(us, them, w)
}
