package eval

// Pure Go reference kernels. These are bit-exact with the AVX2 assembly
// versions (including int16 wraparound semantics) and are used on non-amd64
// platforms, on CPUs without AVX2, and by equivalence tests.

func addGeneric(dst, add []int16) {
	for i := range dst {
		dst[i] += add[i]
	}
}

func addSubGeneric(dst, src, add, sub []int16) {
	for i := range dst {
		dst[i] = src[i] + add[i] - sub[i]
	}
}

func addSubSubGeneric(dst, src, add, sub1, sub2 []int16) {
	for i := range dst {
		dst[i] = src[i] + add[i] - sub1[i] - sub2[i]
	}
}

func addAddSubSubGeneric(dst, src, add1, add2, sub1, sub2 []int16) {
	for i := range dst {
		dst[i] = src[i] + add1[i] + add2[i] - sub1[i] - sub2[i]
	}
}

// outputGeneric computes the SCReLU output layer:
//
//	sum_i clamp(us[i])   * (clamp(us[i])   * w[i])
//	sum_i clamp(them[i]) * (clamp(them[i]) * w[h+i])
//
// where clamp(x) = min(max(x, 0), QA). The inner product (t*w) is truncated
// to int16 exactly like VPMULLW; with |w| <= 127 enforced at load time the
// product 255*127 = 32385 always fits, so no truncation occurs in practice.
func outputGeneric(us, them, w []int16) int64 {
	var sum int64
	h := len(us)
	for i := 0; i < h; i++ {
		t := int32(us[i])
		if t < 0 {
			t = 0
		} else if t > QA {
			t = QA
		}
		q := int16(t * int32(w[i]))
		sum += int64(t) * int64(q)
	}
	for i := 0; i < h; i++ {
		t := int32(them[i])
		if t < 0 {
			t = 0
		} else if t > QA {
			t = QA
		}
		q := int16(t * int32(w[h+i]))
		sum += int64(t) * int64(q)
	}
	return sum
}
