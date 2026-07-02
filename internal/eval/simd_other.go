//go:build !amd64

package eval

// HasAVX2 is always false on non-amd64 platforms.
var HasAVX2 = false

func addVec(dst, add []int16)                        { addGeneric(dst, add) }
func addSubVec(dst, src, add, sub []int16)           { addSubGeneric(dst, src, add, sub) }
func addSubSubVec(dst, src, add, sub1, sub2 []int16) { addSubSubGeneric(dst, src, add, sub1, sub2) }
func addAddSubSubVec(dst, src, add1, add2, sub1, sub2 []int16) {
	addAddSubSubGeneric(dst, src, add1, add2, sub1, sub2)
}
func outputVec(us, them, w []int16) int64 { return outputGeneric(us, them, w) }
