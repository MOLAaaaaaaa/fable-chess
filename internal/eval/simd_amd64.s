//go:build amd64

#include "textflag.h"

// AVX2 NNUE kernels. Vector length n is always a multiple of 32 int16
// elements (= 64 bytes = 2 YMM registers per iteration).

// func detectAVX2() bool
// Checks CPUID leaf 7 EBX bit 5 (AVX2) plus OSXSAVE/AVX and OS YMM state
// support via XGETBV.
TEXT ·detectAVX2(SB), NOSPLIT, $0-1
	// Max CPUID leaf must be >= 7.
	MOVL $0, AX
	CPUID
	CMPL AX, $7
	JL   noavx2

	// Leaf 1 ECX: OSXSAVE (bit 27) and AVX (bit 28).
	MOVL $1, AX
	MOVL $0, CX
	CPUID
	MOVL CX, R8
	ANDL $0x18000000, R8
	CMPL R8, $0x18000000
	JNE  noavx2

	// XGETBV(0): XCR0 bits 1 (XMM) and 2 (YMM) must be set by the OS.
	MOVL $0, CX
	BYTE $0x0F; BYTE $0x01; BYTE $0xD0 // XGETBV -> EDX:EAX
	ANDL $6, AX
	CMPL AX, $6
	JNE  noavx2

	// Leaf 7 subleaf 0 EBX bit 5: AVX2.
	MOVL $7, AX
	MOVL $0, CX
	CPUID
	ANDL $0x20, BX
	JZ   noavx2

	MOVB $1, ret+0(FP)
	RET

noavx2:
	MOVB $0, ret+0(FP)
	RET

// func avx2Add(dst, add *int16, n int)
// dst += add
TEXT ·avx2Add(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ add+8(FP), R8
	MOVQ n+16(FP), CX
	SHRQ $5, CX

addloop:
	VMOVDQU (DI), Y0
	VMOVDQU 32(DI), Y1
	VPADDW  (R8), Y0, Y0
	VPADDW  32(R8), Y1, Y1
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	ADDQ    $64, DI
	ADDQ    $64, R8
	DECQ    CX
	JNZ     addloop
	VZEROUPPER
	RET

// func avx2AddSub(dst, src, add, sub *int16, n int)
// dst = src + add - sub
TEXT ·avx2AddSub(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ add+16(FP), R8
	MOVQ sub+24(FP), R9
	MOVQ n+32(FP), CX
	SHRQ $5, CX

asloop:
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VPADDW  (R8), Y0, Y0
	VPADDW  32(R8), Y1, Y1
	VPSUBW  (R9), Y0, Y0
	VPSUBW  32(R9), Y1, Y1
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	ADDQ    $64, SI
	ADDQ    $64, DI
	ADDQ    $64, R8
	ADDQ    $64, R9
	DECQ    CX
	JNZ     asloop
	VZEROUPPER
	RET

// func avx2AddSubSub(dst, src, add, sub1, sub2 *int16, n int)
// dst = src + add - sub1 - sub2
TEXT ·avx2AddSubSub(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ add+16(FP), R8
	MOVQ sub1+24(FP), R9
	MOVQ sub2+32(FP), R10
	MOVQ n+40(FP), CX
	SHRQ $5, CX

assloop:
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VPADDW  (R8), Y0, Y0
	VPADDW  32(R8), Y1, Y1
	VPSUBW  (R9), Y0, Y0
	VPSUBW  32(R9), Y1, Y1
	VPSUBW  (R10), Y0, Y0
	VPSUBW  32(R10), Y1, Y1
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	ADDQ    $64, SI
	ADDQ    $64, DI
	ADDQ    $64, R8
	ADDQ    $64, R9
	ADDQ    $64, R10
	DECQ    CX
	JNZ     assloop
	VZEROUPPER
	RET

// func avx2AddAddSubSub(dst, src, add1, add2, sub1, sub2 *int16, n int)
// dst = src + add1 + add2 - sub1 - sub2
TEXT ·avx2AddAddSubSub(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ add1+16(FP), R8
	MOVQ add2+24(FP), R9
	MOVQ sub1+32(FP), R10
	MOVQ sub2+40(FP), R11
	MOVQ n+48(FP), CX
	SHRQ $5, CX

aassloop:
	VMOVDQU (SI), Y0
	VMOVDQU 32(SI), Y1
	VPADDW  (R8), Y0, Y0
	VPADDW  32(R8), Y1, Y1
	VPADDW  (R9), Y0, Y0
	VPADDW  32(R9), Y1, Y1
	VPSUBW  (R10), Y0, Y0
	VPSUBW  32(R10), Y1, Y1
	VPSUBW  (R11), Y0, Y0
	VPSUBW  32(R11), Y1, Y1
	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	ADDQ    $64, SI
	ADDQ    $64, DI
	ADDQ    $64, R8
	ADDQ    $64, R9
	ADDQ    $64, R10
	ADDQ    $64, R11
	DECQ    CX
	JNZ     aassloop
	VZEROUPPER
	RET

// SCReLU clamp upper bound QA=255 replicated over 16 int16 lanes.
DATA qaConst<>+0x00(SB)/8, $0x00FF00FF00FF00FF
DATA qaConst<>+0x08(SB)/8, $0x00FF00FF00FF00FF
DATA qaConst<>+0x10(SB)/8, $0x00FF00FF00FF00FF
DATA qaConst<>+0x18(SB)/8, $0x00FF00FF00FF00FF
GLOBL qaConst<>(SB), RODATA, $32

// func avx2Output(us, them, w *int16, n int, out *int32)
// SCReLU output layer. For each element: t = clamp(x, 0, QA);
// accumulate t*(t*w) into 8 int32 lanes written to out. The caller sums the
// lanes in int64. Lane accumulation cannot overflow int32 because
// |w| <= 127 is enforced at network load (see analysis in kernels_generic.go).
TEXT ·avx2Output(SB), NOSPLIT, $0-40
	MOVQ us+0(FP), SI
	MOVQ them+8(FP), DX
	MOVQ w+16(FP), R8
	MOVQ n+24(FP), CX
	MOVQ out+32(FP), DI

	VPXOR   Y9, Y9, Y9          // accumulator
	VPXOR   Y7, Y7, Y7          // zeros for relu floor
	VMOVDQU qaConst<>(SB), Y8   // 255 x16

	MOVQ CX, R9
	SHRQ $4, R9                 // n/16 iterations for "us"

outloop1:
	VMOVDQU  (SI), Y0
	VPMINSW  Y8, Y0, Y0
	VPMAXSW  Y7, Y0, Y0         // t = clamp(x, 0, 255)
	VMOVDQU  (R8), Y1           // w
	VPMULLW  Y0, Y1, Y1         // q = t*w (low 16 bits)
	VPMADDWD Y1, Y0, Y1         // pairwise t*q -> int32
	VPADDD   Y1, Y9, Y9
	ADDQ     $32, SI
	ADDQ     $32, R8
	DECQ     R9
	JNZ      outloop1

	MOVQ CX, R9
	SHRQ $4, R9                 // n/16 iterations for "them"

outloop2:
	VMOVDQU  (DX), Y0
	VPMINSW  Y8, Y0, Y0
	VPMAXSW  Y7, Y0, Y0
	VMOVDQU  (R8), Y1
	VPMULLW  Y0, Y1, Y1
	VPMADDWD Y1, Y0, Y1
	VPADDD   Y1, Y9, Y9
	ADDQ     $32, DX
	ADDQ     $32, R8
	DECQ     R9
	JNZ      outloop2

	VMOVDQU Y9, (DI)
	VZEROUPPER
	RET
