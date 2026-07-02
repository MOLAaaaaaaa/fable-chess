package search

import (
	"sync/atomic"
	"time"

	"fable/internal/chess"
)

// Limits describes the "go" command search constraints.
type Limits struct {
	WTime, BTime int64 // remaining time, ms (0 = not given)
	WInc, BInc   int64 // increments, ms
	MovesToGo    int
	MoveTime     int64 // exact time per move, ms
	Depth        int
	Nodes        int64
	Mate         int // search for mate in N moves
	Infinite     bool
	Ponder       bool
	SearchMoves  []chess.Move
}

// timeManager computes soft (don't start a new iteration) and hard (abort
// mid-search) deadlines. All checks are suppressed while pondering; on
// ponderhit the clock restarts from that moment (restart may be called from
// another goroutine, hence the atomic start time).
type timeManager struct {
	startNanos atomic.Int64
	soft       time.Duration
	hard       time.Duration
	useTime    bool
}

func newTimeManager(lim Limits, side chess.Color, overheadMs int64) *timeManager {
	tm := &timeManager{}
	tm.restart()

	if lim.MoveTime > 0 {
		t := lim.MoveTime - overheadMs
		if t < 1 {
			t = 1
		}
		tm.soft = time.Duration(t) * time.Millisecond
		tm.hard = tm.soft
		tm.useTime = true
		return tm
	}

	myTime, myInc := lim.WTime, lim.WInc
	if side == chess.Black {
		myTime, myInc = lim.BTime, lim.BInc
	}
	if myTime <= 0 {
		return tm // no time control (depth/nodes/infinite)
	}

	mtg := int64(25)
	if lim.MovesToGo > 0 {
		mtg = int64(lim.MovesToGo)
		if mtg > 50 {
			mtg = 50
		}
	}

	avail := myTime - overheadMs
	if avail < 1 {
		avail = 1
	}
	base := avail/mtg + myInc*3/4
	soft := base * 9 / 10
	hard := base * 4
	// Never plan to use more than half the clock in one move (a quarter
	// when many moves remain), and always leave the overhead buffer.
	maxSpend := avail / 2
	if lim.MovesToGo == 0 {
		maxSpend = avail * 4 / 10
	}
	if hard > maxSpend {
		hard = maxSpend
	}
	if soft > hard {
		soft = hard
	}
	if soft < 1 {
		soft = 1
	}
	if hard < 1 {
		hard = 1
	}
	tm.soft = time.Duration(soft) * time.Millisecond
	tm.hard = time.Duration(hard) * time.Millisecond
	tm.useTime = true
	return tm
}

// restart re-arms the deadlines from now (used on ponderhit).
func (tm *timeManager) restart() { tm.startNanos.Store(time.Now().UnixNano()) }

func (tm *timeManager) elapsed() time.Duration {
	return time.Duration(time.Now().UnixNano() - tm.startNanos.Load())
}

// softExpired reports whether a new iteration should not be started.
// stability scales the soft limit: > 1 when the best move keeps changing.
func (tm *timeManager) softExpired(stability float64) bool {
	if !tm.useTime {
		return false
	}
	limit := time.Duration(float64(tm.soft) * stability)
	if limit > tm.hard {
		limit = tm.hard
	}
	return tm.elapsed() >= limit
}

// hardExpired reports whether the search must be aborted immediately.
func (tm *timeManager) hardExpired() bool {
	return tm.useTime && tm.elapsed() >= tm.hard
}
