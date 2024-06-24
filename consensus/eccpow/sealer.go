// Copyright 2017 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package eccpow

/*
#cgo LDFLAGS: -L. -lmineseoulcuda -lcudart
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint8_t MixDigest[32];
    uint8_t Nonce[8];
    uint8_t Codeword[256];
	bool FinishFlag;
	uint64_t Count;
} Header_kernel;

extern void mineSeoulCuda(int number, int gpu_num, uint8_t* hash, uint64_t seed, int n, int m, int wc, int wr, int* rowInCol, int* colInRow, Header_kernel* result, bool* abort);

int getNumDevices() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}
*/
import "C"
import (
	"bytes"
	"context"
	crand "crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"net/http"
	"runtime"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"github.com/cryptoecc/WorldLand/common"
	"github.com/cryptoecc/WorldLand/common/hexutil"
	"github.com/cryptoecc/WorldLand/consensus"
	"github.com/cryptoecc/WorldLand/core/types"
	"github.com/cryptoecc/WorldLand/crypto"
	"github.com/cryptoecc/WorldLand/log"
)

type HeaderKernel struct {
	MixDigest  [32]uint8
	Nonce      [8]uint8
	Codeword   [256]uint8
	FinishFlag bool
	Count      uint64
}

const (
	// staleThreshold is the maximum depth of the acceptable stale but valid ecc solution.
	staleThreshold = 7
)

var (
	errNoMiningWork      = errors.New("no mining work available yet")
	errInvalidSealResult = errors.New("invalid or stale proof-of-work solution")
)

func convertToUnicodeString(nString string) string {
	var result string

	for _, digit := range nString {
		num, err := strconv.Atoi(string(digit))

		if err != nil {
			return ""
		}

		unicodeChar := 0x2002 + num
		result += string(rune(unicodeChar))
	}

	return result
}

// Seal implements consensus.Engine, attempting to find a nonce that satisfies
// the block's difficulty requirements.
func (ecc *ECC) Seal(chain consensus.ChainHeaderReader, block *types.Block, results chan<- *types.Block, stop <-chan struct{}) error {
	// If we're running a fake PoW, simply return a 0 nonce immediately
	if ecc.config.PowMode == ModeFake || ecc.config.PowMode == ModeFullFake {
		header := block.Header()
		header.Nonce, header.MixDigest = types.BlockNonce{}, common.Hash{}
		select {
		case results <- block.WithSeal(header):
		default:
			log.Warn("Sealing result is not read by miner", "mode", "fake", "sealhash", ecc.SealHash(block.Header()))
		}
		return nil
	}
	// If we're running a shared PoW, delegate sealing to it
	if ecc.shared != nil {
		return ecc.shared.Seal(chain, block, results, stop)
	}
	abort := C.bool(false)

	ecc.lock.Lock()
	threads := ecc.threads
	if ecc.rand == nil {
		seed, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
		if err != nil {
			ecc.lock.Unlock()
			return err
		}
		ecc.rand = rand.New(rand.NewSource(seed.Int64()))
	}
	ecc.lock.Unlock()
	if threads == 0 {
		threads = runtime.NumCPU()
	}
	if threads < 0 {
		threads = 0 // Allows disabling local mining without extra logic around local/remote
	}
	// Push new work to remote sealer
	if ecc.remote != nil {
		ecc.remote.workCh <- &sealTask{block: block, results: results}
	}
	var (
		pend  sync.WaitGroup
		found = make(chan HeaderKernel)
	)

	fmt.Printf(" %s \n", convertToUnicodeString(strconv.Itoa(int(ecc.Hashrate()))))
	//fmt.Printf("@@@@@@@ %v @@@@@@@@\n", int(ecc.Hashrate()))

	numDevices := int(C.getNumDevices())
	if numDevices <= 0 {
		fmt.Println("No CUDA-capable devices found.")
	}

	header := types.CopyHeader(block.Header())
	parameters, _ := setParameters_Seoul(header)
	H := generateH(parameters)
	colInRow, rowInCol := generateQ(parameters, H)
	hash := ecc.SealHash(header).Bytes()

	flatColInRow := make([]int32, parameters.wr*parameters.m)
	for i := 0; i < parameters.wr; i++ {
		for j := 0; j < parameters.m; j++ {
			flatColInRow[i*parameters.m+j] = (int32)(colInRow[i][j])
		}
	}

	flatRowInCol := make([]int32, parameters.wc*parameters.n)
	for i := 0; i < parameters.wc; i++ {
		for j := 0; j < parameters.n; j++ {
			flatRowInCol[i*parameters.n+j] = (int32)(rowInCol[i][j])
		}
	}

	c_colInRow := (*C.int)(unsafe.Pointer(&flatColInRow[0]))
	c_rowInCol := (*C.int)(unsafe.Pointer(&flatRowInCol[0]))

	for gpu_num := 0; gpu_num < numDevices; gpu_num++ {
		pend.Add(1)

		go func(gpu_num int, seed uint64) {
			defer pend.Done()

			var foundHeader HeaderKernel
			foundHeader.FinishFlag = false

			C.mineSeoulCuda((C.int)((header.Number).Int64()), (C.int)(gpu_num), (*C.uint8_t)(unsafe.Pointer(&hash[0])), (C.uint64_t)(seed), (C.int)(parameters.n), (C.int)(parameters.m), (C.int)(parameters.wc), (C.int)(parameters.wr), c_rowInCol, c_colInRow, (*C.Header_kernel)(unsafe.Pointer(&foundHeader)), (*C.bool)(unsafe.Pointer(&abort)))
			if foundHeader.FinishFlag {
				found <- foundHeader
			}

			ecc.hashrate.Mark(int64(foundHeader.Count)) //need to fix

			/*if nonce%20 == 0 {
				fmt.Printf("@@ hash: %v\n", hash)
				fmt.Printf("@@ nonce: %v\n", nonce)
				fmt.Printf("@@ n m wc wr: %v %v %v %v\n", parameters.n, parameters.m, parameters.wc, parameters.wr)
				fmt.Printf("@@ colInRow: %v\n", colInRow)
				fmt.Printf("@@ rowInCol: %v\n", rowInCol)
			}*/
		}(gpu_num, uint64(ecc.rand.Int63()))
	}

	// Wait until sealing is terminated or a nonce is found
	go func() {
		var result *types.Block
		select {
		case <-stop:
			// Outside abort, stop all miner threads
			abort = C.bool(true)
		case gpu_header := <-found:
			header.CodeLength = uint64(parameters.n)
			header.MixDigest = gpu_header.MixDigest
			header.Nonce = gpu_header.Nonce
			header.Codeword = gpu_header.Codeword[0:int((parameters.n+7)/8)]

			result = block.WithSeal(header)
			// One of the threads found a block, abort all others
			select {
			case results <- result:
			default:
				ecc.config.Log.Warn("Sealing result is not read by miner", "mode", "local", "sealhash", ecc.SealHash(block.Header()))
			}
			abort = C.bool(true)
		case <-ecc.update:
			// Thread count was changed on user request, restart
			abort = C.bool(true)
			if err := ecc.Seal(chain, block, results, stop); err != nil {
				ecc.config.Log.Error("Failed to restart sealing after update", "err", err)
			}
		}
		pend.Wait()
	}()

	return nil
}

// mine is the actual proof-of-work miner that searches for a nonce starting from
// seed that results in correct final block difficulty.
func (ecc *ECC) mine(block *types.Block, id int, seed uint64, abort chan struct{}, found chan *types.Block) {
	// Extract some data from the header
	var (
		header = block.Header()
		hash   = ecc.SealHash(header).Bytes()
	)
	// Start generating random nonces until we abort or find a good one
	var (
		total_attempts = int64(0)
		attempts       = int64(0)
		nonce          = seed
	)
	logger := log.New("miner", id)
	logger.Trace("Started ecc search for new nonces", "seed", seed)
search:
	for {
		select {
		case <-abort:
			// Mining terminated, update stats and abort
			logger.Trace("ecc nonce search aborted", "attempts", nonce-seed)
			ecc.hashrate.Mark(attempts)
			break search

		default:
			// We don't have to update hash rate on every nonce, so update after after 2^X nonces
			total_attempts = total_attempts + 64
			attempts = attempts + 64
			if (attempts % (1 << 15)) == 0 {
				ecc.hashrate.Mark(attempts)
				attempts = 0
			}
			// Compute the PoW value of this nonce

			flag, _, outputWord, LDPCNonce, digest := RunOptimizedConcurrencyLDPC(header, hash)

			// Correct nonce found, create a new header with it
			if flag == true {
				//level := SearchLevel_Seoul(header.Difficulty)
				//fmt.Printf("level: %v\n", level)
				//fmt.Printf("total attempts: %v\n", total_attempts)
				//fmt.Printf("hashrate: %v\n", ecc.Hashrate())
				//fmt.Printf("Codeword found with nonce = %d\n", LDPCNonce)
				//fmt.Printf("Codeword : %d\n", outputWord)

				header = types.CopyHeader(header)
				header.MixDigest = common.BytesToHash(digest)
				header.Nonce = types.EncodeNonce(LDPCNonce)

				//convert codeword
				var codeword []byte
				var codeVal byte
				for i, v := range outputWord {
					codeVal |= byte(v) << (7 - i%8)
					if i%8 == 7 {
						codeword = append(codeword, codeVal)
						codeVal = 0
					}
				}
				if len(outputWord)%8 != 0 {
					codeword = append(codeword, codeVal)
				}
				header.Codeword = make([]byte, len(codeword))
				copy(header.Codeword, codeword)
				//fmt.Printf("header: %v\n", header)
				//fmt.Printf("header Codeword : %v\n", header.Codeword)

				// Seal and return a block (if still needed)
				select {
				case found <- block.WithSeal(header):
					logger.Trace("ecc nonce found and reported", "LDPCNonce", LDPCNonce)
				case <-abort:
					logger.Trace("ecc nonce found but discarded", "LDPCNonce", LDPCNonce)
				}
				break search
			}
		}
	}
}

func (ecc *ECC) mine_seoul(block *types.Block, id int, seed uint64, abort chan struct{}, found chan *types.Block) {
	// Extract some data from the header
	var (
		header = block.Header()
		hash   = ecc.SealHash(header).Bytes()
	)
	// Start generating random nonces until we abort or find a good one
	var (
		total_attempts = int64(0)
		attempts       = int64(0)
		nonce          = seed
	)
	logger := log.New("miner", id)
	logger.Trace("Started ecc search for new nonces", "seed", seed)

	parameters, _ := setParameters_Seoul(header)
	H := generateH(parameters)
	colInRow, rowInCol := generateQ(parameters, H)

search:
	for {
		select {
		case <-abort:
			// Mining terminated, update stats and abort
			logger.Trace("ecc nonce search aborted", "attempts", nonce-seed)
			ecc.hashrate.Mark(attempts)
			break search

		default:
			// We don't have to update hash rate on every nonce, so update after after 2^X nonces
			total_attempts = total_attempts + 1
			attempts = attempts + 1
			if (attempts % (1 << 15)) == 0 {
				ecc.hashrate.Mark(attempts)
				attempts = 0
			}

			digest := make([]byte, 40)
			copy(digest, hash)
			binary.LittleEndian.PutUint64(digest[32:], nonce)
			digest = crypto.Keccak512(digest)

			goRoutineHashVector := generateHv(parameters, digest)
			goRoutineHashVector, goRoutineOutputWord, _ := OptimizedDecodingSeoul(parameters, goRoutineHashVector, H, rowInCol, colInRow)

			flag, _ := MakeDecision_Seoul(header, colInRow, goRoutineOutputWord)

			if flag == true {
				outputWord := goRoutineOutputWord

				header = types.CopyHeader(header)
				header.CodeLength = uint64(parameters.n)
				header.MixDigest = common.BytesToHash(digest)
				header.Nonce = types.EncodeNonce(nonce)

				//convert codeword
				var codeword []byte
				var codeVal byte
				for i, v := range outputWord {
					codeVal |= byte(v) << (7 - i%8)
					if i%8 == 7 {
						codeword = append(codeword, codeVal)
						codeVal = 0
					}
				}
				if len(outputWord)%8 != 0 {
					codeword = append(codeword, codeVal)
				}
				header.Codeword = make([]byte, len(codeword))
				copy(header.Codeword, codeword)

				// Seal and return a block (if still needed)
				select {
				case found <- block.WithSeal(header):
					logger.Trace("ecc nonce found and reported", "LDPCNonce", nonce)
				case <-abort:
					logger.Trace("ecc nonce found but discarded", "LDPCNonce", nonce)
				}
				break search
			}
			nonce++
		}
	}
}

func (ecc *ECC) mine_seoul_gpu(header types.Header, hash []byte, seed uint64, abort chan struct{}, found chan *types.Header, param_n int, param_m int, param_wc int, param_wr int, colInRow [][]int, rowInCol [][]int) {
	var nonce = seed
	var attempts = int64(0)

search:
	for {
		select {
		case <-abort:
			ecc.hashrate.Mark(attempts)
			break search

		default:
			attempts = attempts + 1
			if (attempts % (1 << 15)) == 0 {
				ecc.hashrate.Mark(attempts)
				attempts = 0
			}

			digest := make([]byte, 40)
			copy(digest, hash)

			digest[32] = byte(nonce)
			digest[33] = byte(nonce >> 8)
			digest[34] = byte(nonce >> 16)
			digest[35] = byte(nonce >> 24)
			digest[36] = byte(nonce >> 32)
			digest[37] = byte(nonce >> 40)
			digest[38] = byte(nonce >> 48)
			digest[39] = byte(nonce >> 56)

			digest = crypto.Keccak512(digest)

			goRoutineHashVector := make([]int, param_n)

			for i := 0; i < param_n/8; i++ {
				decimal := int(digest[i])
				for j := 7; j >= 0; j-- {
					goRoutineHashVector[j+8*i] = decimal % 2
					decimal /= 2
				}
			}

			goRoutineOutputWord := OptimizedDecodingSeoul_gpu(param_n, param_m, param_wc, param_wr, goRoutineHashVector, rowInCol, colInRow)

			flag := MakeDecision_Seoul_gpu(param_n, param_m, param_wr, colInRow, goRoutineOutputWord)

			if flag == true {
				var mixDigest [32]byte
				var nonceEncoded [8]byte
				outputWord := goRoutineOutputWord

				header.CodeLength = uint64(param_n)

				if len(digest) > len(mixDigest) {
					digest = digest[len(digest)-32:]
				}
				copy(mixDigest[32-len(digest):], digest)
				header.MixDigest = mixDigest

				nonceEncoded[0] = byte(nonce >> 56)
				nonceEncoded[1] = byte(nonce >> 48)
				nonceEncoded[2] = byte(nonce >> 40)
				nonceEncoded[3] = byte(nonce >> 32)
				nonceEncoded[4] = byte(nonce >> 24)
				nonceEncoded[5] = byte(nonce >> 16)
				nonceEncoded[6] = byte(nonce >> 8)
				nonceEncoded[7] = byte(nonce)
				header.Nonce = nonceEncoded

				var codeword []byte
				var codeVal byte
				for i, v := range outputWord {
					codeVal |= byte(v) << (7 - i%8)
					if i%8 == 7 {
						codeword = append(codeword, codeVal)
						codeVal = 0
					}
				}
				if len(outputWord)%8 != 0 {
					codeword = append(codeword, codeVal)
				}
				header.Codeword = make([]byte, len(codeword))
				copy(header.Codeword, codeword)

				select {
				case found <- &header:
					break search
				case <-abort:
					break search
				}

			}
			nonce++
		}
	}
}

// GPU MINING... NEED TO UPDTAE
// This is the timeout for HTTP requests to notify external miners.
const remoteSealerTimeout = 1 * time.Second

type remoteSealer struct {
	works        map[common.Hash]*types.Block
	rates        map[common.Hash]hashrate
	currentBlock *types.Block
	currentWork  [4]string
	notifyCtx    context.Context
	cancelNotify context.CancelFunc // cancels all notification requests
	reqWG        sync.WaitGroup     // tracks notification request goroutines

	ecc          *ECC
	noverify     bool
	notifyURLs   []string
	results      chan<- *types.Block
	workCh       chan *sealTask   // Notification channel to push new work and relative result channel to remote sealer
	fetchWorkCh  chan *sealWork   // Channel used for remote sealer to fetch mining work
	submitWorkCh chan *mineResult // Channel used for remote sealer to submit their mining result
	fetchRateCh  chan chan uint64 // Channel used to gather submitted hash rate for local or remote sealer.
	submitRateCh chan *hashrate   // Channel used for remote sealer to submit their mining hashrate
	requestExit  chan struct{}
	exitCh       chan struct{}
}

// sealTask wraps a seal block with relative result channel for remote sealer thread.
type sealTask struct {
	block   *types.Block
	results chan<- *types.Block
}

// mineResult wraps the pow solution parameters for the specified block.
type mineResult struct {
	nonce     types.BlockNonce
	mixDigest common.Hash
	hash      common.Hash

	errc chan error
}

// hashrate wraps the hash rate submitted by the remote sealer.
type hashrate struct {
	id   common.Hash
	ping time.Time
	rate uint64

	done chan struct{}
}

// sealWork wraps a seal work package for remote sealer.
type sealWork struct {
	errc chan error
	res  chan [4]string
}

func startRemoteSealer(ecc *ECC, urls []string, noverify bool) *remoteSealer {
	ctx, cancel := context.WithCancel(context.Background())
	s := &remoteSealer{
		ecc:          ecc,
		noverify:     noverify,
		notifyURLs:   urls,
		notifyCtx:    ctx,
		cancelNotify: cancel,
		works:        make(map[common.Hash]*types.Block),
		rates:        make(map[common.Hash]hashrate),
		workCh:       make(chan *sealTask),
		fetchWorkCh:  make(chan *sealWork),
		submitWorkCh: make(chan *mineResult),
		fetchRateCh:  make(chan chan uint64),
		submitRateCh: make(chan *hashrate),
		requestExit:  make(chan struct{}),
		exitCh:       make(chan struct{}),
	}
	go s.loop()
	return s
}

func (s *remoteSealer) loop() {
	defer func() {
		s.ecc.config.Log.Trace("ECC remote sealer is exiting")
		s.cancelNotify()
		s.reqWG.Wait()
		close(s.exitCh)
	}()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case work := <-s.workCh:
			// Update current work with new received block.
			// Note same work can be past twice, happens when changing CPU threads.
			s.results = work.results
			s.makeWork(work.block)
			s.notifyWork()

		case work := <-s.fetchWorkCh:
			// Return current mining work to remote miner.
			if s.currentBlock == nil {
				work.errc <- errNoMiningWork
			} else {
				work.res <- s.currentWork
			}

		case result := <-s.submitWorkCh:
			// Verify submitted PoW solution based on maintained mining blocks.
			if s.submitWork(result.nonce, result.mixDigest, result.hash) {
				result.errc <- nil
			} else {
				result.errc <- errInvalidSealResult
			}

		case result := <-s.submitRateCh:
			// Trace remote sealer's hash rate by submitted value.
			s.rates[result.id] = hashrate{rate: result.rate, ping: time.Now()}
			close(result.done)

		case req := <-s.fetchRateCh:
			// Gather all hash rate submitted by remote sealer.
			var total uint64
			for _, rate := range s.rates {
				// this could overflow
				total += rate.rate
			}
			req <- total

		case <-ticker.C:
			// Clear stale submitted hash rate.
			for id, rate := range s.rates {
				if time.Since(rate.ping) > 10*time.Second {
					delete(s.rates, id)
				}
			}
			// Clear stale pending blocks
			if s.currentBlock != nil {
				for hash, block := range s.works {
					if block.NumberU64()+staleThreshold <= s.currentBlock.NumberU64() {
						delete(s.works, hash)
					}
				}
			}

		case <-s.requestExit:
			return
		}
	}
}

// makeWork creates a work package for external miner.
//
// The work package consists of 3 strings:
//
//	result[0], 32 bytes hex encoded current block header pow-hash
//	result[1], 32 bytes hex encoded seed hash used for DAG
//	result[2], 32 bytes hex encoded boundary condition ("target"), 2^256/difficulty
//	result[3], hex encoded block number
func (s *remoteSealer) makeWork(block *types.Block) {
	hash := s.ecc.SealHash(block.Header())
	s.currentWork[0] = hash.Hex()
	s.currentWork[1] = common.BytesToHash(SeedHash(block.NumberU64())).Hex()
	s.currentWork[2] = common.BytesToHash(new(big.Int).Div(two256, block.Difficulty()).Bytes()).Hex()
	s.currentWork[3] = hexutil.EncodeBig(block.Number())

	// Trace the seal work fetched by remote sealer.
	s.currentBlock = block
	s.works[hash] = block
}

// notifyWork notifies all the specified mining endpoints of the availability of
// new work to be processed.
func (s *remoteSealer) notifyWork() {
	work := s.currentWork

	// Encode the JSON payload of the notification. When NotifyFull is set,
	// this is the complete block header, otherwise it is a JSON array.
	var blob []byte
	if s.ecc.config.NotifyFull {
		blob, _ = json.Marshal(s.currentBlock.Header())
	} else {
		blob, _ = json.Marshal(work)
	}

	s.reqWG.Add(len(s.notifyURLs))
	for _, url := range s.notifyURLs {
		go s.sendNotification(s.notifyCtx, url, blob, work)
	}
}

func (s *remoteSealer) sendNotification(ctx context.Context, url string, json []byte, work [4]string) {
	defer s.reqWG.Done()

	req, err := http.NewRequest("POST", url, bytes.NewReader(json))
	if err != nil {
		s.ecc.config.Log.Warn("Can't create remote miner notification", "err", err)
		return
	}
	ctx, cancel := context.WithTimeout(ctx, remoteSealerTimeout)
	defer cancel()
	req = req.WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		s.ecc.config.Log.Warn("Failed to notify remote miner", "err", err)
	} else {
		s.ecc.config.Log.Trace("Notified remote miner", "miner", url, "hash", work[0], "target", work[2])
		resp.Body.Close()
	}
}

// submitWork verifies the submitted pow solution, returning
// whether the solution was accepted or not (not can be both a bad pow as well as
// any other error, like no pending work or stale mining result).
func (s *remoteSealer) submitWork(nonce types.BlockNonce, mixDigest common.Hash, sealhash common.Hash) bool {
	if s.currentBlock == nil {
		s.ecc.config.Log.Error("Pending work without block", "sealhash", sealhash)
		return false
	}
	// Make sure the work submitted is present
	block := s.works[sealhash]
	if block == nil {
		s.ecc.config.Log.Warn("Work submitted but none pending", "sealhash", sealhash, "curnumber", s.currentBlock.NumberU64())
		return false
	}
	// Verify the correctness of submitted result.
	header := block.Header()
	header.Nonce = nonce
	header.MixDigest = mixDigest

	start := time.Now()
	if !s.noverify {
		if err := s.ecc.verifySeal(nil, header); err != nil {
			s.ecc.config.Log.Warn("Invalid proof-of-work submitted", "sealhash", sealhash, "elapsed", common.PrettyDuration(time.Since(start)), "err", err)
			return false
		}
	}
	// Make sure the result channel is assigned.
	if s.results == nil {
		s.ecc.config.Log.Warn("Eccresult channel is empty, submitted mining result is rejected")
		return false
	}
	s.ecc.config.Log.Trace("Verified correct proof-of-work", "sealhash", sealhash, "elapsed", common.PrettyDuration(time.Since(start)))

	// Solutions seems to be valid, return to the miner and notify acceptance.
	solution := block.WithSeal(header)

	// The submitted solution is within the scope of acceptance.
	if solution.NumberU64()+staleThreshold > s.currentBlock.NumberU64() {
		select {
		case s.results <- solution:
			s.ecc.config.Log.Debug("Work submitted is acceptable", "number", solution.NumberU64(), "sealhash", sealhash, "hash", solution.Hash())
			return true
		default:
			s.ecc.config.Log.Warn("Sealing result is not read by miner", "mode", "remote", "sealhash", sealhash)
			return false
		}
	}
	// The submitted block is too old to accept, drop it.
	s.ecc.config.Log.Warn("Work submitted is too old", "number", solution.NumberU64(), "sealhash", sealhash, "hash", solution.Hash())
	return false
}
