#include <iostream>
#include <stdint.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#define BigInfinity 1000000.0
#define Inf 64.0
#define maxIter 20
#define crossErr 0.01
#define STREAM_SIZE 2   //2
#define GRID_SIZE 6400     //6400
#define BLOCK_SIZE 4    //4
#define KECCAKF_ROUNDS 24

#define CUDA_SAFE_CALL(call)                                                          \
    cudaError_t err = call;                                                           \
    if (cudaSuccess != err)                                                           \
    {                                                                                 \
        fprintf(stderr, "CUDA error in func %s at line %d: %s\n", __FUNCTION__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                           \
    }


#define ROL64(a, offset) ((a << offset) | (a >> (64 - offset)))

typedef struct {
    uint8_t MixDigest[32];
    uint8_t Nonce[8];
    uint8_t Codeword[256];
    bool FinishFlag;
} Header_kernel;

__device__ static const uint64_t keccakf_rndc[KECCAKF_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ static const int keccakf_rotc[KECCAKF_ROUNDS] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static const int keccakf_piln[KECCAKF_ROUNDS] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

__device__ void keccakf(uint64_t st[25]) {
    int i, j, r;
    uint64_t t, bc[5];

    for (r = 0; r < KECCAKF_ROUNDS; r++) {
        for (i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ ROL64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5) {
                st[j + i] ^= t;
            }
        }

        t = st[1];
        for (i = 0; i < 24; i++) {
            j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ROL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++) {
                bc[i] = st[j + i];
            }
            for (i = 0; i < 5; i++) {
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
        }

        st[0] ^= keccakf_rndc[r];
    }
}

__device__ void keccak_absorb(uint64_t* state, const uint8_t* in) {
    size_t inlen = 40;
    size_t rsize = 72;
    size_t i;
    uint8_t temp[200] = { 0 };

    memcpy(temp, in, inlen);
    temp[inlen++] = 0x01;
    temp[rsize - 1] |= 0x80;

    for (i = 0; i < rsize / 8; i++) {
        state[i] ^= ((uint64_t*)temp)[i];
    }
    keccakf(state);
}

__device__ void keccak_squeeze(uint64_t* state, uint8_t* out) {
    size_t outlen = 64;
    size_t rsize = 72;
    size_t i;

    while (outlen > 0) {
        if (outlen >= rsize) {
            for (i = 0; i < rsize / 8; i++) {
                ((uint64_t*)out)[i] = state[i];
            }
            keccakf(state);
            out += rsize;
            outlen -= rsize;
        }
        else {
            memcpy(out, state, outlen);
            break;
        }
    }
}

__device__ void keccak512(uint8_t* digest) {
    uint64_t state[25] = { 0 };

    keccak_absorb(state, digest);
    keccak_squeeze(state, digest);
}

///////////////////////////////////
///////////////////////////////////

__device__ float infinityTest(float x) {
    if (x >= Inf) {
        return Inf;
    } else if (x <= -Inf) {
        return -Inf;
    } else {
        return x;
    }
}

__device__ float funcF(float x) {
    if (x >= BigInfinity) {
        return 1.0 / BigInfinity;
    } else if (x <= (1.0 / BigInfinity)) {
        return BigInfinity;
    } else {
        return log((exp(x) + 1) / (exp(x) - 1));
    }
}

__device__ void optimizedDecodingSeoulCuda(int param_n, int param_m, int param_wc, int param_wr, uint8_t* hashVector, uint16_t* rowInCol, uint16_t* colInRow, float* g_a, float* g_b, float* g_c, float* g_d, int stream_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int ab_offset = idx * param_n * param_m + stream_id * GRID_SIZE * BLOCK_SIZE * param_n * param_m;
    int cd_offset = idx * param_n + stream_id * GRID_SIZE * BLOCK_SIZE * param_n;

    for (int i = 0; i < param_n; i++) {
        for (int j = 0; j < param_m; j++) {
            g_a[ab_offset + i * param_m + j] = 0.0;
            g_b[ab_offset + i * param_m + j] = 0.0;
        }
        g_c[cd_offset+i] = log((1 - crossErr) / crossErr) * float((hashVector[i] * 2 - 1));
    }

    for (int ind = 1; ind <= maxIter; ind++) {
        for (int t = 0; t < param_n; t++) {
            float temp3 = 0.0;

            for (int mp = 0; mp < param_wc; mp++) {
                temp3 = infinityTest(temp3 + g_b[ab_offset + t * param_m + rowInCol[mp * param_n + t]]);
            }

            for (int m = 0; m < param_wc; m++) {
                float temp4 = temp3;
                temp4 = infinityTest(temp4 - g_b[ab_offset + t * param_m + rowInCol[m * param_n + t]]);
                g_a[ab_offset + t * param_m + rowInCol[m * param_n + t]] = infinityTest(g_c[cd_offset+t] + temp4);
            }
        }

        for (int k = 0; k < param_m; k++) {
            for (int l = 0; l < param_wr; l++) {
                float temp3 = 0.0;
                float sign = 1.0;
                float tempSign = 0.0;

                for (int m = 0; m < param_wr; m++) {
                    if (m != l) {
                        temp3 += funcF(fabs(g_a[ab_offset + colInRow[m * param_m + k] * param_m + k]));
                        if (g_a[ab_offset + colInRow[m * param_m + k] * param_m + k] > 0.0) {
                            tempSign = 1.0;
                        } else {
                            tempSign = -1.0;
                        }
                        sign *= tempSign;
                    }
                }

                float magnitude = funcF(temp3);
                g_b[ab_offset + colInRow[l * param_m + k] * param_m + k] = infinityTest(sign * magnitude);
            }
        }

        for (int t = 0; t < param_n; t++) {
            g_d[cd_offset+t] = infinityTest(g_c[cd_offset+t]);
            for (int k = 0; k < param_wc; k++) {
                g_d[cd_offset+t] += g_b[ab_offset + t * param_m + rowInCol[k * param_n + t]];
                g_d[cd_offset+t] = infinityTest(g_d[cd_offset+t]);
            }

            if (g_d[cd_offset+t] >= 0) {
                hashVector[t] = 1;
            } else {
                hashVector[t] = 0;
            }
        }
    }
}

__device__ bool makeDecisionSeoulCuda(int param_n, int param_m, int param_wr, uint16_t* colInRow, uint8_t* outputWord) {
    for (int i = 0; i < param_m; i++) {
        int sum = 0;
        for (int j = 0; j < param_wr; j++) {
            sum += outputWord[colInRow[j * param_m + i]];
        }
        if (sum % 2 == 1) {
            return false;
        }
    }

    int numOfOnes = 0;
    for (int i = 0; i < param_n; i++) {
        numOfOnes += outputWord[i];
    }

    if (numOfOnes >= param_n / 4 && numOfOnes <= param_n / 4 * 3) {
        return true;
    }

    return false;
}

__global__ void mineSeoulCudaKernel(uint8_t* hash, uint64_t seed, int param_n, int param_m, int param_wc, int param_wr, uint16_t* rowInCol, uint16_t* colInRow, Header_kernel* result, volatile bool* found, float* g_a, float* g_b, float* g_c, float* g_d, int stream_id, uint8_t* g_outputWord) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = seed + idx;
    int outputWord_offset = idx * param_n + stream_id * GRID_SIZE * BLOCK_SIZE * param_n;
    
    uint8_t digest[64];
    uint8_t* outputWord = g_outputWord + outputWord_offset;
    memset(outputWord, 0, param_n);
    memcpy(digest, hash, 32);

    digest[32] = (uint8_t)nonce;
    digest[33] = (uint8_t)(nonce >> 8);
    digest[34] = (uint8_t)(nonce >> 16);
    digest[35] = (uint8_t)(nonce >> 24);
    digest[36] = (uint8_t)(nonce >> 32);
    digest[37] = (uint8_t)(nonce >> 40);
    digest[38] = (uint8_t)(nonce >> 48);
    digest[39] = (uint8_t)(nonce >> 56);

    keccak512(digest);

    /*for(int i=0; i<64; i++){
        printf("%d,", digest[i]);
    }
    printf("\n\n");*/

    for (int i = 0; i < param_n / 8; i++) {
        int decimal = (int)digest[i];
        for (int j = 7; j >= 0; j--) {
            outputWord[j + 8 * i] = decimal % 2;
            decimal /= 2;
        }
    }
   
    /*for(int i=0; i<param_n; i++){
        printf("%d", outputWord[i]);
    }
    printf("\n\n");*/

    optimizedDecodingSeoulCuda(param_n, param_m, param_wc, param_wr, outputWord, rowInCol, colInRow, g_a, g_b, g_c, g_d, stream_id);

    //if (true){
    if (makeDecisionSeoulCuda(param_n, param_m, param_wr, colInRow, outputWord)) {
    //if (outputWord[0]==1&&outputWord[1]==1&&outputWord[2]==1&&outputWord[3]==1&&outputWord[4]==1&&outputWord[5]==1&&outputWord[6]==1&&outputWord[7]==1&&outputWord[8]==1&&outputWord[9]==1&&outputWord[10]==1&&outputWord[11]==1&&outputWord[12]==1&&outputWord[13]==1&&outputWord[14]==1&&outputWord[15]==1&&outputWord[16]==1&&outputWord[17]==1&&outputWord[18]==1&&outputWord[19]==1&&outputWord[20]==1&&outputWord[21]==1&&outputWord[22]==1) {
        printf("found! %lld %d %d %d\n", nonce, stream_id, blockIdx.x, threadIdx.x);
        uint8_t mixDigest[32];
        int digestLen = sizeof(digest) / sizeof(digest[0]);
        int outputWordLen = param_n;

        if (digestLen > 32) {
            int startIndex = digestLen - 32;
            for (int i = 0; i < 32; ++i) {
                digest[i] = digest[startIndex + i];
            }
            digestLen = 32;
        }

        memcpy(&mixDigest[32 - digestLen], digest, digestLen);
        memcpy(result->MixDigest, mixDigest, 32);

        uint8_t nonceEncoded[8] = {
            (uint8_t)(nonce >> 56),
            (uint8_t)(nonce >> 48),
            (uint8_t)(nonce >> 40),
            (uint8_t)(nonce >> 32),
            (uint8_t)(nonce >> 24),
            (uint8_t)(nonce >> 16),
            (uint8_t)(nonce >> 8),
            (uint8_t)(nonce)
        };
        memcpy(result->Nonce, nonceEncoded, 8);

        int codewordLen = (outputWordLen + 7) / 8;
        uint8_t* codeword = new uint8_t[codewordLen];
        uint8_t codeVal = 0;
        for (uint64_t i = 0; i < outputWordLen; ++i) {
            codeVal |= (outputWord[i] << (7 - (i % 8)));
            if (i % 8 == 7) {
                codeword[i / 8] = codeVal;
                codeVal = 0;
            }
        }
        if (outputWordLen % 8 != 0) {
            codeword[codewordLen - 1] = codeVal;
        }

        memcpy(result->Codeword, codeword, codewordLen);

        *found = true;
    }
}


extern "C" {
    __declspec(dllexport) void mineSeoulCuda(int gpu_num, uint8_t* c_hash, uint64_t seed, int param_n, int param_m, int param_wc, int param_wr, uint16_t* c_rowInCol, uint16_t* c_colInRow, Header_kernel* result, bool* abort) {
        printf("$$$$$$$$$$$$$$$$$$$$ New block start $$$$$$$$$$$$$$$$$$$$\n");
        
        int count = 0;
        uint64_t nonce = seed;
        CUDA_SAFE_CALL(cudaSetDevice(gpu_num));

        size_t free_mem = 0;
        size_t total_mem = 0;

        cudaStream_t streams[STREAM_SIZE];

        Header_kernel* g_found_header;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_found_header, sizeof(Header_kernel)));
        
        volatile bool* g_found;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_found, sizeof(bool)));
        CUDA_SAFE_CALL(cudaMemset((void*)g_found, 0, sizeof(bool)));

        uint8_t *g_hash;    //to __constant__
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_hash, 32 * sizeof(uint8_t)));
        CUDA_SAFE_CALL(cudaMemcpy(g_hash, c_hash, 32 * sizeof(uint8_t), cudaMemcpyHostToDevice));

        uint16_t *g_colInRow;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_colInRow, param_wr * param_m * sizeof(uint16_t)));
        CUDA_SAFE_CALL(cudaMemcpy(g_colInRow, c_colInRow, param_wr * param_m * sizeof(uint16_t), cudaMemcpyHostToDevice));

        uint16_t *g_rowInCol;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_rowInCol, param_wc * param_n * sizeof(uint16_t)));
        CUDA_SAFE_CALL(cudaMemcpy(g_rowInCol, c_rowInCol, param_wc * param_n * sizeof(uint16_t), cudaMemcpyHostToDevice));
        
        for (int i = 0; i < STREAM_SIZE; i++) {
            CUDA_SAFE_CALL(cudaStreamCreate(&streams[i]));
        }

        float *g_a;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_a, STREAM_SIZE * GRID_SIZE * BLOCK_SIZE * param_m * param_n * sizeof(float)));

        float *g_b;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_b, STREAM_SIZE * GRID_SIZE * BLOCK_SIZE * param_m * param_n * sizeof(float)));

        float *g_c;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_c, STREAM_SIZE * GRID_SIZE * BLOCK_SIZE * param_n * sizeof(float)));

        float *g_d;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_d, STREAM_SIZE * GRID_SIZE * BLOCK_SIZE * param_n * sizeof(float)));

        uint8_t *g_outputWord;
        CUDA_SAFE_CALL(cudaMalloc((void**)&g_outputWord, STREAM_SIZE * GRID_SIZE * BLOCK_SIZE * param_n * sizeof(uint8_t)));

        int stream_id;
        bool c_found = false;
        clock_t start_time = clock();

        /*CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
        printf("%zu  %zu\n", free_mem, total_mem);*/

        while (!c_found && !(*abort)) {
            for (stream_id = 0; stream_id < STREAM_SIZE ; stream_id++) {
                mineSeoulCudaKernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams[stream_id]>>>(g_hash, nonce+count, param_n, param_m, param_wc, param_wr, g_rowInCol, g_colInRow, g_found_header, g_found, g_a, g_b, g_c, g_d, stream_id, g_outputWord);
                count+=GRID_SIZE*BLOCK_SIZE;
            }

            CUDA_SAFE_CALL(cudaMemcpy((void*)&c_found, (void*)g_found, sizeof(bool), cudaMemcpyDeviceToHost));

            clock_t current_time = clock();
            double elapsed_time = (double)(current_time - start_time) / CLOCKS_PER_SEC;

            double counts_per_sec = count / elapsed_time;
            printf("%d > %f H/s\n", count, counts_per_sec);
        }

        printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\\n");

        if (c_found) {
            result->FinishFlag = true;
            CUDA_SAFE_CALL(cudaMemcpy((void*)result, (void*)g_found_header, sizeof(Header_kernel), cudaMemcpyDeviceToHost));
        }

        for (int i = 0; i < STREAM_SIZE; i++) {
            CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]));
        }

        CUDA_SAFE_CALL(cudaFree((void*)g_found_header));
        CUDA_SAFE_CALL(cudaFree((void*)g_found));
        CUDA_SAFE_CALL(cudaFree((void*)g_hash));
        CUDA_SAFE_CALL(cudaFree((void*)g_colInRow));
        CUDA_SAFE_CALL(cudaFree((void*)g_rowInCol));
        CUDA_SAFE_CALL(cudaFree((void*)g_a));
        CUDA_SAFE_CALL(cudaFree((void*)g_b));
        CUDA_SAFE_CALL(cudaFree((void*)g_c));
        CUDA_SAFE_CALL(cudaFree((void*)g_d));
        CUDA_SAFE_CALL(cudaFree((void*)g_outputWord));
    }
}