/* author: Kexuan Zou
   date: 03/11/2018
*/

#ifndef _AES_H
#define _AES_H
#include "types.h"

#define AES_BLOCK_BITS 128 // block size is 128 bits
#define AES_BLOCK_SIZE 16
#define AES_KEY_BITS AES_BLOCK_BITS // key size for AES 128 is 128 bits
#define AES_NUM_ROUND 10 // number of rounds for AES 128
#define AES_COLS_STATE 4 // number of columns in a state is 4
#define AES_ROW_STATE (AES_BLOCK_SIZE / AES_COLS_STATE)
#define AES_KEY_ENTRIES 4 // number of entries in a key
#define AES_KEY_SIZE AES_BLOCK_SIZE
#define AES_BYTE_MASK 0xFF
#define AES_SCHED_SIZE 176 // expansion key size is 176
#define AES_EXPD_ROUND (AES_SCHED_SIZE / AES_KEY_ENTRIES) // expansion key rounds are 44

void sub_word(uint8_t* word);
void rot_word(uint8_t* word);
void aes_128_key_expansion(uint8_t* key);
__global__ void aes_128_encryption(uint8_t* buffer, uint8_t* sched_key);
__global__ void aes_128_decryption(uint8_t* buffer, uint8_t* sched_key);

/**
 * XTIME - helper function for power of x multiplication. higher powers of x can be implementated by repeated application of this funciton.
 * @param num - number to be transformed
 */
#define XTIME(num) \
    (((num) << 1) ^ ((((num) >> 7) & 1) * 0x1B))

/**
 * gf_mul - multiple two numbers in the field of GF(2^8).
 * @param x, y - two bytes to multiply
 * @return - its GF(2^8) multiplicative value
 */
__device__ static inline uint8_t gf_mul(uint8_t x, uint8_t y) {
    return (((y & 1) * x) ^
     ((y >> 1 & 1) * XTIME(x)) ^
     ((y >> 2 & 1) * XTIME(XTIME(x))) ^
     ((y >> 3 & 1) * XTIME(XTIME(XTIME(x)))) ^
     ((y >> 4 & 1) * XTIME(XTIME(XTIME(XTIME(x))))));
}


/**
 * sub_bytes - substitutes value in the state matrix with equivalent sbox values
 * @param state - state of the block in current round
 * @param sbox - sbox in device shared memeory
 */
__device__ static inline void sub_bytes(uint8_t* state, uint8_t* sbox) {
    int i = 0;
    for (; i < AES_BLOCK_SIZE; i++)
        state[i] = sbox[state[i]];
}


/**
 * inv_sub_bytes - inversely substitutes value in the state matrix with equivalent sbox values
 * @param state - state of the block in current round
 * @param invsbox - inverse sbox in device shared memory
 */
__device__ static inline void inv_sub_bytes(uint8_t* state, uint8_t* invsbox) {
    int i = 0;
    for (; i < AES_BLOCK_SIZE; i++)
        state[i] = invsbox[state[i]];
}


/**
 * mix_columns - combine four bytes of each column of the states using invertible linear transformation, defined as follows:
 *  2  3  1  1
 *  1  2  3  1
 *  1  1  2  3
 *  3  1  1  2
 * @param state - state of the block in current round
 */
__device__ static inline void mix_columns(uint8_t* state) {
    uint8_t temp[AES_BLOCK_SIZE]; // temporary array to hold the state
    int i;
    for (i = 0; i < AES_BLOCK_SIZE; i++)
        temp[i] = state[i];
    for (i = 0; i < AES_COLS_STATE; i++) {
        state[AES_ROW_STATE*i] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x02) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x03) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x01) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x01);
        
        state[AES_ROW_STATE*i+1] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x01) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x02) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x03) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x01);
        
        state[AES_ROW_STATE*i+2] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x01) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x01) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x02) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x03);
        
        state[AES_ROW_STATE*i+3] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x03) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x01) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x01) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x02);
    }
}


/**
* mix_columns - inversely transform the state. the invese matrix is defined as follows.
*  e  b  d  9
*  9  e  b  d
*  d  9  e  b
*  b  d  9  e
* @param state - state of the block in current round
 */
__device__ static inline void inv_mix_columns(uint8_t* state) {
    uint8_t temp[AES_BLOCK_SIZE]; // temporary array to hold the state
    int i;
    for (i = 0; i < AES_BLOCK_SIZE; i++)
        temp[i] = state[i];
    for (i = 0; i < AES_COLS_STATE; i++) {
        state[AES_ROW_STATE*i] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x0E) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x0B) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x0D) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x09);
        
        state[AES_ROW_STATE*i+1] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x09) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x0E) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x0B) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x0D);
        
        state[AES_ROW_STATE*i+2] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x0D) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x09) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x0E) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x0B);
        
        state[AES_ROW_STATE*i+3] = 
        gf_mul(temp[AES_ROW_STATE*i], 0x0B) ^ 
        gf_mul(temp[AES_ROW_STATE*i+1], 0x0D) ^ 
        gf_mul(temp[AES_ROW_STATE*i+2], 0x09) ^ 
        gf_mul(temp[AES_ROW_STATE*i+3], 0x0E);
    }
}


/**
 * shift_rows - shift rows of the state matrix to left, each with a different offset scheme stated as follows:
 * 0: [s0]  [s4]  [s8]  [s12] << 0
 * 1: [s1]  [s5]  [s9]  [s13] << 1
 * 2: [s2]  [s6]  [s10] [s14] << 2
 * 3: [s3]  [s7]  [s11] [s15] << 3
 * @param state - state of the block in current round
 */
__device__ static inline void shift_rows(uint8_t* state) {
    register uint8_t temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;
    temp = state[3];
    state[3] = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = temp;
}


/**
 * inv_shift_rows - shift rows of the state matrix to right, each with a different offset scheme stated as follows:
 * 0: [s0]  [s4]  [s8]  [s12] >> 0
 * 1: [s1]  [s5]  [s9]  [s13] >> 1
 * 2: [s2]  [s6]  [s10] [s14] >> 2
 * 3: [s3]  [s7]  [s11] [s15] >> 3
 * @param state - state of the block in current round
 */
__device__ static inline void inv_shift_rows(uint8_t* state) {
    uint8_t temp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = temp;
    temp = state[14];
    state[14] = state[6];
    state[6] = temp;
    temp = state[10];
    state[10] = state[2];
    state[2] = temp;
    temp = state[7];
    state[7] = state[11];
    state[11] = state[15];
    state[15] = state[3];
    state[3] = temp;
}


/**
 * add_round_key - adds round key to each state be an xor operation
 * @param state - state of the block in current round
 * @param round_idx - round key to add to each state
 */
__device__ static inline void add_round_key(uint8_t* state, uint8_t* key, int round_idx) {
    int i = 0;
    for (; i < AES_COLS_STATE; i++) {
        state[i] = state[i] ^ key[(AES_BLOCK_SIZE*round_idx)+i];
        state[i+4] = state[i+4] ^ key[(AES_BLOCK_SIZE*round_idx)+i+4];
        state[i+8] = state[i+8] ^ key[(AES_BLOCK_SIZE*round_idx)+i+8];
        state[i+12] = state[i+12] ^ key[(AES_BLOCK_SIZE*round_idx)+i+12];
    }
}

#endif