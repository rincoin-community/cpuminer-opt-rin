/*
 * RinHash core: BLAKE3(80-byte header) -> Argon2d -> SHA3-256
 *
 * Per hash the work is, in this order:
 *   1. BLAKE3 of the header. The first 64 bytes are constant for a job, so
 *      their chaining value is computed once and each nonce costs a single
 *      BLAKE3 compression.
 *   2. Argon2d H0, one BLAKE2b compression.
 *   3. The first two Argon2 blocks, 31 chained BLAKE2b compressions each.
 *   4. The Argon2 memory fill, 126 blocks of BlaMka.
 *   5. The Argon2 tag, BLAKE2b over the last block, 9 compressions.
 *   6. SHA3-256 of the tag, one Keccak-f[1600].
 *
 * The Argon2 library wrapper (argon2d_ctx) is not used here: it validates
 * its inputs, zeroes the memory and wipes it again on every call, none of
 * which a miner needs. Argon2 writes every block before reading it, so the
 * memory does not have to be cleared between hashes.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "rinhash_core.h"
#include "blake3/blake3.h"
#include "blake3/blake3_impl.h"
#include "argon2/include/argon2.h"
#include "argon2/core.h"
#include "argon2/blake2/blake2.h"
#include "argon2/argon2_dispatch.h"

#if RINHASH_BATCH > 1
#include <immintrin.h>
#endif

#define RIN_MEM_BLOCKS  64
#define RIN_T_COST      2
#define RIN_SEGMENT     ( RIN_MEM_BLOCKS / ARGON2_SYNC_POINTS )
#define RIN_SALT        "RinCoinSalt"
#define RIN_SALTLEN     11
// argon2_set_impl: any value without an implementation selects the best one
#define RIN_IMPL_AUTO   ARGON2_IMPL_COUNT

#define likely( x )     __builtin_expect( !!(x), 1 )

#ifdef RINHASH_PROFILE
#include <x86intrin.h>
__thread uint64_t rinhash_prof[ RINHASH_PROF_N ];
#define PROF_START      uint64_t prof_t = __rdtsc()
#define PROF( stage ) \
   do { uint64_t t_ = __rdtsc(); rinhash_prof[ stage ] += t_ - prof_t; \
        prof_t = t_; } while (0)
#else
#define PROF_START      do {} while (0)
#define PROF( stage )   do {} while (0)
#endif

struct rinhash_ctx
{
   block *memory;                // RIN_MEM_BLOCKS Argon2 blocks
   argon2_context argon;
   argon2_instance_t inst;
   uint32_t b3_cv[8];            // BLAKE3 state after header bytes 0..63
   uint8_t  b3_tail[64];         // header bytes 64..79, zero padded
#if RINHASH_BATCH > 1
   // Word w of nonce n is at [w][n], so that one aligned vector load yields
   // the same word of all nonces.
   uint64_t first[2][ ARGON2_QWORDS_IN_BLOCK ][ RINHASH_BATCH ]
                                          __attribute__ ((aligned (64)));
   uint64_t last[ ARGON2_QWORDS_IN_BLOCK ][ RINHASH_BATCH ]
                                          __attribute__ ((aligned (64)));
#endif
};

extern int FLAG_clear_internal_memory;

#if RINHASH_BATCH > 1
static void rin_h0_template_init( void );
#endif

void rinhash_global_init( void )
{
   argon2_dispatch_init();
#if RINHASH_BATCH > 1
   rin_h0_template_init();
#endif
   // The Argon2 code wipes its temporaries after every BLAKE2b call. Nothing
   // here is secret.
   FLAG_clear_internal_memory = 0;
}

rinhash_ctx_t *rinhash_ctx_new( void )
{
   rinhash_ctx_t *ctx = NULL;
   void *mem = NULL;
   const size_t mem_size = RIN_MEM_BLOCKS * sizeof(block);

   if ( posix_memalign( (void**)&ctx, 64, sizeof(*ctx) ) ) return NULL;
   if ( posix_memalign( &mem, 4096, mem_size ) )
   {
      free( ctx );
      return NULL;
   }
   // First touch by the calling thread places the pages on its NUMA node.
   memset( ctx, 0, sizeof(*ctx) );
   memset( mem, 0, mem_size );
   ctx->memory = (block*)mem;

   ctx->argon.outlen  = 32;
   ctx->argon.pwdlen  = 32;
   ctx->argon.salt    = (uint8_t*)RIN_SALT;
   ctx->argon.saltlen = RIN_SALTLEN;
   ctx->argon.t_cost  = RIN_T_COST;
   ctx->argon.m_cost  = RIN_MEM_BLOCKS;
   ctx->argon.lanes   = 1;
   ctx->argon.threads = 1;
   ctx->argon.version = ARGON2_VERSION_13;
   ctx->argon.flags   = ARGON2_DEFAULT_FLAGS;

   ctx->inst.memory         = ctx->memory;
   ctx->inst.version        = ARGON2_VERSION_13;
   ctx->inst.passes         = RIN_T_COST;
   ctx->inst.memory_blocks  = RIN_MEM_BLOCKS;
   ctx->inst.segment_length = RIN_SEGMENT;
   ctx->inst.lane_length    = RIN_MEM_BLOCKS;
   ctx->inst.lanes          = 1;
   ctx->inst.threads        = 1;
   ctx->inst.type           = Argon2_d;
   ctx->inst.context_ptr    = &ctx->argon;
   return ctx;
}

void rinhash_ctx_free( rinhash_ctx_t *ctx )
{
   if ( !ctx ) return;
   free( ctx->memory );
   free( ctx );
}

void rinhash_set_header( rinhash_ctx_t *ctx, const void *header )
{
   memcpy( ctx->b3_cv, IV, sizeof(ctx->b3_cv) );
   blake3_compress_in_place( ctx->b3_cv, (const uint8_t*)header,
                             BLAKE3_BLOCK_LEN, 0, CHUNK_START );
   memcpy( ctx->b3_tail, (const uint8_t*)header + 64, 16 );
}

// BLAKE3 of the loaded header with the given nonce. The header is one
// 80-byte chunk, so the second compression is also the root.
static inline void rin_blake3( rinhash_ctx_t *ctx, uint32_t nonce,
                               uint32_t out[8] )
{
   uint8_t tail[64] __attribute__ ((aligned (32)));
   memcpy( tail, ctx->b3_tail, 64 );
   memcpy( tail + 12, &nonce, 4 );
   memcpy( out, ctx->b3_cv, 32 );
   blake3_compress_in_place( out, tail, 16, 0, CHUNK_END | ROOT );
}

/*
 * Keccak-f[1600], written over a lane type so that the same rounds serve one
 * state in 64-bit registers or several states in vector registers.
 */
static const uint64_t keccak_rc[24] =
{
   0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
   0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
   0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
   0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
   0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
   0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
   0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
   0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

#define KECCAK_F1600( T, s, XOR, ROL, ANDN, RC ) \
do { \
   T c0, c1, c2, c3, c4, d0, d1, d2, d3, d4; \
   T b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12; \
   T b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24; \
   for ( int kr_ = 0; kr_ < 24; kr_++ ) \
   { \
      c0 = XOR( XOR( XOR( s[0], s[5] ), XOR( s[10], s[15] ) ), s[20] ); \
      c1 = XOR( XOR( XOR( s[1], s[6] ), XOR( s[11], s[16] ) ), s[21] ); \
      c2 = XOR( XOR( XOR( s[2], s[7] ), XOR( s[12], s[17] ) ), s[22] ); \
      c3 = XOR( XOR( XOR( s[3], s[8] ), XOR( s[13], s[18] ) ), s[23] ); \
      c4 = XOR( XOR( XOR( s[4], s[9] ), XOR( s[14], s[19] ) ), s[24] ); \
      d0 = XOR( c4, ROL( c1, 1 ) ); \
      d1 = XOR( c0, ROL( c2, 1 ) ); \
      d2 = XOR( c1, ROL( c3, 1 ) ); \
      d3 = XOR( c2, ROL( c4, 1 ) ); \
      d4 = XOR( c3, ROL( c0, 1 ) ); \
      b0  =      XOR( s[ 0], d0 ); \
      b1  = ROL( XOR( s[ 6], d1 ), 44 ); \
      b2  = ROL( XOR( s[12], d2 ), 43 ); \
      b3  = ROL( XOR( s[18], d3 ), 21 ); \
      b4  = ROL( XOR( s[24], d4 ), 14 ); \
      b5  = ROL( XOR( s[ 3], d3 ), 28 ); \
      b6  = ROL( XOR( s[ 9], d4 ), 20 ); \
      b7  = ROL( XOR( s[10], d0 ),  3 ); \
      b8  = ROL( XOR( s[16], d1 ), 45 ); \
      b9  = ROL( XOR( s[22], d2 ), 61 ); \
      b10 = ROL( XOR( s[ 1], d1 ),  1 ); \
      b11 = ROL( XOR( s[ 7], d2 ),  6 ); \
      b12 = ROL( XOR( s[13], d3 ), 25 ); \
      b13 = ROL( XOR( s[19], d4 ),  8 ); \
      b14 = ROL( XOR( s[20], d0 ), 18 ); \
      b15 = ROL( XOR( s[ 4], d4 ), 27 ); \
      b16 = ROL( XOR( s[ 5], d0 ), 36 ); \
      b17 = ROL( XOR( s[11], d1 ), 10 ); \
      b18 = ROL( XOR( s[17], d2 ), 15 ); \
      b19 = ROL( XOR( s[23], d3 ), 56 ); \
      b20 = ROL( XOR( s[ 2], d2 ), 62 ); \
      b21 = ROL( XOR( s[ 8], d3 ), 55 ); \
      b22 = ROL( XOR( s[14], d4 ), 39 ); \
      b23 = ROL( XOR( s[15], d0 ), 41 ); \
      b24 = ROL( XOR( s[21], d1 ),  2 ); \
      s[ 0] = XOR( XOR( b0, ANDN( b1, b2 ) ), RC( keccak_rc[ kr_ ] ) ); \
      s[ 1] = XOR( b1,  ANDN( b2,  b3  ) ); \
      s[ 2] = XOR( b2,  ANDN( b3,  b4  ) ); \
      s[ 3] = XOR( b3,  ANDN( b4,  b0  ) ); \
      s[ 4] = XOR( b4,  ANDN( b0,  b1  ) ); \
      s[ 5] = XOR( b5,  ANDN( b6,  b7  ) ); \
      s[ 6] = XOR( b6,  ANDN( b7,  b8  ) ); \
      s[ 7] = XOR( b7,  ANDN( b8,  b9  ) ); \
      s[ 8] = XOR( b8,  ANDN( b9,  b5  ) ); \
      s[ 9] = XOR( b9,  ANDN( b5,  b6  ) ); \
      s[10] = XOR( b10, ANDN( b11, b12 ) ); \
      s[11] = XOR( b11, ANDN( b12, b13 ) ); \
      s[12] = XOR( b12, ANDN( b13, b14 ) ); \
      s[13] = XOR( b13, ANDN( b14, b10 ) ); \
      s[14] = XOR( b14, ANDN( b10, b11 ) ); \
      s[15] = XOR( b15, ANDN( b16, b17 ) ); \
      s[16] = XOR( b16, ANDN( b17, b18 ) ); \
      s[17] = XOR( b17, ANDN( b18, b19 ) ); \
      s[18] = XOR( b18, ANDN( b19, b15 ) ); \
      s[19] = XOR( b19, ANDN( b15, b16 ) ); \
      s[20] = XOR( b20, ANDN( b21, b22 ) ); \
      s[21] = XOR( b21, ANDN( b22, b23 ) ); \
      s[22] = XOR( b22, ANDN( b23, b24 ) ); \
      s[23] = XOR( b23, ANDN( b24, b20 ) ); \
      s[24] = XOR( b24, ANDN( b20, b21 ) ); \
   } \
} while (0)

#define K64_XOR( a, b )    ( (a) ^ (b) )
#define K64_ROL( a, n )    ( ( (a) << (n) ) | ( (a) >> ( 64 - (n) ) ) )
#define K64_ANDN( a, b )   ( ~(a) & (b) )
#define K64_RC( c )        ( c )

// SHA3-256 of a 32-byte message: one permutation. The message, the domain
// separator (byte 32) and the final bit of the 136-byte rate (byte 135) are
// placed directly into the state.
static inline void rin_sha3_256( void *out32, const void *in32 )
{
   uint64_t st[25] = { 0 };
   memcpy( st, in32, 32 );
   st[4]  = 0x06;
   st[16] = 0x8000000000000000ULL;
   KECCAK_F1600( uint64_t, st, K64_XOR, K64_ROL, K64_ANDN, K64_RC );
   memcpy( out32, st, 32 );
}

// Generic memory fill: the Argon2 library code selected at run time.
static void rin_fill_generic( rinhash_ctx_t *ctx )
{
   for ( uint32_t pass = 0; pass < RIN_T_COST; pass++ )
   for ( uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; slice++ )
   {
      argon2_position_t pos = { pass, 0, (uint8_t)slice, 0 };
      fill_segment_dispatch( &ctx->inst, pos );
   }
}

#if defined(__AVX2__)

#define RIN_FAST_FILL   "AVX2 (RinHash specific)"

/*
 * Memory fill written for the RinHash geometry: Argon2d, one lane of 64
 * blocks, two passes, version 1.3.
 *
 * new block = P( prev ^ ref ) ^ prev ^ ref, in the second pass also ^ old
 * block. P is a column round over eight groups of 256-bit words followed by
 * a row round. The usual code first builds prev ^ ref in a state array and a
 * copy of it, runs the rounds on the state, then combines both. Here the
 * column rounds take prev ^ ref straight from the two blocks and the row
 * rounds write the finished words to the new block, which removes two passes
 * over the 1 KiB state.
 */
#define RIN_LD( p, i )  _mm256_load_si256( (const __m256i*)(p) + (i) )

// Index of the reference block. With a single lane the rules of the
// Argon2 specification reduce to an area that depends only on the position:
// in the first pass every block written so far except the previous one, in
// the second pass the last three segments plus this one up to the previous
// block. 64 blocks: the modulo is a mask.
static inline __attribute__ ((always_inline))
uint32_t rin_ref_index( const uint64_t j, const uint32_t pass,
                        const uint32_t slice, const uint32_t index )
{
   const uint64_t j1 = (uint32_t)j;
   const uint64_t area = pass ? RIN_MEM_BLOCKS - RIN_SEGMENT + index - 1
                              : slice * RIN_SEGMENT + index - 1;
   const uint32_t start = ( pass && slice != ARGON2_SYNC_POINTS - 1 )
                          ? ( slice + 1 ) * RIN_SEGMENT : 0;
   const uint64_t x = ( j1 * j1 ) >> 32;
   const uint64_t z = area - 1 - ( ( area * x ) >> 32 );
   return ( start + (uint32_t)z ) & ( RIN_MEM_BLOCKS - 1 );
}

/*
 * BlaMka rounds over four independent sets of (a, b, c, d) rows at once.
 *
 * One set is a dependency chain: every step needs the result of the one
 * before it, and a multiply-add step takes six cycles. Written set after
 * set, as two calls of the usual two-set round, the second pair of sets
 * comes more instructions later than the CPU looks ahead when two threads
 * share the core, so it waits for the first pair although it does not
 * depend on it. Here each step is issued for all four sets before the next
 * step, which keeps four chains in flight.
 */
#define RIN_ROR32( x )  _mm256_shuffle_epi32( x, _MM_SHUFFLE( 2, 3, 0, 1 ) )
#define RIN_ROR24( x )  _mm256_shuffle_epi8( x, r24 )
#define RIN_ROR16( x )  _mm256_shuffle_epi8( x, r16 )
#define RIN_ROR63( x )  _mm256_xor_si256( _mm256_srli_epi64( x, 63 ), \
                                          _mm256_add_epi64( x, x ) )
#define RIN_SWAP64( x )  _mm256_shuffle_epi32( x, _MM_SHUFFLE( 1, 0, 3, 2 ) )
#define RIN_SWAP128( x ) _mm256_permute2x128_si256( x, x, 1 )

// x = x + y + 2 * lo32(x) * lo32(y)
#define RIN_MULADD( x, y ) \
do { \
   __m256i ml_ = _mm256_mul_epu32( x, y ); \
   ml_ = _mm256_add_epi64( ml_, ml_ ); \
   x = _mm256_add_epi64( x, _mm256_add_epi64( y, ml_ ) ); \
} while (0)

#define RIN_STEP4( X0, Y0, X1, Y1, X2, Y2, X3, Y3, Z0, Z1, Z2, Z3, ROR ) \
do { \
   RIN_MULADD( X0, Y0 );  RIN_MULADD( X1, Y1 ); \
   RIN_MULADD( X2, Y2 );  RIN_MULADD( X3, Y3 ); \
   Z0 = ROR( _mm256_xor_si256( Z0, X0 ) ); \
   Z1 = ROR( _mm256_xor_si256( Z1, X1 ) ); \
   Z2 = ROR( _mm256_xor_si256( Z2, X2 ) ); \
   Z3 = ROR( _mm256_xor_si256( Z3, X3 ) ); \
} while (0)

#define RIN_G4( A0, B0, C0, D0, A1, B1, C1, D1, \
                A2, B2, C2, D2, A3, B3, C3, D3 ) \
do { \
   RIN_STEP4( A0, B0, A1, B1, A2, B2, A3, B3, D0, D1, D2, D3, RIN_ROR32 ); \
   RIN_STEP4( C0, D0, C1, D1, C2, D2, C3, D3, B0, B1, B2, B3, RIN_ROR24 ); \
   RIN_STEP4( A0, B0, A1, B1, A2, B2, A3, B3, D0, D1, D2, D3, RIN_ROR16 ); \
   RIN_STEP4( C0, D0, C1, D1, C2, D2, C3, D3, B0, B1, B2, B3, RIN_ROR63 ); \
} while (0)

// Column round: a set is one 16-word group, each row in one register. The
// diagonals come from rotating rows b, c and d by one, two and three words.
#define RIN_ROWS_ROT( B, C, D, IB, ID ) \
do { \
   B = _mm256_permute4x64_epi64( B, IB ); \
   C = RIN_SWAP128( C ); \
   D = _mm256_permute4x64_epi64( D, ID ); \
} while (0)

#define RIN_ROUND_COLS( A0, B0, C0, D0, A1, B1, C1, D1, \
                        A2, B2, C2, D2, A3, B3, C3, D3 ) \
do { \
   RIN_G4( A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3 ); \
   RIN_ROWS_ROT( B0, C0, D0, _MM_SHUFFLE( 0, 3, 2, 1 ), _MM_SHUFFLE( 2, 1, 0, 3 ) ); \
   RIN_ROWS_ROT( B1, C1, D1, _MM_SHUFFLE( 0, 3, 2, 1 ), _MM_SHUFFLE( 2, 1, 0, 3 ) ); \
   RIN_ROWS_ROT( B2, C2, D2, _MM_SHUFFLE( 0, 3, 2, 1 ), _MM_SHUFFLE( 2, 1, 0, 3 ) ); \
   RIN_ROWS_ROT( B3, C3, D3, _MM_SHUFFLE( 0, 3, 2, 1 ), _MM_SHUFFLE( 2, 1, 0, 3 ) ); \
   RIN_G4( A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3 ); \
   RIN_ROWS_ROT( B0, C0, D0, _MM_SHUFFLE( 2, 1, 0, 3 ), _MM_SHUFFLE( 0, 3, 2, 1 ) ); \
   RIN_ROWS_ROT( B1, C1, D1, _MM_SHUFFLE( 2, 1, 0, 3 ), _MM_SHUFFLE( 0, 3, 2, 1 ) ); \
   RIN_ROWS_ROT( B2, C2, D2, _MM_SHUFFLE( 2, 1, 0, 3 ), _MM_SHUFFLE( 0, 3, 2, 1 ) ); \
   RIN_ROWS_ROT( B3, C3, D3, _MM_SHUFFLE( 2, 1, 0, 3 ), _MM_SHUFFLE( 0, 3, 2, 1 ) ); \
} while (0)

// Row round: two sets share their rows, half a row of each per 128-bit
// half of a register. Sets 0 and 1 are one such pair, sets 2 and 3 another.
#define RIN_PAIR_DIAG( B0, C0, D0, B1, C1, D1 ) \
do { \
   __m256i t1_ = _mm256_blend_epi32( B0, B1, 0xCC ); \
   __m256i t2_ = _mm256_blend_epi32( B0, B1, 0x33 ); \
   B1 = RIN_SWAP64( t1_ );  B0 = RIN_SWAP64( t2_ ); \
   t1_ = C0;  C0 = C1;  C1 = t1_; \
   t1_ = _mm256_blend_epi32( D0, D1, 0xCC ); \
   t2_ = _mm256_blend_epi32( D0, D1, 0x33 ); \
   D0 = RIN_SWAP64( t1_ );  D1 = RIN_SWAP64( t2_ ); \
} while (0)

#define RIN_PAIR_UNDIAG( B0, C0, D0, B1, C1, D1 ) \
do { \
   __m256i t1_ = _mm256_blend_epi32( B0, B1, 0xCC ); \
   __m256i t2_ = _mm256_blend_epi32( B0, B1, 0x33 ); \
   B0 = RIN_SWAP64( t1_ );  B1 = RIN_SWAP64( t2_ ); \
   t1_ = C0;  C0 = C1;  C1 = t1_; \
   t1_ = _mm256_blend_epi32( D0, D1, 0x33 ); \
   t2_ = _mm256_blend_epi32( D0, D1, 0xCC ); \
   D0 = RIN_SWAP64( t1_ );  D1 = RIN_SWAP64( t2_ ); \
} while (0)

#define RIN_ROUND_ROWS( A0, B0, C0, D0, A1, B1, C1, D1, \
                        A2, B2, C2, D2, A3, B3, C3, D3 ) \
do { \
   RIN_G4( A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3 ); \
   RIN_PAIR_DIAG( B0, C0, D0, B1, C1, D1 ); \
   RIN_PAIR_DIAG( B2, C2, D2, B3, C3, D3 ); \
   RIN_G4( A0, B0, C0, D0, A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3 ); \
   RIN_PAIR_UNDIAG( B0, C0, D0, B1, C1, D1 ); \
   RIN_PAIR_UNDIAG( B2, C2, D2, B3, C3, D3 ); \
} while (0)

// pos = pass * 64 + slice * 16 + index of the block being written.
static inline __attribute__ ((always_inline))
void rin_fill_block( block *mem, const uint32_t pos )
{
   const __m256i r24 = _mm256_setr_epi8(
                     3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10,
                     3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10 );
   const __m256i r16 = _mm256_setr_epi8(
                     2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9,
                     2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9 );
   const uint32_t pass = pos >> 6, slice = ( pos >> 4 ) & 3, index = pos & 15;
   const uint32_t curr = pos & ( RIN_MEM_BLOCKS - 1 );
   const uint64_t *prev = mem[ ( curr - 1 ) & ( RIN_MEM_BLOCKS - 1 ) ].v;
   const uint64_t *ref = mem[ rin_ref_index( prev[0], pass, slice, index ) ].v;
   uint64_t *next = mem[ curr ].v;
   const int with_xor = pass;
   __m256i t[32];

#define RIN_IN( k )  _mm256_xor_si256( RIN_LD( prev, k ), RIN_LD( ref, k ) )

   // Column rounds: the eight 16-word groups, four at a time.
   for ( int i = 0; i < 32; i += 16 )
   {
      __m256i x0  = RIN_IN( i+0  ), x1  = RIN_IN( i+1  );
      __m256i x2  = RIN_IN( i+2  ), x3  = RIN_IN( i+3  );
      __m256i x4  = RIN_IN( i+4  ), x5  = RIN_IN( i+5  );
      __m256i x6  = RIN_IN( i+6  ), x7  = RIN_IN( i+7  );
      __m256i x8  = RIN_IN( i+8  ), x9  = RIN_IN( i+9  );
      __m256i x10 = RIN_IN( i+10 ), x11 = RIN_IN( i+11 );
      __m256i x12 = RIN_IN( i+12 ), x13 = RIN_IN( i+13 );
      __m256i x14 = RIN_IN( i+14 ), x15 = RIN_IN( i+15 );
      RIN_ROUND_COLS( x0, x1, x2,  x3,  x4,  x5,  x6,  x7,
                      x8, x9, x10, x11, x12, x13, x14, x15 );
      t[i+0]  = x0;   t[i+1]  = x1;   t[i+2]  = x2;   t[i+3]  = x3;
      t[i+4]  = x4;   t[i+5]  = x5;   t[i+6]  = x6;   t[i+7]  = x7;
      t[i+8]  = x8;   t[i+9]  = x9;   t[i+10] = x10;  t[i+11] = x11;
      t[i+12] = x12;  t[i+13] = x13;  t[i+14] = x14;  t[i+15] = x15;
   }

#define RIN_OUT( a, k ) \
   do { \
      __m256i x_ = _mm256_xor_si256( a, RIN_IN( k ) ); \
      if ( with_xor ) x_ = _mm256_xor_si256( x_, RIN_LD( next, k ) ); \
      _mm256_store_si256( (__m256i*)next + (k), x_ ); \
   } while (0)

   // Row rounds: registers i, 4+i, ... 28+i hold two interleaved 16-word
   // groups; i and i+1 together are four sets.
   for ( int i = 0; i < 4; i += 2 )
   {
      __m256i a0 = t[i],     a1 = t[4+i],  b0 = t[8+i],  b1 = t[12+i];
      __m256i c0 = t[16+i],  c1 = t[20+i], d0 = t[24+i], d1 = t[28+i];
      __m256i a2 = t[i+1],   a3 = t[5+i],  b2 = t[9+i],  b3 = t[13+i];
      __m256i c2 = t[17+i],  c3 = t[21+i], d2 = t[25+i], d3 = t[29+i];
      RIN_ROUND_ROWS( a0, b0, c0, d0, a1, b1, c1, d1,
                      a2, b2, c2, d2, a3, b3, c3, d3 );
      RIN_OUT( a0, i );       RIN_OUT( a1, 4+i );
      RIN_OUT( b0, 8+i );     RIN_OUT( b1, 12+i );
      RIN_OUT( c0, 16+i );    RIN_OUT( c1, 20+i );
      RIN_OUT( d0, 24+i );    RIN_OUT( d1, 28+i );
      RIN_OUT( a2, i+1 );     RIN_OUT( a3, 5+i );
      RIN_OUT( b2, 9+i );     RIN_OUT( b3, 13+i );
      RIN_OUT( c2, 17+i );    RIN_OUT( c3, 21+i );
      RIN_OUT( d2, 25+i );    RIN_OUT( d3, 29+i );
   }
#undef RIN_OUT
#undef RIN_IN
}

static void rin_fill_fast( block *mem )
{
   // The first two blocks of the first pass are already there.
   for ( uint32_t pos = 2; pos < RIN_T_COST * RIN_MEM_BLOCKS; pos++ )
      rin_fill_block( mem, pos );
}

#endif // AVX2

// Cleared by the self test if the RinHash specific fill gives a wrong result.
static int rin_use_fast_fill = 1;

static inline void rin_fill( rinhash_ctx_t *ctx )
{
#ifdef RIN_FAST_FILL
   if ( likely( rin_use_fast_fill ) )
   {
      rin_fill_fast( ctx->memory );
      return;
   }
#endif
   rin_fill_generic( ctx );
}

// One hash, using the Argon2 library building blocks directly.
static void rin_hash_1way( rinhash_ctx_t *ctx, uint32_t nonce, void *out32 )
{
   uint32_t pwd[8] __attribute__ ((aligned (32)));
   uint8_t tag[32] __attribute__ ((aligned (32)));
   uint8_t blockhash[ ARGON2_PREHASH_SEED_LENGTH ];
   PROF_START;

   rin_blake3( ctx, nonce, pwd );
   PROF( RINHASH_PROF_BLAKE3 );

   ctx->argon.pwd = (uint8_t*)pwd;
   initial_hash( blockhash, &ctx->argon, Argon2_d );
   memset( blockhash + ARGON2_PREHASH_DIGEST_LENGTH, 0,
           ARGON2_PREHASH_SEED_LENGTH - ARGON2_PREHASH_DIGEST_LENGTH );
   PROF( RINHASH_PROF_H0 );

   fill_first_blocks( blockhash, &ctx->inst );
   PROF( RINHASH_PROF_FIRST_BLOCKS );

   rin_fill( ctx );
   PROF( RINHASH_PROF_FILL );

   blake2b_long( tag, 32, ctx->memory[ RIN_MEM_BLOCKS - 1 ].v,
                 ARGON2_BLOCK_SIZE );
   PROF( RINHASH_PROF_FINAL );

   rin_sha3_256( out32, tag );
   PROF( RINHASH_PROF_SHA3 );
}

#if RINHASH_BATCH > 1

/*
 * N-way code: N consecutive nonces are hashed together, one nonce per 64-bit
 * element of a vector register (N = 2 with SSE2, 4 with AVX2, 8 with
 * AVX-512).
 *
 * BLAKE2b and Keccak are sequences of 64-bit additions, XORs and rotations
 * with no data dependent branches or addresses, so N independent hashes run
 * in the instructions of one. That covers everything except the Argon2
 * memory fill, whose block addresses depend on the data; the fill runs per
 * nonce on the single 64 KiB buffer.
 */

#if defined(RINHASH_TEST_8WAY)

// Eight lanes on a pair of AVX2 registers, to test the 8-way code paths.
typedef struct { __m256i l, h; } vec_t;
#define V_TEST2( name, f ) \
   static inline vec_t name( vec_t a, vec_t b ) \
   { return (vec_t){ f( a.l, b.l ), f( a.h, b.h ) }; }
V_TEST2( v_add_test,  _mm256_add_epi64 )
V_TEST2( v_xor_test,  _mm256_xor_si256 )
V_TEST2( v_or_test,   _mm256_or_si256 )
V_TEST2( v_andn_test, _mm256_andnot_si256 )
static inline vec_t v_set1_test( uint64_t x )
{
   return (vec_t){ _mm256_set1_epi64x( (long long)x ),
                   _mm256_set1_epi64x( (long long)x ) };
}
static inline vec_t v_load_test( const void *p )
{
   return (vec_t){ _mm256_load_si256( (const __m256i*)p ),
                   _mm256_load_si256( (const __m256i*)p + 1 ) };
}
static inline void v_store_test( void *p, vec_t v )
{
   _mm256_store_si256( (__m256i*)p, v.l );
   _mm256_store_si256( (__m256i*)p + 1, v.h );
}
static inline vec_t v_slli_test( vec_t a, int n )
{
   return (vec_t){ _mm256_slli_epi64( a.l, n ), _mm256_slli_epi64( a.h, n ) };
}
static inline vec_t v_srli_test( vec_t a, int n )
{
   return (vec_t){ _mm256_srli_epi64( a.l, n ), _mm256_srli_epi64( a.h, n ) };
}
#define V_ADD( a, b )      v_add_test( a, b )
#define V_XOR( a, b )      v_xor_test( a, b )
#define V_OR( a, b )       v_or_test( a, b )
#define V_ANDN( a, b )     v_andn_test( a, b )
#define V_SET1( x )        v_set1_test( x )
#define V_ZERO             v_set1_test( 0 )
#define V_LOAD( p )        v_load_test( p )
#define V_STORE( p, v )    v_store_test( p, v )
#define V_SLLI( a, n )     v_slli_test( a, n )
#define V_SRLI( a, n )     v_srli_test( a, n )
#define V_ROL( a, n )      v_or_test( v_slli_test( a, n ), \
                                      v_srli_test( a, 64 - (n) ) )
#define V_ROR32( a )       V_ROL( a, 32 )
#define V_ROR24( a )       V_ROL( a, 40 )
#define V_ROR16( a )       V_ROL( a, 48 )
#define V_ROR63( a )       V_ROL( a, 1 )
#define V_NAME             "8-way (test build on AVX2)"

#elif RINHASH_BATCH == 8

typedef __m512i vec_t;
#define V_ADD( a, b )      _mm512_add_epi64( a, b )
#define V_XOR( a, b )      _mm512_xor_si512( a, b )
#define V_OR( a, b )       _mm512_or_si512( a, b )
#define V_ANDN( a, b )     _mm512_andnot_si512( a, b )
#define V_SET1( x )        _mm512_set1_epi64( (long long)(x) )
#define V_ZERO             _mm512_setzero_si512()
#define V_LOAD( p )        _mm512_load_si512( (const void*)(p) )
#define V_STORE( p, v )    _mm512_store_si512( (void*)(p), v )
#define V_SLLI( a, n )     _mm512_slli_epi64( a, n )
#define V_SRLI( a, n )     _mm512_srli_epi64( a, n )
#define V_ROL( a, n )      _mm512_rol_epi64( a, n )
#define V_ROR32( a )       _mm512_ror_epi64( a, 32 )
#define V_ROR24( a )       _mm512_ror_epi64( a, 24 )
#define V_ROR16( a )       _mm512_ror_epi64( a, 16 )
#define V_ROR63( a )       _mm512_ror_epi64( a, 63 )
#define V_NAME             "8-way AVX-512"

#elif RINHASH_BATCH == 4

typedef __m256i vec_t;
#define V_ADD( a, b )      _mm256_add_epi64( a, b )
#define V_XOR( a, b )      _mm256_xor_si256( a, b )
#define V_OR( a, b )       _mm256_or_si256( a, b )
#define V_ANDN( a, b )     _mm256_andnot_si256( a, b )
#define V_SET1( x )        _mm256_set1_epi64x( (long long)(x) )
#define V_ZERO             _mm256_setzero_si256()
#define V_LOAD( p )        _mm256_load_si256( (const __m256i*)(p) )
#define V_STORE( p, v )    _mm256_store_si256( (__m256i*)(p), v )
#define V_SLLI( a, n )     _mm256_slli_epi64( a, n )
#define V_SRLI( a, n )     _mm256_srli_epi64( a, n )
#define V_ROL( a, n )      V_OR( V_SLLI( a, n ), V_SRLI( a, 64 - (n) ) )
#define V_ROR32( a )       _mm256_shuffle_epi32( a, _MM_SHUFFLE( 2, 3, 0, 1 ) )
#define V_ROR24( a )       _mm256_shuffle_epi8( a, _mm256_setr_epi8( \
                              3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10, \
                              3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10 ) )
#define V_ROR16( a )       _mm256_shuffle_epi8( a, _mm256_setr_epi8( \
                              2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, \
                              2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9 ) )
#define V_ROR63( a )       V_OR( V_SRLI( a, 63 ), V_ADD( a, a ) )
#define V_NAME             "4-way AVX2"

#else

typedef __m128i vec_t;
#define V_ADD( a, b )      _mm_add_epi64( a, b )
#define V_XOR( a, b )      _mm_xor_si128( a, b )
#define V_OR( a, b )       _mm_or_si128( a, b )
#define V_ANDN( a, b )     _mm_andnot_si128( a, b )
#define V_SET1( x )        _mm_set1_epi64x( (long long)(x) )
#define V_ZERO             _mm_setzero_si128()
#define V_LOAD( p )        _mm_load_si128( (const __m128i*)(p) )
#define V_STORE( p, v )    _mm_store_si128( (__m128i*)(p), v )
#define V_SLLI( a, n )     _mm_slli_epi64( a, n )
#define V_SRLI( a, n )     _mm_srli_epi64( a, n )
#define V_ROL( a, n )      V_OR( V_SLLI( a, n ), V_SRLI( a, 64 - (n) ) )
#define V_ROR32( a )       _mm_shuffle_epi32( a, _MM_SHUFFLE( 2, 3, 0, 1 ) )
#if defined(__SSSE3__)
#define V_ROR24( a )       _mm_shuffle_epi8( a, _mm_setr_epi8( \
                              3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10 ) )
#define V_ROR16( a )       _mm_shuffle_epi8( a, _mm_setr_epi8( \
                              2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9 ) )
#define V_NAME             "2-way SSSE3"
#else
#define V_ROR24( a )       V_ROL( a, 40 )
#define V_ROR16( a )       V_ROL( a, 48 )
#define V_NAME             "2-way SSE2"
#endif
#define V_ROR63( a )       V_OR( V_SRLI( a, 63 ), V_ADD( a, a ) )

#endif

#define B2B_IV0   0x6a09e667f3bcc908ULL
#define B2B_IV1   0xbb67ae8584caa73bULL
#define B2B_IV2   0x3c6ef372fe94f82bULL
#define B2B_IV3   0xa54ff53a5f1d36f1ULL
#define B2B_IV4   0x510e527fade682d1ULL
#define B2B_IV5   0x9b05688c2b3e6c1fULL
#define B2B_IV6   0x1f83d9abfb41bd6bULL
#define B2B_IV7   0x5be0cd19137e2179ULL
// IV0 xor the parameter block of an unkeyed hash with the given digest size
#define B2B_H0( outlen )   ( B2B_IV0 ^ 0x01010000ULL ^ (uint64_t)(outlen) )

#define B2B_G( a, b, c, d, x, y ) \
do { \
   a = V_ADD( V_ADD( a, b ), x ); \
   d = V_ROR32( V_XOR( d, a ) ); \
   c = V_ADD( c, d ); \
   b = V_ROR24( V_XOR( b, c ) ); \
   a = V_ADD( V_ADD( a, b ), y ); \
   d = V_ROR16( V_XOR( d, a ) ); \
   c = V_ADD( c, d ); \
   b = V_ROR63( V_XOR( b, c ) ); \
} while (0)

// One round with the message word order written out, so that words known
// to be zero cost nothing.
#define B2B_ROUND( s0, s1, s2, s3, s4, s5, s6, s7, \
                   s8, s9, s10, s11, s12, s13, s14, s15 ) \
do { \
   B2B_G( v0, v4, v8,  v12, m##s0,  m##s1  ); \
   B2B_G( v1, v5, v9,  v13, m##s2,  m##s3  ); \
   B2B_G( v2, v6, v10, v14, m##s4,  m##s5  ); \
   B2B_G( v3, v7, v11, v15, m##s6,  m##s7  ); \
   B2B_G( v0, v5, v10, v15, m##s8,  m##s9  ); \
   B2B_G( v1, v6, v11, v12, m##s10, m##s11 ); \
   B2B_G( v2, v7, v8,  v13, m##s12, m##s13 ); \
   B2B_G( v3, v4, v9,  v14, m##s14, m##s15 ); \
} while (0)

// BLAKE2b compression of one 128-byte block for N hashes. t is the byte
// count including this block, the same for every hash.
static inline __attribute__ ((always_inline))
void b2b_compress_nway( vec_t h[8],
        vec_t m0, vec_t m1, vec_t m2,  vec_t m3,  vec_t m4,  vec_t m5,
        vec_t m6, vec_t m7, vec_t m8,  vec_t m9,  vec_t m10, vec_t m11,
        vec_t m12, vec_t m13, vec_t m14, vec_t m15,
        const uint64_t t, const int last )
{
   vec_t v0 = h[0], v1 = h[1], v2 = h[2], v3 = h[3];
   vec_t v4 = h[4], v5 = h[5], v6 = h[6], v7 = h[7];
   vec_t v8  = V_SET1( B2B_IV0 ),     v9  = V_SET1( B2B_IV1 );
   vec_t v10 = V_SET1( B2B_IV2 ),     v11 = V_SET1( B2B_IV3 );
   vec_t v12 = V_SET1( B2B_IV4 ^ t ), v13 = V_SET1( B2B_IV5 );
   vec_t v14 = V_SET1( last ? ~B2B_IV6 : B2B_IV6 );
   vec_t v15 = V_SET1( B2B_IV7 );

   B2B_ROUND(  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 );
   B2B_ROUND( 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 );
   B2B_ROUND( 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 );
   B2B_ROUND(  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 );
   B2B_ROUND(  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 );
   B2B_ROUND(  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 );
   B2B_ROUND( 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 );
   B2B_ROUND( 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 );
   B2B_ROUND(  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 );
   B2B_ROUND( 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 );
   B2B_ROUND(  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 );
   B2B_ROUND( 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 );

   h[0] = V_XOR( h[0], V_XOR( v0, v8  ) );
   h[1] = V_XOR( h[1], V_XOR( v1, v9  ) );
   h[2] = V_XOR( h[2], V_XOR( v2, v10 ) );
   h[3] = V_XOR( h[3], V_XOR( v3, v11 ) );
   h[4] = V_XOR( h[4], V_XOR( v4, v12 ) );
   h[5] = V_XOR( h[5], V_XOR( v5, v13 ) );
   h[6] = V_XOR( h[6], V_XOR( v6, v14 ) );
   h[7] = V_XOR( h[7], V_XOR( v7, v15 ) );
}

static inline __attribute__ ((always_inline))
void b2b_init_nway( vec_t h[8], const unsigned outlen )
{
   h[0] = V_SET1( B2B_H0( outlen ) );
   h[1] = V_SET1( B2B_IV1 );  h[2] = V_SET1( B2B_IV2 );
   h[3] = V_SET1( B2B_IV3 );  h[4] = V_SET1( B2B_IV4 );
   h[5] = V_SET1( B2B_IV5 );  h[6] = V_SET1( B2B_IV6 );
   h[7] = V_SET1( B2B_IV7 );
}

// v = BLAKE2b-512( v ) for a 64-byte v: one block whose upper half is zero.
// noinline: one copy of the twelve rounds stays hot in the uop cache.
static __attribute__ ((noinline))
void b2b_hash64_nway( vec_t v[8] )
{
   vec_t h[8];
   b2b_init_nway( h, 64 );
   b2b_compress_nway( h, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                      V_ZERO, V_ZERO, V_ZERO, V_ZERO,
                      V_ZERO, V_ZERO, V_ZERO, V_ZERO, 64, 1 );
   v[0] = h[0];  v[1] = h[1];  v[2] = h[2];  v[3] = h[3];
   v[4] = h[4];  v[5] = h[5];  v[6] = h[6];  v[7] = h[7];
}

// A full 128-byte block read from memory.
static __attribute__ ((noinline))
void b2b_block_nway( vec_t h[8], const vec_t m[16], const uint64_t t,
                     const int last )
{
   if ( last )
      b2b_compress_nway( h, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                         m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15],
                         t, 1 );
   else
      b2b_compress_nway( h, m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                         m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15],
                         t, 0 );
}

// Argon2 H0 input with a zero password: every field is fixed for RinHash
// except the 32 password bytes at offset 28.
static uint64_t rin_h0_template[16];
#define RIN_H0_LEN   83

static void rin_h0_template_init( void )
{
   uint8_t b[128] = { 0 };
   const uint32_t f[7] = { 1, 32, RIN_MEM_BLOCKS, RIN_T_COST,
                           ARGON2_VERSION_13, Argon2_d, 32 };
   const uint32_t saltlen = RIN_SALTLEN;
   // lanes, tag length, m_cost, t_cost, version, type, password length
   memcpy( b, f, 28 );
   // password at 28..59
   memcpy( b + 60, &saltlen, 4 );
   memcpy( b + 64, RIN_SALT, RIN_SALTLEN );
   // secret length and associated data length at 75..82 are zero
   memcpy( rin_h0_template, b, 128 );
}

static void rin_hash_nway( rinhash_ctx_t *ctx, uint32_t first_nonce,
                           void *out )
{
   uint32_t pwd[ RINHASH_BATCH ][8] __attribute__ ((aligned (64)));
   uint64_t tmp[8][ RINHASH_BATCH ] __attribute__ ((aligned (64)));
   vec_t h0[8], h[8], m[16], w, prev;
   PROF_START;

   // 1. BLAKE3
   for ( int n = 0; n < RINHASH_BATCH; n++ )
      rin_blake3( ctx, first_nonce + n, pwd[n] );
   PROF( RINHASH_PROF_BLAKE3 );

   // 2. H0 = BLAKE2b-512( Argon2 parameters, password, salt )
   for ( int n = 0; n < RINHASH_BATCH; n++ )
   {
      tmp[0][n] = rin_h0_template[3] | ( (uint64_t)pwd[n][0] << 32 );
      tmp[1][n] = pwd[n][1] | ( (uint64_t)pwd[n][2] << 32 );
      tmp[2][n] = pwd[n][3] | ( (uint64_t)pwd[n][4] << 32 );
      tmp[3][n] = pwd[n][5] | ( (uint64_t)pwd[n][6] << 32 );
      tmp[4][n] = pwd[n][7] | rin_h0_template[7];
   }
   b2b_init_nway( h0, 64 );
   b2b_compress_nway( h0, V_SET1( rin_h0_template[0] ),
                      V_SET1( rin_h0_template[1] ),
                      V_SET1( rin_h0_template[2] ),
                      V_LOAD( tmp[0] ), V_LOAD( tmp[1] ), V_LOAD( tmp[2] ),
                      V_LOAD( tmp[3] ), V_LOAD( tmp[4] ),
                      V_SET1( rin_h0_template[8] ),
                      V_SET1( rin_h0_template[9] ),
                      V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO,
                      RIN_H0_LEN, 1 );
   PROF( RINHASH_PROF_H0 );

   // 3. Blocks 0 and 1 of every nonce: H'( H0 || block index || lane 0 ) with
   // 1024 bytes of output. H' is a chain of BLAKE2b-512 hashes, each over the
   // previous digest, emitting 32 bytes per link and 64 at the end. The
   // first link hashes le32(1024) || H0 || le32(index) || le32(0).
   for ( int blk = 0; blk < 2; blk++ )
   {
      uint64_t (*dst)[ RINHASH_BATCH ] = ctx->first[ blk ];

      b2b_init_nway( h, 64 );
      b2b_compress_nway( h,
            V_OR( V_SLLI( h0[0], 32 ), V_SET1( ARGON2_BLOCK_SIZE ) ),
            V_OR( V_SRLI( h0[0], 32 ), V_SLLI( h0[1], 32 ) ),
            V_OR( V_SRLI( h0[1], 32 ), V_SLLI( h0[2], 32 ) ),
            V_OR( V_SRLI( h0[2], 32 ), V_SLLI( h0[3], 32 ) ),
            V_OR( V_SRLI( h0[3], 32 ), V_SLLI( h0[4], 32 ) ),
            V_OR( V_SRLI( h0[4], 32 ), V_SLLI( h0[5], 32 ) ),
            V_OR( V_SRLI( h0[5], 32 ), V_SLLI( h0[6], 32 ) ),
            V_OR( V_SRLI( h0[6], 32 ), V_SLLI( h0[7], 32 ) ),
            V_OR( V_SRLI( h0[7], 32 ), V_SET1( (uint64_t)blk << 32 ) ),
            V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO,
            ARGON2_PREHASH_SEED_LENGTH + 4, 1 );

      for ( int i = 0; i < 4; i++ ) V_STORE( dst[i], h[i] );
      for ( int link = 1; link < 30; link++ )
      {
         b2b_hash64_nway( h );
         for ( int i = 0; i < 4; i++ ) V_STORE( dst[ 4*link + i ], h[i] );
      }
      b2b_hash64_nway( h );
      for ( int i = 0; i < 8; i++ ) V_STORE( dst[ 120 + i ], h[i] );
   }
   PROF( RINHASH_PROF_FIRST_BLOCKS );

   // 4. Memory fill, one nonce after the other on the same buffer.
   for ( int n = 0; n < RINHASH_BATCH; n++ )
   {
      uint64_t *b0 = ctx->memory[0].v, *b1 = ctx->memory[1].v;
      const uint64_t *bl = ctx->memory[ RIN_MEM_BLOCKS - 1 ].v;
      for ( int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++ )
      {
         b0[i] = ctx->first[0][i][n];
         b1[i] = ctx->first[1][i][n];
      }
      rin_fill( ctx );
      for ( int i = 0; i < ARGON2_QWORDS_IN_BLOCK; i++ )
         ctx->last[i][n] = bl[i];
   }
   PROF( RINHASH_PROF_FILL );

   // 5. Tag = BLAKE2b-256( le32(32) || last block ): 1028 bytes, eight full
   // blocks and four more bytes. The 4-byte prefix shifts every word of the
   // block by half a word.
   b2b_init_nway( h, 32 );
   prev = V_SET1( 32 );
   for ( int blk = 0; blk < 8; blk++ )
   {
      for ( int i = 0; i < 16; i++ )
      {
         w = V_LOAD( ctx->last[ 16*blk + i ] );
         m[i] = V_OR( prev, V_SLLI( w, 32 ) );
         prev = V_SRLI( w, 32 );
      }
      b2b_block_nway( h, m, 128 * ( blk + 1 ), 0 );
   }
   b2b_compress_nway( h, prev, V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO,
                      V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO, V_ZERO,
                      V_ZERO, V_ZERO, ARGON2_BLOCK_SIZE + 4, 1 );
   PROF( RINHASH_PROF_FINAL );

   // 6. SHA3-256 of the 32-byte tag
   {
      vec_t st[25];
      st[0] = h[0];  st[1] = h[1];  st[2] = h[2];  st[3] = h[3];
      st[4] = V_SET1( 0x06 );
      for ( int i = 5; i < 25; i++ ) st[i] = V_ZERO;
      st[16] = V_SET1( 0x8000000000000000ULL );
      KECCAK_F1600( vec_t, st, V_XOR, V_ROL, V_ANDN, V_SET1 );
      for ( int i = 0; i < 4; i++ ) V_STORE( tmp[i], st[i] );
   }
   for ( int n = 0; n < RINHASH_BATCH; n++ )
   {
      uint64_t r[4] = { tmp[0][n], tmp[1][n], tmp[2][n], tmp[3][n] };
      memcpy( (uint8_t*)out + 32*n, r, 32 );
   }
   PROF( RINHASH_PROF_SHA3 );
}

#endif // RINHASH_BATCH > 1

// Cleared by the self test if the N-way code gives a wrong result.
static int rin_use_nway = ( RINHASH_BATCH > 1 );

void rinhash_hash_nonce( rinhash_ctx_t *ctx, uint32_t nonce, void *out32 )
{
   rin_hash_1way( ctx, nonce, out32 );
}

void rinhash_hash_batch( rinhash_ctx_t *ctx, uint32_t first_nonce, void *out )
{
#if RINHASH_BATCH > 1
   if ( likely( rin_use_nway ) )
   {
      rin_hash_nway( ctx, first_nonce, out );
      return;
   }
#endif
   for ( int i = 0; i < RINHASH_BATCH; i++ )
      rin_hash_1way( ctx, first_nonce + i, (uint8_t*)out + 32 * i );
}

void rinhash_hash80( rinhash_ctx_t *ctx, const void *header, void *out32 )
{
   uint32_t nonce;
   memcpy( &nonce, (const uint8_t*)header + 76, 4 );
   rinhash_set_header( ctx, header );
   rin_hash_1way( ctx, nonce, out32 );
}

/*
 * Self test
 */

// Block header and its RinHash, also used by other RinHash implementations.
static const uint8_t test_header[80] =
{
   0x00,0x00,0x00,0x20,0xa1,0x4a,0xe1,0x41,0x58,0x5f,0x60,0x30,0x1e,0xd2,0xda,
   0xfa,0xb2,0xa6,0x50,0x45,0xd5,0xd1,0x2a,0xfa,0x0b,0x23,0x22,0x33,0xa0,0x06,
   0x5f,0xa7,0x01,0x00,0x00,0x00,0xa7,0x10,0xd5,0x40,0x64,0xe2,0xa2,0x72,0x8c,
   0x2f,0x43,0x5a,0x8a,0x75,0x93,0xda,0xcd,0x26,0x1d,0x94,0x18,0xd1,0xb6,0x9d,
   0xd0,0xbe,0x6e,0xcd,0x99,0x1c,0x78,0xef,0xa4,0x13,0x45,0x68,0x38,0xdd,0x01,
   0x1d,0x00,0x00,0x00,0x00
};
static const uint8_t test_hash[32] =
{
   0xa5,0x7b,0x72,0x38,0xb9,0xab,0x81,0x8a,0x25,0xcc,0xcd,0xb5,0x68,0x34,0x07,
   0x20,0x76,0x37,0xbe,0x79,0xcc,0x5c,0x58,0x1d,0x8b,0x16,0x72,0xb6,0xda,0x89,
   0x72,0x92
};

// Plain library call chain: BLAKE3 API, argon2d_ctx, generic SHA3 padding.
static int rin_hash_reference( const void *header, void *out32 )
{
   blake3_hasher h;
   uint8_t pwd[32], tag[32];
   argon2_context a = { 0 };

   blake3_hasher_init( &h );
   blake3_hasher_update( &h, header, 80 );
   blake3_hasher_finalize( &h, pwd, 32 );

   a.out = tag;       a.outlen = 32;
   a.pwd = pwd;       a.pwdlen = 32;
   a.salt = (uint8_t*)RIN_SALT;   a.saltlen = RIN_SALTLEN;
   a.t_cost = RIN_T_COST;         a.m_cost = RIN_MEM_BLOCKS;
   a.lanes = 1;       a.threads = 1;
   a.version = ARGON2_VERSION_13; a.flags = ARGON2_DEFAULT_FLAGS;
   if ( argon2d_ctx( &a ) != ARGON2_OK ) return -1;

   rin_sha3_256( out32, tag );
   return 0;
}

int rinhash_selftest( void )
{
   uint8_t header[80], ref[32], one[32], out[ 32 * RINHASH_BATCH ];
   rinhash_ctx_t *ctx;
   int bad_1way = 0, bad_nway = 0;

   // Known answer with the plain C Argon2 fill.
   argon2_set_impl( ARGON2_IMPL_REF );
   if ( rin_hash_reference( test_header, ref ) || memcmp( ref, test_hash, 32 ) )
   {
      fprintf( stderr, "RinHash self test: reference path is wrong\n" );
      return -1;
   }
   argon2_set_impl( RIN_IMPL_AUTO );

   ctx = rinhash_ctx_new();
   if ( !ctx ) return -1;

   memcpy( header, test_header, 80 );
   for ( uint32_t round = 0; round < 8; round++ )
   {
      uint32_t nonce;
      header[0] = (uint8_t)round;
      header[70] = (uint8_t)( 3 * round );
      memcpy( &nonce, header + 76, 4 );
      nonce += 1000003u * round;

      rinhash_set_header( ctx, header );
      rinhash_hash_batch( ctx, nonce, out );
      for ( int i = 0; i < RINHASH_BATCH; i++ )
      {
         uint32_t n = nonce + i;
         memcpy( header + 76, &n, 4 );
         argon2_set_impl( ARGON2_IMPL_REF );
         rin_hash_reference( header, ref );
         argon2_set_impl( RIN_IMPL_AUTO );
         rinhash_hash_nonce( ctx, n, one );
         if ( memcmp( one, ref, 32 ) )           bad_1way = 1;
         if ( memcmp( out + 32 * i, ref, 32 ) )  bad_nway = 1;
      }
   }
   rinhash_ctx_free( ctx );

   if ( bad_1way )
   {
      fprintf( stderr, "RinHash self test: the optimized Argon2 fill is "
               "wrong, using the reference code\n" );
      argon2_set_impl( ARGON2_IMPL_REF );
      rin_use_fast_fill = 0;
      rin_use_nway = 0;
      return 1;
   }
   if ( bad_nway )
   {
      fprintf( stderr, "RinHash self test: the N-way code is wrong, "
               "hashing one nonce at a time\n" );
      rin_use_nway = 0;
      return 1;
   }
   return 0;
}

const char *rinhash_impl_name( void )
{
   static char name[96];
   const char *fill = argon2_get_impl_name();
#ifdef RIN_FAST_FILL
   if ( rin_use_fast_fill ) fill = RIN_FAST_FILL;
#endif
#if RINHASH_BATCH > 1
   if ( rin_use_nway )
      snprintf( name, sizeof(name), V_NAME " BLAKE2b and SHA3, Argon2 fill %s",
                fill );
   else
#endif
   snprintf( name, sizeof(name), "1-way, Argon2 fill %s", fill );
   return name;
}
