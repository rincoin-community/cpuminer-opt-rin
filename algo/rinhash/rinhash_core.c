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

#define RIN_MEM_BLOCKS  64
#define RIN_T_COST      2
#define RIN_SEGMENT     ( RIN_MEM_BLOCKS / ARGON2_SYNC_POINTS )
#define RIN_SALT        "RinCoinSalt"
#define RIN_SALTLEN     11
// argon2_set_impl: any value without an implementation selects the best one
#define RIN_IMPL_AUTO   ARGON2_IMPL_COUNT

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
};

extern int FLAG_clear_internal_memory;

void rinhash_global_init( void )
{
   argon2_dispatch_init();
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

static inline void rin_fill( rinhash_ctx_t *ctx )
{
   for ( uint32_t pass = 0; pass < RIN_T_COST; pass++ )
   for ( uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; slice++ )
   {
      argon2_position_t pos = { pass, 0, (uint8_t)slice, 0 };
      fill_segment_dispatch( &ctx->inst, pos );
   }
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

void rinhash_hash_nonce( rinhash_ctx_t *ctx, uint32_t nonce, void *out32 )
{
   rin_hash_1way( ctx, nonce, out32 );
}

void rinhash_hash_batch( rinhash_ctx_t *ctx, uint32_t first_nonce, void *out )
{
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
   uint8_t header[80], ref[32], out[ 32 * RINHASH_BATCH ];
   rinhash_ctx_t *ctx;
   int rc = 0;

   // Reference result with the plain C Argon2 fill.
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
   for ( uint32_t round = 0; round < 8 && !rc; round++ )
   {
      uint32_t nonce;
      header[0] = (uint8_t)round;
      header[70] = (uint8_t)( 3 * round );
      memcpy( &nonce, header + 76, 4 );
      nonce += 1000003u * round;

      rinhash_set_header( ctx, header );
      rinhash_hash_batch( ctx, nonce, out );
      for ( int i = 0; i < RINHASH_BATCH && !rc; i++ )
      {
         uint8_t single[32];
         uint32_t n = nonce + i;
         memcpy( header + 76, &n, 4 );
         argon2_set_impl( ARGON2_IMPL_REF );
         rin_hash_reference( header, ref );
         argon2_set_impl( RIN_IMPL_AUTO );
         rinhash_hash_nonce( ctx, n, single );
         if ( memcmp( single, ref, 32 ) || memcmp( out + 32 * i, ref, 32 ) )
            rc = 1;
      }
   }
   rinhash_ctx_free( ctx );

   if ( rc )
   {
      fprintf( stderr, "RinHash self test: optimized path (%s) is wrong, "
                       "using the reference code\n", rinhash_impl_name() );
      argon2_set_impl( ARGON2_IMPL_REF );
   }
   return rc;
}

const char *rinhash_impl_name( void )
{
   static char name[64];
   snprintf( name, sizeof(name), "1-way, Argon2 fill %s",
             argon2_get_impl_name() );
   return name;
}
