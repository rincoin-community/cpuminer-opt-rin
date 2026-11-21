#ifndef RINHASH_CORE_H__
#define RINHASH_CORE_H__ 1

/*
 * RinHash core: BLAKE3(80-byte header) -> Argon2d -> SHA3-256
 *
 * Argon2d parameters: version 1.3, t_cost 2, m_cost 64 KiB, 1 lane,
 * salt "RinCoinSalt", 32-byte tag.
 *
 * This file has no dependency on the miner framework so that the hash can be
 * tested and benchmarked on its own (see test/rinhash_bench.c).
 */

#include <stdint.h>
#include <stddef.h>

/* Number of consecutive nonces hashed by one rinhash_hash_batch call. */
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__)
#define RINHASH_BATCH 8
#elif defined(__AVX2__)
#define RINHASH_BATCH 4
#else
#define RINHASH_BATCH 1
#endif

typedef struct rinhash_ctx rinhash_ctx_t;

/* Once per process, before any other call. */
void rinhash_global_init( void );

/* Per-thread context. Allocate it in the thread that uses it: the memory is
   placed on that thread's NUMA node by first touch. */
rinhash_ctx_t *rinhash_ctx_new( void );
void rinhash_ctx_free( rinhash_ctx_t *ctx );

/* Load an 80-byte block header. Bytes 0..63 are hashed once here, bytes
   64..75 are kept, bytes 76..79 are the nonce and are supplied per hash. */
void rinhash_set_header( rinhash_ctx_t *ctx, const void *header );

/* Hash the loaded header with the nonce stored in host byte order at bytes
   76..79, exactly as "pdata[19] = nonce" would store it. */
void rinhash_hash_nonce( rinhash_ctx_t *ctx, uint32_t nonce, void *out32 );

/* Hash the loaded header for nonces first_nonce .. first_nonce +
   RINHASH_BATCH - 1. out receives RINHASH_BATCH consecutive 32-byte hashes. */
void rinhash_hash_batch( rinhash_ctx_t *ctx, uint32_t first_nonce, void *out );

/* One complete 80-byte header, any nonce. */
void rinhash_hash80( rinhash_ctx_t *ctx, const void *header, void *out32 );

/* Checks the optimized paths against a known answer and against the
   reference Argon2 code. Returns 0 if everything matches. If only an
   optimized path is wrong it is switched off, a message is printed and 1 is
   returned; -1 means that even the reference path is wrong. */
int rinhash_selftest( void );

/* Human readable description of the code paths in use. */
const char *rinhash_impl_name( void );

/* Optional per-stage cycle counters, compiled in with -DRINHASH_PROFILE. */
enum
{
   RINHASH_PROF_BLAKE3,
   RINHASH_PROF_H0,
   RINHASH_PROF_FIRST_BLOCKS,
   RINHASH_PROF_FILL,
   RINHASH_PROF_FINAL,
   RINHASH_PROF_SHA3,
   RINHASH_PROF_N
};
#ifdef RINHASH_PROFILE
extern __thread uint64_t rinhash_prof[ RINHASH_PROF_N ];
#endif

#endif
