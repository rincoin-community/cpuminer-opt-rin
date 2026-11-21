/*
 * RinHash verification and benchmark, independent of the miner framework.
 *
 * Build with test/build_rinhash_bench.sh after "make" (it links the miner's
 * own object files), then:
 *
 *    test/rinhash_bench [hashes]         verify, then time one thread
 *    taskset -c N test/rinhash_bench     pin it to a CPU
 *
 * rinhash_bench_prof is the same program with per-stage cycle counters.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#include "../algo/rinhash/rinhash_core.h"
#include "../algo/rinhash/blake3/blake3.h"
#include "../algo/rinhash/argon2/include/argon2.h"
#include "../algo/verthash/tiny_sha3/sha3.h"

static double now_cpu( void )
{
   struct timespec ts;
   clock_gettime( CLOCK_THREAD_CPUTIME_ID, &ts );
   return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double now_mono( void )
{
   struct timespec ts;
   clock_gettime( CLOCK_MONOTONIC, &ts );
   return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Independent reference: public BLAKE3 and Argon2 APIs, generic SHA3.
static void reference( const uint8_t *header, uint8_t *out )
{
   blake3_hasher h;
   uint8_t pwd[32], tag[32];

   blake3_hasher_init( &h );
   blake3_hasher_update( &h, header, 80 );
   blake3_hasher_finalize( &h, pwd, 32 );
   if ( argon2d_hash_raw( 2, 64, 1, pwd, 32, "RinCoinSalt", 11, tag, 32 ) )
   {
      printf( "FAIL: argon2d_hash_raw\n" );
      exit( 1 );
   }
   sha3( tag, 32, out, 32 );
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t rng( void )
{
   rng_state ^= rng_state << 13;
   rng_state ^= rng_state >> 7;
   rng_state ^= rng_state << 17;
   return rng_state;
}

static int verify( rinhash_ctx_t *ctx, int headers )
{
   uint8_t header[80], ref[32], one[32], batch[ 32 * RINHASH_BATCH ];
   int bad = 0;

   for ( int k = 0; k < headers && !bad; k++ )
   {
      uint32_t nonce;
      for ( int i = 0; i < 80; i += 8 )
      {
         uint64_t r = rng();
         memcpy( header + i, &r, 8 );
      }
      memcpy( &nonce, header + 76, 4 );
      if ( k == 1 ) nonce = 0xfffffffeU;        // nonce wrap inside a batch

      rinhash_set_header( ctx, header );
      rinhash_hash_batch( ctx, nonce, batch );
      for ( int i = 0; i < RINHASH_BATCH; i++ )
      {
         uint32_t n = nonce + i;
         memcpy( header + 76, &n, 4 );
         reference( header, ref );
         rinhash_hash_nonce( ctx, n, one );
         if ( memcmp( one, ref, 32 ) ) { bad = 1; printf( "FAIL: single hash, header %d\n", k ); }
         if ( memcmp( batch + 32*i, ref, 32 ) ) { bad = 1; printf( "FAIL: batch lane %d, header %d\n", i, k ); }
         rinhash_hash80( ctx, header, one );
         if ( memcmp( one, ref, 32 ) ) { bad = 1; printf( "FAIL: hash80, header %d\n", k ); }
      }
   }
   return bad;
}

int main( int argc, char **argv )
{
   static const char *hex =
      "00000020a14ae141585f60301ed2dafab2a65045d5d12afa0b232233a0065fa7"
      "01000000a710d54064e2a2728c2f435a8a7593dacd261d9418d1b69dd0be6ecd9"
      "91c78efa413456838dd011d00000000";
   static const char *expect =
      "a57b7238b9ab818a25cccdb5683407207637be79cc5c581d8b1672b6da897292";
   int n = argc > 1 ? atoi( argv[1] ) : 40000;
   uint8_t header[80], out[ 32 * RINHASH_BATCH ];
   char hexout[65];
   rinhash_ctx_t *ctx;
   int st, bad;

   for ( int i = 0; i < 80; i++ )
   {
      unsigned v;
      sscanf( hex + 2*i, "%2x", &v );
      header[i] = v;
   }

   rinhash_global_init();
   st = rinhash_selftest();
   ctx = rinhash_ctx_new();
   rinhash_hash80( ctx, header, out );
   for ( int i = 0; i < 32; i++ ) sprintf( hexout + 2*i, "%02x", out[i] );
   bad = strcmp( hexout, expect ) != 0;
   bad |= verify( ctx, 200 );
   printf( "impl=%s  batch=%d  selftest=%d  hash=%s  verify=%s\n",
           rinhash_impl_name(), RINHASH_BATCH, st, hexout, bad ? "FAIL" : "ok" );
   if ( bad || st ) return 1;

   n -= n % RINHASH_BATCH;
   rinhash_set_header( ctx, header );
   for ( int i = 0; i < 2000; i += RINHASH_BATCH )
      rinhash_hash_batch( ctx, i, out );

#ifdef RINHASH_PROFILE
   memset( rinhash_prof, 0, sizeof(rinhash_prof) );
#endif
   double m0 = now_mono(), c0 = now_cpu();
   uint64_t r0 = __rdtsc();
   for ( int i = 0; i < n; i += RINHASH_BATCH )
      rinhash_hash_batch( ctx, 5000 + i, out );
   uint64_t r1 = __rdtsc();
   double c1 = now_cpu(), m1 = now_mono();

   printf( "production path: %.2f us/hash (thread CPU time)  = %.0f H/s\n",
           ( c1 - c0 ) * 1e6 / n, n / ( c1 - c0 ) );
#ifdef RINHASH_PROFILE
   {
      static const char *name[ RINHASH_PROF_N ] =
      {
         "blake3 (16 bytes + finalize)", "argon2 H0 (blake2b)",
         "argon2 first 2 blocks (blake2b_long x2)", "argon2 fill (126 blocks)",
         "argon2 final blake2b_long(1024->32)", "sha3-256"
      };
      double ns_per_tick = ( m1 - m0 ) * 1e9 / (double)( r1 - r0 );
      uint64_t tot = 0;
      for ( int s = 0; s < RINHASH_PROF_N; s++ ) tot += rinhash_prof[s];
      for ( int s = 0; s < RINHASH_PROF_N; s++ )
         printf( "  %-42s %7.2f us  %5.1f%%\n", name[s],
                 rinhash_prof[s] * ns_per_tick / n / 1000.0,
                 100.0 * rinhash_prof[s] / tot );
   }
#else
   (void)m0; (void)m1; (void)r0; (void)r1;
#endif
   return 0;
}
