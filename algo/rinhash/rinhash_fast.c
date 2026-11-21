/*
 * RinHash for cpuminer: miner glue around the hash in rinhash_core.c.
 *
 * Algorithm: BLAKE3(80-byte header) -> Argon2d -> SHA3-256
 * Argon2d: t_cost=2, m_cost=64, lanes=1, salt="RinCoinSalt", 32-byte tag.
 */

#include "rinhash-gate.h"
#include "rinhash_core.h"
#include "miner.h"
#include <string.h>

// One context per mining thread, created by that thread.
static __thread rinhash_ctx_t *rin_ctx = NULL;

static inline bool rin_ctx_ready( void )
{
   if ( unlikely( !rin_ctx ) ) rin_ctx = rinhash_ctx_new();
   return rin_ctx != NULL;
}

void rinhash_fast( void *state, const void *input )
{
   if ( unlikely( !rin_ctx_ready() ) )
   {
      memset( state, 0, 32 );
      return;
   }
   rinhash_hash80( rin_ctx, input, state );
}

int scanhash_rinhash_fast( struct work *work, uint32_t max_nonce,
                           uint64_t *hashes_done, struct thr_info *mythr )
{
   uint32_t hash[ 8 * RINHASH_BATCH ] __attribute__ ((aligned (64)));
   uint32_t *pdata = work->data;
   const uint32_t *ptarget = work->target;
   const uint32_t first_nonce = pdata[19];
   const uint32_t last_nonce = max_nonce - RINHASH_BATCH;
   const uint32_t targ_hi = ptarget[7];
   const int thr_id = mythr->id;
   const bool bench = opt_benchmark;
   uint32_t n = first_nonce;

   if ( unlikely( !rin_ctx_ready() ) )
   {
      *hashes_done = 0;
      return 0;
   }

   // Header bytes 0..75 are fixed for this call, only the nonce changes.
   rinhash_set_header( rin_ctx, pdata );

   do
   {
      rinhash_hash_batch( rin_ctx, n, hash );

      for ( int i = 0; i < RINHASH_BATCH; i++ )
      if ( unlikely( hash[ 8*i + 7 ] <= targ_hi ) )
      if ( fulltest( hash + 8*i, ptarget ) && !bench )
      {
         pdata[19] = n + i;
         submit_solution( work, hash + 8*i, mythr );
      }
      n += RINHASH_BATCH;
   } while ( likely( ( n < last_nonce ) && !work_restart[ thr_id ].restart ) );

   // The miner thread increments the nonce before the next scan.
   pdata[19] = n - 1;
   *hashes_done = n - first_nonce;
   return 0;
}

static bool rinhash_fast_thread_init( int thr_id )
{
   return rin_ctx_ready();
}

bool register_rin_algo_fast( algo_gate_t *gate )
{
   rinhash_global_init();
   if ( rinhash_selftest() < 0 )
   {
      applog( LOG_ERR, "RinHash self test failed" );
      return false;
   }
   if ( !opt_quiet )
      applog( LOG_INFO, "RinHash: %s", rinhash_impl_name() );

   gate->miner_thread_init = (void*)&rinhash_fast_thread_init;
   gate->scanhash = (void*)&scanhash_rinhash_fast;
   gate->hash = (void*)&rinhash_fast;
   gate->optimizations = SSE2_OPT | AVX2_OPT | AVX512_OPT;
   gate->build_stratum_request = (void*)&std_be_build_stratum_request;
   return true;
}
