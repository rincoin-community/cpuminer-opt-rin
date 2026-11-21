#ifndef RINHASH_GATE_H__
#define RINHASH_GATE_H__ 1

#include "algo-gate-api.h"

bool register_rin_algo_fast( algo_gate_t* gate );
int scanhash_rinhash_fast( struct work *work, uint32_t max_nonce,
                           uint64_t *hashes_done, struct thr_info *mythr );
void rinhash_fast( void *state, const void *input );

#endif
