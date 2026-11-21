/*
 * Argon2 reference source code package - reference C implementations
 *
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
 *
 * You may use this work under the terms of a Creative Commons CC0 1.0
 * License/Waiver or the Apache Public License 2.0, at your option. The terms of
 * these licenses can be found at:
 *
 * - CC0 1.0 Universal : https://creativecommons.org/publicdomain/zero/1.0
 * - Apache 2.0        : https://www.apache.org/licenses/LICENSE-2.0
 *
 * You should have received a copy of both of these licenses along with this
 * software. If not, they may be obtained at the above URLs.
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "blake2.h"
#include "blake2-impl.h"

static const uint64_t blake2b_IV[8] = {
    UINT64_C(0x6a09e667f3bcc908), UINT64_C(0xbb67ae8584caa73b),
    UINT64_C(0x3c6ef372fe94f82b), UINT64_C(0xa54ff53a5f1d36f1),
    UINT64_C(0x510e527fade682d1), UINT64_C(0x9b05688c2b3e6c1f),
    UINT64_C(0x1f83d9abfb41bd6b), UINT64_C(0x5be0cd19137e2179)};

static const unsigned int blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

static BLAKE2_INLINE void blake2b_set_lastnode(blake2b_state *S) {
    S->f[1] = (uint64_t)-1;
}

static BLAKE2_INLINE void blake2b_set_lastblock(blake2b_state *S) {
    if (S->last_node) {
        blake2b_set_lastnode(S);
    }
    S->f[0] = (uint64_t)-1;
}

static BLAKE2_INLINE void blake2b_increment_counter(blake2b_state *S,
                                                    uint64_t inc) {
    S->t[0] += inc;
    S->t[1] += (S->t[0] < inc);
}

static BLAKE2_INLINE void blake2b_invalidate_state(blake2b_state *S) {
    clear_internal_memory(S, sizeof(*S));      /* wipe */
    blake2b_set_lastblock(S); /* invalidate for further use */
}

static BLAKE2_INLINE void blake2b_init0(blake2b_state *S) {
    memset(S, 0, sizeof(*S));
    memcpy(S->h, blake2b_IV, sizeof(S->h));
}

int blake2b_init_param(blake2b_state *S, const blake2b_param *P) {
    const unsigned char *p = (const unsigned char *)P;
    unsigned int i;

    if (NULL == P || NULL == S) {
        return -1;
    }

    blake2b_init0(S);
    /* IV XOR Parameter Block */
    for (i = 0; i < 8; ++i) {
        S->h[i] ^= load64(&p[i * sizeof(S->h[i])]);
    }
    S->outlen = P->digest_length;
    return 0;
}

/* Sequential blake2b initialization */
int blake2b_init(blake2b_state *S, size_t outlen) {
    blake2b_param P;

    if (S == NULL) {
        return -1;
    }

    if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
        blake2b_invalidate_state(S);
        return -1;
    }

    /* Setup Parameter Block for unkeyed BLAKE2 */
    P.digest_length = (uint8_t)outlen;
    P.key_length = 0;
    P.fanout = 1;
    P.depth = 1;
    P.leaf_length = 0;
    P.node_offset = 0;
    P.node_depth = 0;
    P.inner_length = 0;
    memset(P.reserved, 0, sizeof(P.reserved));
    memset(P.salt, 0, sizeof(P.salt));
    memset(P.personal, 0, sizeof(P.personal));

    return blake2b_init_param(S, &P);
}

int blake2b_init_key(blake2b_state *S, size_t outlen, const void *key,
                     size_t keylen) {
    blake2b_param P;

    if (S == NULL) {
        return -1;
    }

    if ((outlen == 0) || (outlen > BLAKE2B_OUTBYTES)) {
        blake2b_invalidate_state(S);
        return -1;
    }

    if ((key == 0) || (keylen == 0) || (keylen > BLAKE2B_KEYBYTES)) {
        blake2b_invalidate_state(S);
        return -1;
    }

    /* Setup Parameter Block for keyed BLAKE2 */
    P.digest_length = (uint8_t)outlen;
    P.key_length = (uint8_t)keylen;
    P.fanout = 1;
    P.depth = 1;
    P.leaf_length = 0;
    P.node_offset = 0;
    P.node_depth = 0;
    P.inner_length = 0;
    memset(P.reserved, 0, sizeof(P.reserved));
    memset(P.salt, 0, sizeof(P.salt));
    memset(P.personal, 0, sizeof(P.personal));

    if (blake2b_init_param(S, &P) < 0) {
        blake2b_invalidate_state(S);
        return -1;
    }

    {
        uint8_t block[BLAKE2B_BLOCKBYTES];
        memset(block, 0, BLAKE2B_BLOCKBYTES);
        memcpy(block, key, keylen);
        blake2b_update(S, block, BLAKE2B_BLOCKBYTES);
        /* Burn the key from stack */
        clear_internal_memory(block, BLAKE2B_BLOCKBYTES);
    }
    return 0;
}

#if defined(__AVX2__)

#include <immintrin.h>

/*
 * AVX2 compression function: each of the four rows of the BLAKE2b state lives
 * in one 256-bit register, so a column step or a diagonal step is one
 * vectorized G.
 */
static void blake2b_compress(blake2b_state *S, const uint8_t *block) {
    const __m256i r24 = _mm256_setr_epi8(
        3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10,
        3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10);
    const __m256i r16 = _mm256_setr_epi8(
        2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9,
        2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9);
    uint64_t m[16];
    __m256i a, b, c, d;
    unsigned int r;

    memcpy(m, block, sizeof(m));

    a = _mm256_loadu_si256((const __m256i *)&S->h[0]);
    b = _mm256_loadu_si256((const __m256i *)&S->h[4]);
    c = _mm256_loadu_si256((const __m256i *)&blake2b_IV[0]);
    d = _mm256_xor_si256(
        _mm256_loadu_si256((const __m256i *)&blake2b_IV[4]),
        _mm256_set_epi64x((long long)S->f[1], (long long)S->f[0],
                          (long long)S->t[1], (long long)S->t[0]));

#define MSG(s, i0, i1, i2, i3)                                                 \
    _mm256_set_epi64x((long long)m[s[i3]], (long long)m[s[i2]],                \
                      (long long)m[s[i1]], (long long)m[s[i0]])

#define G(x, y)                                                                \
    do {                                                                       \
        a = _mm256_add_epi64(_mm256_add_epi64(a, b), x);                       \
        d = _mm256_shuffle_epi32(_mm256_xor_si256(d, a),                       \
                                 _MM_SHUFFLE(2, 3, 0, 1));                     \
        c = _mm256_add_epi64(c, d);                                            \
        b = _mm256_shuffle_epi8(_mm256_xor_si256(b, c), r24);                  \
        a = _mm256_add_epi64(_mm256_add_epi64(a, b), y);                       \
        d = _mm256_shuffle_epi8(_mm256_xor_si256(d, a), r16);                  \
        c = _mm256_add_epi64(c, d);                                            \
        b = _mm256_xor_si256(b, c);                                            \
        b = _mm256_or_si256(_mm256_srli_epi64(b, 63),                          \
                            _mm256_add_epi64(b, b));                           \
    } while ((void)0, 0)

    for (r = 0; r < 12; ++r) {
        const unsigned int *s = blake2b_sigma[r];

        /* columns */
        G(MSG(s, 0, 2, 4, 6), MSG(s, 1, 3, 5, 7));
        /* diagonals */
        b = _mm256_permute4x64_epi64(b, _MM_SHUFFLE(0, 3, 2, 1));
        c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(1, 0, 3, 2));
        d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(2, 1, 0, 3));
        G(MSG(s, 8, 10, 12, 14), MSG(s, 9, 11, 13, 15));
        b = _mm256_permute4x64_epi64(b, _MM_SHUFFLE(2, 1, 0, 3));
        c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(1, 0, 3, 2));
        d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(0, 3, 2, 1));
    }

#undef G
#undef MSG

    a = _mm256_xor_si256(a, c);
    b = _mm256_xor_si256(b, d);
    _mm256_storeu_si256(
        (__m256i *)&S->h[0],
        _mm256_xor_si256(a, _mm256_loadu_si256((const __m256i *)&S->h[0])));
    _mm256_storeu_si256(
        (__m256i *)&S->h[4],
        _mm256_xor_si256(b, _mm256_loadu_si256((const __m256i *)&S->h[4])));
}

#else /* portable */

static void blake2b_compress(blake2b_state *S, const uint8_t *block) {
    uint64_t m[16];
    uint64_t v[16];
    unsigned int i, r;

    for (i = 0; i < 16; ++i) {
        m[i] = load64(block + i * sizeof(m[i]));
    }

    for (i = 0; i < 8; ++i) {
        v[i] = S->h[i];
    }

    v[8] = blake2b_IV[0];
    v[9] = blake2b_IV[1];
    v[10] = blake2b_IV[2];
    v[11] = blake2b_IV[3];
    v[12] = blake2b_IV[4] ^ S->t[0];
    v[13] = blake2b_IV[5] ^ S->t[1];
    v[14] = blake2b_IV[6] ^ S->f[0];
    v[15] = blake2b_IV[7] ^ S->f[1];

#define G(r, i, a, b, c, d)                                                    \
    do {                                                                       \
        a = a + b + m[blake2b_sigma[r][2 * i + 0]];                            \
        d = rotr64(d ^ a, 32);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 24);                                                 \
        a = a + b + m[blake2b_sigma[r][2 * i + 1]];                            \
        d = rotr64(d ^ a, 16);                                                 \
        c = c + d;                                                             \
        b = rotr64(b ^ c, 63);                                                 \
    } while ((void)0, 0)

#define ROUND(r)                                                               \
    do {                                                                       \
        G(r, 0, v[0], v[4], v[8], v[12]);                                      \
        G(r, 1, v[1], v[5], v[9], v[13]);                                      \
        G(r, 2, v[2], v[6], v[10], v[14]);                                     \
        G(r, 3, v[3], v[7], v[11], v[15]);                                     \
        G(r, 4, v[0], v[5], v[10], v[15]);                                     \
        G(r, 5, v[1], v[6], v[11], v[12]);                                     \
        G(r, 6, v[2], v[7], v[8], v[13]);                                      \
        G(r, 7, v[3], v[4], v[9], v[14]);                                      \
    } while ((void)0, 0)

    for (r = 0; r < 12; ++r) {
        ROUND(r);
    }

    for (i = 0; i < 8; ++i) {
        S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
    }

#undef G
#undef ROUND
}

#endif /* __AVX2__ */

int blake2b_update(blake2b_state *S, const void *in, size_t inlen) {
    const uint8_t *pin = (const uint8_t *)in;

    if (inlen == 0) {
        return 0;
    }

    /* Sanity check */
    if (S == NULL || in == NULL) {
        return -1;
    }

    /* Is this a reused state? */
    if (S->f[0] != 0) {
        return -1;
    }

    if (S->buflen + inlen > BLAKE2B_BLOCKBYTES) {
        /* Complete current block */
        size_t left = S->buflen;
        size_t fill = BLAKE2B_BLOCKBYTES - left;
        memcpy(&S->buf[left], pin, fill);
        blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
        blake2b_compress(S, S->buf);
        S->buflen = 0;
        inlen -= fill;
        pin += fill;
        /* Avoid buffer copies when possible */
        while (inlen > BLAKE2B_BLOCKBYTES) {
            blake2b_increment_counter(S, BLAKE2B_BLOCKBYTES);
            blake2b_compress(S, pin);
            inlen -= BLAKE2B_BLOCKBYTES;
            pin += BLAKE2B_BLOCKBYTES;
        }
    }
    memcpy(&S->buf[S->buflen], pin, inlen);
    S->buflen += (unsigned int)inlen;
    return 0;
}

int blake2b_final(blake2b_state *S, void *out, size_t outlen) {
    uint8_t buffer[BLAKE2B_OUTBYTES] = {0};
    unsigned int i;

    /* Sanity checks */
    if (S == NULL || out == NULL || outlen < S->outlen) {
        return -1;
    }

    /* Is this a reused state? */
    if (S->f[0] != 0) {
        return -1;
    }

    blake2b_increment_counter(S, S->buflen);
    blake2b_set_lastblock(S);
    memset(&S->buf[S->buflen], 0, BLAKE2B_BLOCKBYTES - S->buflen); /* Padding */
    blake2b_compress(S, S->buf);

    for (i = 0; i < 8; ++i) { /* Output full hash to temp buffer */
        store64(buffer + sizeof(S->h[i]) * i, S->h[i]);
    }

    memcpy(out, buffer, S->outlen);
    clear_internal_memory(buffer, sizeof(buffer));
    clear_internal_memory(S->buf, sizeof(S->buf));
    clear_internal_memory(S->h, sizeof(S->h));
    return 0;
}

int blake2b(void *out, size_t outlen, const void *in, size_t inlen,
            const void *key, size_t keylen) {
    blake2b_state S;
    int ret = -1;

    /* Verify parameters */
    if (NULL == in && inlen > 0) {
        goto fail;
    }

    if (NULL == out || outlen == 0 || outlen > BLAKE2B_OUTBYTES) {
        goto fail;
    }

    if ((NULL == key && keylen > 0) || keylen > BLAKE2B_KEYBYTES) {
        goto fail;
    }

    if (keylen > 0) {
        if (blake2b_init_key(&S, outlen, key, keylen) < 0) {
            goto fail;
        }
    } else {
        if (blake2b_init(&S, outlen) < 0) {
            goto fail;
        }
    }

    if (blake2b_update(&S, in, inlen) < 0) {
        goto fail;
    }
    ret = blake2b_final(&S, out, outlen);

fail:
    clear_internal_memory(&S, sizeof(S));
    return ret;
}

/* Argon2 Team - Begin Code */
int blake2b_long(void *pout, size_t outlen, const void *in, size_t inlen) {
    uint8_t *out = (uint8_t *)pout;
    blake2b_state blake_state;
    uint8_t outlen_bytes[sizeof(uint32_t)] = {0};
    int ret = -1;

    if (outlen > UINT32_MAX) {
        goto fail;
    }

    /* Ensure little-endian byte order! */
    store32(outlen_bytes, (uint32_t)outlen);

#define TRY(statement)                                                         \
    do {                                                                       \
        ret = statement;                                                       \
        if (ret < 0) {                                                         \
            goto fail;                                                         \
        }                                                                      \
    } while ((void)0, 0)

    if (outlen <= BLAKE2B_OUTBYTES) {
        TRY(blake2b_init(&blake_state, outlen));
        TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
        TRY(blake2b_update(&blake_state, in, inlen));
        TRY(blake2b_final(&blake_state, out, outlen));
    } else {
        uint32_t toproduce;
        uint8_t out_buffer[BLAKE2B_OUTBYTES];
        uint8_t in_buffer[BLAKE2B_OUTBYTES];
        TRY(blake2b_init(&blake_state, BLAKE2B_OUTBYTES));
        TRY(blake2b_update(&blake_state, outlen_bytes, sizeof(outlen_bytes)));
        TRY(blake2b_update(&blake_state, in, inlen));
        TRY(blake2b_final(&blake_state, out_buffer, BLAKE2B_OUTBYTES));
        memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
        out += BLAKE2B_OUTBYTES / 2;
        toproduce = (uint32_t)outlen - BLAKE2B_OUTBYTES / 2;

        while (toproduce > BLAKE2B_OUTBYTES) {
            memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
            TRY(blake2b(out_buffer, BLAKE2B_OUTBYTES, in_buffer,
                        BLAKE2B_OUTBYTES, NULL, 0));
            memcpy(out, out_buffer, BLAKE2B_OUTBYTES / 2);
            out += BLAKE2B_OUTBYTES / 2;
            toproduce -= BLAKE2B_OUTBYTES / 2;
        }

        memcpy(in_buffer, out_buffer, BLAKE2B_OUTBYTES);
        TRY(blake2b(out_buffer, toproduce, in_buffer, BLAKE2B_OUTBYTES, NULL,
                    0));
        memcpy(out, out_buffer, toproduce);
    }
fail:
    clear_internal_memory(&blake_state, sizeof(blake_state));
    return ret;
#undef TRY
}
/* Argon2 Team - End Code */
