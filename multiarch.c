/*
 * Dispatcher of the multi-level x86-64 build made by build-multiarch.sh.
 *
 * That build contains the algorithm code of the miner four times, compiled
 * for four CPU levels, each copy with its symbols prefixed ml1_ .. ml4_. The
 * miner framework (cpu-miner.c, util.c, api.c, ...) exists once, compiled for
 * the baseline CPU, and reaches the algorithm code through the six functions
 * below. They forward to the copy that fits the CPU the program runs on.
 *
 *   level 1  sse2     x86-64
 *   level 2  sse4.2   x86-64-v2
 *   level 3  avx2     x86-64-v3 with AES
 *   level 4  avx512   x86-64-v4 with AES
 *
 * The environment variable CPUMINER_LEVEL=1..4 selects a lower level than
 * the CPU supports, for testing.
 *
 * This file is not part of a normal build.
 */

#define _GNU_SOURCE
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__GLIBC__)
#include <pthread.h>
#endif

#define LEVEL_FUNCTIONS( n ) \
   bool ml##n##_register_algo_gate( int algo, void *gate ); \
   void ml##n##_exec_hash_function( int algo, void *output, const void *pdata ); \
   void ml##n##_get_algo_alias( char **algo_or_alias ); \
   void ml##n##_sha256d( void *hash, const void *data, int len ); \
   void ml##n##_sph_sha256_full( void *hash, const void *data, size_t len ); \
   void ml##n##_sph_sha1_full( void *hash, const void *msg, size_t len );

LEVEL_FUNCTIONS( 1 )
LEVEL_FUNCTIONS( 2 )
LEVEL_FUNCTIONS( 3 )
LEVEL_FUNCTIONS( 4 )

typedef struct
{
   const char *name;
   bool (*register_algo_gate)( int, void* );
   void (*exec_hash_function)( int, void*, const void* );
   void (*get_algo_alias)( char** );
   void (*sha256d)( void*, const void*, int );
   void (*sph_sha256_full)( void*, const void*, size_t );
   void (*sph_sha1_full)( void*, const void*, size_t );
} level_t;

#define LEVEL( n, name ) \
   { name, ml##n##_register_algo_gate, ml##n##_exec_hash_function, \
     ml##n##_get_algo_alias, ml##n##_sha256d, ml##n##_sph_sha256_full, \
     ml##n##_sph_sha1_full }

static const level_t levels[4] =
{
   LEVEL( 1, "SSE2 (x86-64)" ),
   LEVEL( 2, "SSE4.2 (x86-64-v2)" ),
   LEVEL( 3, "AVX2 (x86-64-v3, AES)" ),
   LEVEL( 4, "AVX-512 (x86-64-v4, AES)" )
};

static const level_t *level = NULL;
static int cpu_max = 0, in_use = 0;

static int cpu_level( void )
{
   __builtin_cpu_init();
   const bool aes = __builtin_cpu_supports( "aes" );
   if ( __builtin_cpu_supports( "x86-64-v4" ) && aes ) return 4;
   if ( __builtin_cpu_supports( "x86-64-v3" ) && aes ) return 3;
   if ( __builtin_cpu_supports( "x86-64-v2" ) )        return 2;
   return 1;
}

static inline const level_t *selected( void )
{
   if ( __builtin_expect( !level, 0 ) )
   {
      const char *env = getenv( "CPUMINER_LEVEL" );
      cpu_max = in_use = cpu_level();
      if ( env )
      {
         const int want = atoi( env );
         if ( want >= 1 && want <= cpu_max )
            in_use = want;
         else
            fprintf( stderr, "CPUMINER_LEVEL=%s ignored, this CPU supports "
                             "levels 1 to %d\n", env, cpu_max );
      }
      level = &levels[ in_use - 1 ];
   }
   return level;
}

/*
 * glibc places the thread-local variables of the program at the top of every
 * thread's stack. Many algorithms keep large hashing contexts thread-local,
 * and with four copies of them these variables are larger than the default
 * 8 MiB stack. Give new threads a stack that has room for both; the memory
 * is only reserved, pages are used when touched.
 */
static void enlarge_thread_stacks( void )
{
#if defined(__GLIBC__)
   pthread_attr_t attr;
   if ( pthread_attr_init( &attr ) ) return;
   if ( !pthread_attr_setstacksize( &attr, (size_t)48 << 20 ) )
      pthread_setattr_default_np( &attr );
   pthread_attr_destroy( &attr );
#endif
}

// Called once from main() before any thread is started.
bool register_algo_gate( int algo, void *gate )
{
   const level_t *l = selected();
   enlarge_thread_stacks();
   printf( "Multi-level build: using the %s code, level %d of the %d this "
           "CPU supports\n", l->name, in_use, cpu_max );
   return l->register_algo_gate( algo, gate );
}

void exec_hash_function( int algo, void *output, const void *pdata )
{
   selected()->exec_hash_function( algo, output, pdata );
}

void get_algo_alias( char **algo_or_alias )
{
   selected()->get_algo_alias( algo_or_alias );
}

void sha256d( void *hash, const void *data, int len )
{
   selected()->sha256d( hash, data, len );
}

void sph_sha256_full( void *hash, const void *data, size_t len )
{
   selected()->sph_sha256_full( hash, data, len );
}

void sph_sha1_full( void *hash, const void *msg, size_t len )
{
   selected()->sph_sha1_full( hash, msg, len );
}
