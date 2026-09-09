/* Regenerate the table in wirl/ggml_types.py from llama.cpp itself.
 *
 *   gcc -fopenmp -I<llama.cpp>/ggml/include tools/dump_ggml_types.c \
 *       -o /tmp/dump_ggml_types <llama.cpp>/build/ggml/src/libggml-base.a \
 *       -lm -lstdc++ -lpthread
 *   /tmp/dump_ggml_types
 *
 * Prints: <id> <name> <block_size> <bytes_per_block>
 * Types reported with size 0 are removed/deprecated and must stay out of the
 * table -- a file using one will not load, so it should not be priced.
 */
#include "ggml.h"
#include <stdio.h>

int main(void) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        enum ggml_type ty = (enum ggml_type)t;
        const char *n = ggml_type_name(ty);
        if (!n) continue;
        printf("%d %s %lld %zu\n", t, n,
               (long long)ggml_blck_size(ty), ggml_type_size(ty));
    }
    return 0;
}
