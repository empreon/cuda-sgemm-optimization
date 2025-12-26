#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"

int main() {
    int N = 4096;

    run_naive(N);
    run_coalesced(N);
    run_tiling(N);
    run_vectorized(N);
}