#include <algorithm>
#include <cstdio>
#include <execution>
#include <vector>

int main() {
  unsigned n = 5;

  // Allocate arrays
  std::vector<int> a(n), b(n), c(n);

  // Initialize values
  for (unsigned i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 1;
  }

  // Run element-wise multiplication on device
  std::transform(std::execution::par_unseq, a.begin(), a.end(), b.begin(),
                 c.begin(), [](int i, int j) { return i * j; });

  for (unsigned i = 0; i < n; i++) {
    printf("c[%d] = %d\n", i, c[i]);
  }

  return 0;
}
