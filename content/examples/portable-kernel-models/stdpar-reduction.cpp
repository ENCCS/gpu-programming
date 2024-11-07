#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

int main() {
  unsigned n = 10;

  std::vector<int> a(n);

  std::iota(a.begin(), a.end(), 0); // Fill the array

  // Run reduction on the device
  int sum = std::reduce(std::execution::par_unseq, a.cbegin(), a.cend(), 0,
                        std::plus<int>{});

  // Print results
  printf("sum = %d\n", sum);

  return 0;
}
