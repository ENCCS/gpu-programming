**alpaka** - code examples
==========================

To guarantee a compact and consistent code style for the alpaka examples the code is formatted with clang-format 14.0.0.
To format the code examples, run the following command from the root of this repository:

```bash
clang-format -style="{BasedOnStyle: Google, IndentWidth: 2, ColumnLimit: 80, BreakBeforeBraces: Attach, BinPackParameters: false, AlignAfterOpenBracket: DontAlign}" -i content/examples/portable-kernel-models/alpaka*.cpp
```

To check if all examples compile and can be executed correctly, run the following commands:
  - You must have ROCm installed (because the default API in the examples is AMD HIP).
  - Update the path after `-I` to the alpaka include folder in the following command.
  - Run the command from the root of this repository.

```bash
# compile
for testCase in $(ls -w1 content/examples/portable-kernel-models/alpaka*) 
do 
  outName=$(basename "$testCase" .cpp)
  echo $outName
  hipcc -I ../alpaka3/include/ -std=c++20 -x hip $testCase -o "$outName"
done

# run
for testCase in $(ls -w1 content/examples/portable-kernel-models/alpaka*) 
do 
  outName=$(basename "$testCase" .cpp)
  echo "execute: $outName"
    ./"$outName"
done
```
