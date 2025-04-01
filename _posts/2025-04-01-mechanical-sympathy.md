---
title: "Mechanical Sympathy"
---

# Mechanical Sympathy

**It is not a blog, more like notes. A lot of the content is from the reference, but the code and outputs are my own.**

## Introduction

The term "Mechanical Sympathy" was coined by legendary racing driver Jackie Stewart. As a driver, Stewart emphasized that while you don't need to be an engineer to excel in racing, having an understanding of how a car works is crucial to becoming a better driver. His famous quote, "You don’t have to be an engineer to be a racing driver, but you do have to have Mechanical Sympathy," reflects his belief that understanding the mechanics of a car allows a driver to better handle its performance, anticipate issues, and ultimately improve their racing skills.

Martin Thompson applied the concept of "Mechanical Sympathy" to the world of software development. Just like in racing, where understanding how a car works improves a driver’s performance, Thompson believes that in software development, you don’t need to be a hardware engineer, but you do need to understand how the hardware functions. This understanding is essential when designing software, as it allows developers to write more efficient, optimized code that works in harmony with the underlying hardware, ultimately improving performance and reducing potential issues.

Modern processors employ sophisticated mechanisms for executing instructions efficiently. Writing code that takes advantage of these mechanisms for delivering maximum performance is what I call "hardware-aware coding".

## Three CPU Concepts Every Developer Should Know

1. Instruction Pipelining
1. Memory Caching
1. Speculative Execution

### Instruction Pipelining

Instruction processing is pipelined in the processors to enable them to process multiple instructions simultaneously. The pipeline consists of several stages and as one instruction moves from the first stage to the second, it makes space for a new instruction to be introduced into the pipeline.

The number of stages in a CPU pipeline varies. Some simpler architectures consist of a five stage pipeline while there are also more complex pipelines in high performance architectures (such as x64) that have very deep pipelines consisting of 15-20 stages. Let’s use a simple five stage pipeline as an example to understand instruction pipelining better. These five stages are:

- Fetch: An instruction is fetched from memory. This is similar to taking the next order
- Decode: The instruction is decoded to identify the operation (e.g. add, multiply, divide) and its operands. In CISC style processors (e.g. X86), the instruction is also broken down into one or more simpler instructions, called microinstructions (μops). This is similar to how an order is broken down into its individual items and the required ingredients
- Execute: Decoded μops are picked up and executed (assuming all the required operands are available). This is similar to how an individual order item is picked up by a cook for preparation.
- Memory access: Any memory reads or writes required by the instruction are done
- Write back: The final instruction result is written back to the destination

_Parallel Order Processing_ : Modern CPU's have multiple ALUs for arithmetic computations, and multiple load/store units for memory operations. But, utilizing these requires that the CPU can issue multiple instructions in parallel, for instance, if there are two add instructions in the program, the CPU can process them in parallel instead of serial order. Modern X64 processors can issue up to 4 new instructions each cycle and with their deep pipelines, they can have hundreds of instructions in different stages of completion. Under a steady state, such a processor can deliver a throughput of up to 4 instructions/cycle (peak instruction throughput).

_Out-of-Order Execution_: Taking advantage of this instruction level parallelism (ILP) requires that the processor is able to find enough independent units of work which can be executed in parallel in the same cycle.

**Applying Mechanical Sympathy: Loop Unrolling**
There are situations where the code is written such that we have a sequence of instructions where the next instruction depends on the previous one’s result. For example:

```python
def loop(arr):

    sum = 0
    for i in range(len(arr)):
        # Each addition depends on previous result
        sum += arr[i]
```

The optimization to fix such a bottleneck is called “loop unrolling”. It basically means executing multiple steps of the loop in a single iteration. For example, we can compute four parallel sums in each iteration. Each sum value can be updated independently of the other one, the CPU will notice that and execute those four add operations in parallel.

```python
# Better superscalar utilization - independent operations
def loop_unrolling(arr):
    sum1, sum2, sum3, sum4 = 0, 0, 0, 0
    for i in range(len(arr), 2):
        sum1 += arr[i]
        sum2 += arr[i + 1]
        sum3 += arr[i + 2]
        sum4 += arr[i + 3]

    sum = sum1 + sum2 + sum3 + sum4
```

Usually, we don’t have to do such kind of optimization ourselves, the compilers these days can do this for us.

Full implementation:

```python
import timeit
import random

arr = [random.randint(0, 1000) for _ in range(random.randint(10, 10**6))]


def loop(arr):

    sum = 0
    for i in range(len(arr)):
        # Each addition depends on previous result
        sum += arr[i]


# Better superscalar utilization - independent operations
def loop_unrolling(arr):
    sum1, sum2, sum3, sum4 = 0, 0, 0, 0
    for i in range(len(arr), 2):
        sum1 += arr[i]
        sum2 += arr[i + 1]
        sum3 += arr[i + 2]
        sum4 += arr[i + 3]

    sum = sum1 + sum2 + sum3 + sum4


if __name__ == "__main__":
    # Run the function multiple times and measure the average time
    execution_time = timeit.timeit("loop(arr)", globals=globals(), number=100)
    ms_execution_time = timeit.timeit(
        "loop_unrolling(arr)", globals=globals(), number=100
    )

    print(f"Arr Length: {len(arr)}")
    print(f"Average execution time over 100 runs: {execution_time:.10f} seconds")
    print(f"Average execution time over 100 runs: {ms_execution_time:.10f} seconds")
    print(
        f"Differnce between execution time: {execution_time - ms_execution_time:.10f} seconds"
    )

```

Example Output:

```
Arr Length: 412489
Average execution time over 100 runs: 6.7541765530 seconds
Average execution time over 100 runs: 0.0000588250 seconds
Differnce between execution time: 6.7541177280 seconds

Arr Length: 367481
Average execution time over 100 runs: 5.9285027110 seconds
Average execution time over 100 runs: 0.0000557910 seconds
Differnce between execution time: 5.9284469200 seconds

Arr Length: 908559
Average execution time over 100 runs: 14.3761746600 seconds
Average execution time over 100 runs: 0.0000551420 seconds
Differnce between execution time: 14.3761195180 seconds

Arr Length: 948938
Average execution time over 100 runs: 15.1395270200 seconds
Average execution time over 100 runs: 0.0000515000 seconds
Differnce between execution time: 15.1394755200 seconds
```

### Memory Caching

CPU needs the operand values for an instruction to be available in the CPU to execute them. For example, consider a line of code a = b + c, where a, b, c are integers. To execute this code, the values of the variables b and c need to be brought into the CPU registers from memory.

All the program data resides in the main memory, and it takes 100-200 cycles to read any data from there. In the above example, if both the values need to be fetched from main memory, the simple add operation will take 200-400 cycles which is extremely slow. For comparison, the CPU can do the addition within a single cycle if the operands are in its registers, so this is 200x slower at the very least.

We usually have three levels of caches in the processors, called L1, L2 and L3 caches–each larger and slower and farther from the CPU than the previous level. The access latency of the L1 cache is usually 3-4 cycles which is significantly faster than main memory but it is of very small capacity, 32-64 kB on modern X64 hardware.

```
CPU Core (Registers: 0-1 cycles) > L1 Cache (~3-5 cycles) > L2 Cache (~12-20 cycles) > L3 Cache (~30-70 cycles) > Main Memory (~100-500 cycles)
```

**Temporal Locality**

Processors keeps the most recently used data in the L1 cache, anticipating temporal locality. As these caches are small, they also employ an eviction policy to make space for new data. Most of the processors implement some approximate variant of the least recently used eviction (LRU) policy, which means evict cache lines which have not been accessed for a while.

**Spacial Locality**
Processors stored related data contiguously or nearby in memory and instead of bringing just the value at the requested address into the cache, the cache brings an entire block of data. The block size depends on the memory bus capacity, which is usually 64 bytes on present day hardware.

The CPU caches are also organized to cache these entire blocks rather than caching the exact value that the application requested. Within the cache, these blocks are called cache lines and these are also 64 bytes in size (on X64 hardware at least).

Consider this example: When an application requests an 8-byte long value stored at memory address 130, the hardware doesn't retrieve only those specific bytes. Instead, it fetches the entire 64-byte block starting at address 128 (the nearest 64-byte aligned address) and loads this block into one of the available cache lines. Subsequently, if the program attempts to access any value within the address range of 128-191, that data will already reside in the cache, eliminating the need for additional time-consuming trips to main memory.

Taking advantage of spatial and temporal locality results in improved system performance and better utilization of the memory bandwidth. But doing this requires writing code with mechanical sympathy. Let’s understand with some examples.

Full implementation: Attributes Grouping

```python
import random
import timeit


class GameEntityScattered:
    def __init__(self):
        # Hot path data - scattered
        self.position_x = 10  # Frequently accessed
        self.name = "Entity " + str(
            random.randint(1, 100)
        )  # Cold data - rarely accessed
        self.position_y = 10  # Frequently accessed
        self.description = "A game entity."  # Cold data - rarely accessed
        self.position_z = 10  # Frequently accessed
        self.model_path = "/path/to/model.obj"  # Cold data - rarely accessed
        self.rotation = 10  # Frequently accessed
        self.id = random.randint(1, 1000)  # Frequently accessed


# Benchmark Scattered layout
def benchmark_scattered(num_entities):
    entities = [GameEntityScattered() for _ in range(num_entities)]

    for entity in entities:
        # Access hot path data
        _ = (
            entity.position_x
            + entity.position_y
            + entity.position_z
            + entity.rotation
            + entity.id
        )

class GameEntityGrouped:
    def __init__(self):
        # Hot path data - scattered
        self.position_x = 10  # Frequently accessed
        self.position_y = 10  # Frequently accessed
        self.position_z = 10  # Frequently accessed
        self.rotation = 10  # Frequently accessed
        self.id = random.randint(1, 1000)  # Frequently accessed

        # Cold data - rarely accessed
        self.name = "Entity " + str(random.randint(1, 100))
        self.description = "A game entity."
        self.model_path = "/path/to/model.obj"


# Benchmark Grouped layout
def benchmark_grouped(num_entities):
    entities = [GameEntityGrouped() for _ in range(num_entities)]

    for entity in entities:
        # Access hot path data
        _ = (
            entity.position_x
            + entity.position_y
            + entity.position_z
            + entity.rotation
            + entity.id
        )

if __name__ == "__main__":
    for i in range(4):
        num_entities = 10**i
        print(f"Num of entities: {num_entities}")

        execution_time = timeit.timeit(
            "benchmark_scattered(num_entities)", globals=globals(), number=100
        )
        print(f"Average execution time (Scattered) over 100 runs: {execution_time:5f} seconds")

        gr_execution_time = timeit.timeit(
            "benchmark_grouped(num_entities)", globals=globals(), number=100
        )
        print(f"Average execution time (Grouped) over 100 runs: {gr_execution_time:.5f} seconds")
```

Example Output:

```
Num of entities: 1
Average execution time (Scattered) over 100 runs: 0.001057 seconds
Average execution time (Grouped) over 100 runs: 0.00071 seconds

Num of entities: 10
Average execution time (Scattered) over 100 runs: 0.012468 seconds
Average execution time (Grouped) over 100 runs: 0.00751 seconds

Num of entities: 100
Average execution time (Scattered) over 100 runs: 0.049320 seconds
Average execution time (Grouped) over 100 runs: 0.05049 seconds

Num of entities: 1000
Average execution time (Scattered) over 100 runs: 0.452807 seconds
Average execution time (Grouped) over 100 runs: 0.44467 seconds
```

Full implementation: Matrix Traversal Sum

```python
import timeit

a = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
]


def matrix_multiply_row_wise():
    # Variable to store the sum
    sum = 0

    # Loop through the matrix
    for i in range(6):  # Outer loop for 6 rows
        for j in range(4):  # Inner loop for 4 columns
            sum += a[i][j]  # Add the element to sum


def matrix_multiply_col_wise():
    # Variable to store the sum
    sum = 0

    # Loop through the matrix column-wise
    for j in range(4):  # Outer loop for 4 columns
        for i in range(6):  # Inner loop for 6 rows
            sum += a[i][j]  # Add the element to sum


if __name__ == "__main__":

    for _ in range(8):

        execution_time = timeit.timeit(
            "matrix_multiply_row_wise()", globals=globals(), number=100
        )

        print(f"Average execution time (Row) over 100 runs: {execution_time:.10f} seconds")

        col_execution_time = timeit.timeit(
            "matrix_multiply_col_wise()", globals=globals(), number=100
        )

        print(
            f"Average execution time (Column) over 100 runs: {col_execution_time:.10f} seconds"
        )

```

Example Output:

```
Average execution time (Row) over 100 runs: 0.0007095990 seconds
Average execution time (Column) over 100 runs: 0.0004877360 seconds

Average execution time (Row) over 100 runs: 0.0009539990 seconds
Average execution time (Column) over 100 runs: 0.0011540680 seconds

Average execution time (Row) over 100 runs: 0.0007116180 seconds
Average execution time (Column) over 100 runs: 0.0006330580 seconds

Average execution time (Row) over 100 runs: 0.0003995010 seconds
Average execution time (Column) over 100 runs: 0.0004394730 second
```

As we can see in both examples, the output is not consistent with locality rules, which suggest that we need to investigate Python's memory management further to understand why this is the case.

### Speculative Execution

While the hardware can execute instructions very fast, fetching new instructions from memory takes time. To keep the execution resources in the processor busy, the instructions must be supplied at a fast rate. If programs had a linear structure where one instruction followed another, this wouldn’t be a problem. The processor already prefetches sequential instructions and keeps them cached ahead of time to keep the pipeline fed.

However, program structure is not always linear. All non-trivial programs consist of branches in the form of if/else blocks, switch cases, loops, and function calls. For example, in the case of an if/else conditional block, the value of the condition decides which block of code needs to be executed next. The branch condition itself involves one or more instructions which need to be executed before the processor knows which instructions to fetch next. If the processor waits for that to occur, the pipeline will not be fetching and decoding any new instructions until that time, resulting in a significant drop in the instruction throughput. And the performance penalty doesn’t end there - after the branching direction is determined and the location of the next instruction is known, the instructions from that address need to be fetched and decoded, which is another level of delay (especially if those instructions are not in the cache).

To avoid this performance degradation, the CPU implements branch predictors to predict the target address of branches. But, whenever the branch predictor is wrong, there is a penalty on performance as well. When the processor finally evaluates the branch condition and finds that the branch predictor did a misprediction, it has to flush the pipeline because it was processing the wrong set of instructions. Once the pipeline is flushed, the processor starts to execute the right set of instructions. On modern X64 processors this misprediction can cause a performance penalty of ~20-30 cycles. Modern processors have sophisticated branch predictors which can learn very complex branching patterns and offer accuracy of up to ~95%. But they still depend on predictable branching patterns in code.

Full Implementation:

```python
import timeit


def abs(x):
    if x >= 0:
        return x

    return -x


def abs_bitwise(x):
    y = x >> 31
    return (x**y) - y


if __name__ == "__main__":
    for i in range(-2, 2):
        print(f"X: {i}")

        execution_time = timeit.timeit("abs(i)", globals=globals(), number=100)
        print(
            f"Average execution time (Absolute) over 100 runs: {execution_time:5f} seconds"
        )

        bt_execution_time = timeit.timeit(
            "abs_bitwise(i)", globals=globals(), number=100
        )
        print(
            f"Average execution time (Absolute Bitwise) over 100 runs: {bt_execution_time:.5f} seconds"
        )

        print("")

```

Example Output:

```
X: -2
Average execution time (Absolute) over 100 runs: 0.000029 seconds
Average execution time (Absolute Bitwise) over 100 runs: 0.00012 seconds

X: -1
Average execution time (Absolute) over 100 runs: 0.000035 seconds
Average execution time (Absolute Bitwise) over 100 runs: 0.00011 seconds

X: 0
Average execution time (Absolute) over 100 runs: 0.000014 seconds
Average execution time (Absolute Bitwise) over 100 runs: 0.00002 seconds

X: 1
Average execution time (Absolute) over 100 runs: 0.000013 seconds
Average execution time (Absolute Bitwise) over 100 runs: 0.00006 seconds
```

## Conclusion

Understanding how processors work "under the hood" is essential for writing truly performant code. Modern CPUs are complex systems with many optimizations that work best when code aligns with their expectations:

- **Instruction pipelining and superscalar execution** enable multiple instructions to be processed simultaneously, but only when we provide independent operations
- **Memory hierarchy and caching** dramatically reduce latency, but only when our data structures and access patterns exhibit good locality
- **Speculative execution** keeps the pipeline full through branches, but becomes a performance liability with unpredictable branching patterns

The most effective optimization approach is to:

1. Write clean, maintainable code first
1. Profile to identify performance bottlenecks
1. Apply mechanical sympathy principles specifically to those hot spots

As software engineers, we don't need to fight the hardware - we can work with it. When we align our algorithms and data structures with the grain of the processor rather than against it, we achieve that satisfying moment when a program that once took seconds now completes in milliseconds.

## Reference

1. [Hardware-Aware Coding: CPU Architecture Concepts Every Developer Should Know](https://blog.codingconfessions.com/p/hardware-aware-coding?r=ruo3a&utm_campaign=post&utm_medium=web)
1. [Mechanical Sympathy: Understanding the Hardware Makes You a Better Developer](https://dzone.com/articles/mechanical-sympathy)

## Next Read

1. [The power of Mechanical Sympathy in Software Engineering](https://venkat.eu/the-power-of-mechanical-sympathy-in-software-engineering)
