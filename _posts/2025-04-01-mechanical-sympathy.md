---
title: "Mechanical Sympathy"
---

# Mechanical Sympathy

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

```mermaid
CPU Core (Registers: 0-1 cycles) --> L1 Cache (~3-5 cycles) --> L2 Cache (~12-20 cycles) --> L3 Cache (~30-70 cycles) --> Main Memory (~100-500 cycles)
```

## Reference

1. [Hardware-Aware Coding: CPU Architecture Concepts Every Developer Should Know](https://blog.codingconfessions.com/p/hardware-aware-coding?r=ruo3a&utm_campaign=post&utm_medium=web)
1. [Mechanical Sympathy: Understanding the Hardware Makes You a Better Developer](https://dzone.com/articles/mechanical-sympathy)

## Next Read

1. [The power of Mechanical Sympathy in Software Engineering](https://venkat.eu/the-power-of-mechanical-sympathy-in-software-engineering)
