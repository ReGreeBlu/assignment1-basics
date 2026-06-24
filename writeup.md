### Problem (train_bpe_tinystories): BPE Training on TinyStories

(a) 

time: 25 minutes

memory: 10 GB

longest token: < accomplishment>

```bash
     1495.36 real      1487.05 user         6.77 sys
         10758242304  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
             1776898  page reclaims
                 963  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   1  messages sent
                   1  messages received
                   1  signals received
                 246  voluntary context switches
               37508  involuntary context switches
           181603710  instructions retired
            96825550  cycles elapsed
            12599656  peak memory footprint
```

(b) 

cProfile result on TinyStoriesV2-GPT4-valid.txt, vocabulary size 10000:

```bash
         150824515 function calls (150824406 primitive calls) in 19.819 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
9745/9744    8.533    0.001   15.379    0.002 {built-in method builtins.max}
148599155    6.846    0.000    6.846    0.000 train_bpe.py:24(<lambda>)
    27631    2.724    0.000    2.726    0.000 __init__.py:921(_keep_positive)
    27631    0.753    0.000    0.753    0.000 {method 'findall' of '_regex.Pattern' objects}
    27631    0.235    0.000    2.964    0.000 __init__.py:928(__iadd__)
     9743    0.200    0.000   15.642    0.002 train_bpe.py:10(token_merge)
    27631    0.171    0.000    0.171    0.000 {built-in method _collections._count_elements}
        1    0.074    0.074   19.816   19.816 train_bpe.py:66(train_bpe)
    27632    0.032    0.000    0.130    0.000 _main.py:459(_compile)
    55423    0.028    0.000    0.081    0.000 enum.py:1609(__and__)
```

Profiling shows the bottleneck shifted to the argmax over pair_counts.

Scanning all pairs on every merge step costs ~78% of total runtime.

A heap with lazy deletion would further optimize the algorithm.

