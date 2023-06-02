Glossary
========

..
   how to refer to terms:
   :term:`thread`
   :term:`threads <thread>`  - different text
   :term:`thread`\ s  - different way to make plural

.. glossary::
   :sorted:

   thread
      Definition.  otherframework: :term:`workitem`

   workitem
      Definition.  otherframework: :term:`thread`


.. list-table::  
   :widths: 100 100
   :header-rows: 1

   * - CPU
     - GPU
   * - General purpose
     - Highly specialized for parallelism
   * - Good for serial processing
     - Good for parallel processing
   * - Great for task parallelism
     - Great for data parallelism
   * - Low latency per thread
     - High-throughput
   * - Large area dedicated cache and control
     - Hundreds of floating-point execution units


Abbreviations
^^^^^^^^^^^^^

+-------------+--------------------------------------------+
| abbreviations | full name                         |
+=============+============================================+
| DAG   | directed acyclic graph                           |
+-------------+--------------------------------------------+
| FPGAs   | field-programmable gate arrays                 |
+-------------+--------------------------------------------+
| SP          | Streaming Processors                       |
+-------------+--------------------------------------------+
| SMP   | Streaming Multi-Processors                       
+-------------+--------------------------------------------+
| SVM   | shared virtual memory                            |
+-------------+--------------------------------------------+
| USM   | unified shared memory                            |
+-------------+--------------------------------------------+
|             |                                            |
+-------------+--------------------------------------------+
