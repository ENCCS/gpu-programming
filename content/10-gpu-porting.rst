.. _gpu-porting:

Preparing code for GPU porting
==============================

.. questions::

   - What are the key steps involved in porting code to take advantage of GPU parallel processing capability?
   - How can I identify the computationally intensive parts of my code that can benefit from GPU acceleration?
   - What are the considerations for refactoring loops to suit the GPU architecture and improve memory access patterns?

.. objectives::

   - Provide a basic understanding of the steps involved in porting code to GPUs to take advantage of their parallel processing capability.
   - Provide guidelines for porting code to GPUs

.. instructor-note::

   - 20 min teaching
   - 10 min exercises

When porting code to take advantage of the parallel processing capability of GPUs , several steps need to be followed. These steps help identify the computationally intensive parts of the code, refactor loops for GPU execution, and optimize memory access. Let's go through the steps:

Identify Targeted Parts

Taking advantage of the parallel processing capability of the GPUs requires modifying the original code. However some work is required before writing actual code running on the GPUs:

* **Identify Targeted Parts**: Begin by identifying the parts of the code that contribute significantly to the execution time. These are often computationally intensive sections such as loops or matrix operations. The Pareto principle suggests that roughly 10% of the code accounts for 90% of the execution time.

* **Equivalent GPU Libraries**: If the original code uses CPU libraries like BLAS, FFT, etc, it's crucial to identify the equivalent GPU libraries. For example, cuBLAS or hipBLAS can replace CPU-based BLAS libraries. Utilizing GPU-specific libraries ensures efficient GPU utilization.

* **Refactor Loops**: When porting loops directly to GPUs, some refactoring is necessary to suit the GPU architecture. This typically involves splitting the loop into multiple steps or modifying operations to exploit the independence between iterations and improve memory access patterns. Each step of the original loop can be mapped to a kernel, executed by multiple GPU threads, with each thread corresponding to an iteration.

* **Memory Access Optimization**: Consider the memory access patterns in the code. GPUs perform best when memory access is coalesced and aligned. Minimizing global memory accesses and maximizing utilization of shared memory or registers can significantly enhance performance. Review the code to ensure optimal memory access for GPU execution.

Discussion
^^^^^^^^^^
 .. challenge:: Example: ``How would this be ported?``
 
.. tabs:: 

   .. tab:: Fortran

      .. code-block:: Fortran

         k2 = 0
         do i = 1, n_sites
           do j = 1, n_neigh(i)
           k2 = k2 + 1
           counter = 0 
           counter2 = 0
           do n = 1, n_max
             do np = n, n_max
               do l = 0, l_max
                 if( skip_soap_component(l, np, n) )cycle
               counter = counter+1
               do m = 0, l
                 k = 1 + l*(l+1)/2 + m
                 counter2 = counter2 + 1 
                 multiplicity = multiplicity_array(counter2)
                 soap_rad_der(counter, k2) = soap_rad_der(counter, k2) + multiplicity * real( cnk_rad_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_rad_der(k, np, k2)) )
                 soap_azi_der(counter, k2) = soap_azi_der(counter, k2) + multiplicity * real( cnk_azi_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_azi_der(k, np, k2)) )
                 soap_pol_der(counter, k2) = soap_pol_der(counter, k2) + multiplicity * real( cnk_pol_der(k, n, k2) * conjg(cnk(k, np, i)) + cnk(k, n, i) * conjg(cnk_pol_der(k, np, k2)) )
                 end do
               end do
             end do
           end do
           
           soap_rad_der(1:n_soap, k2) = soap_rad_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_rad_der(1:n_soap, k2) )
           soap_azi_der(1:n_soap, k2) = soap_azi_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_azi_der(1:n_soap, k2) )
           soap_pol_der(1:n_soap, k2) = soap_pol_der(1:n_soap, k2) / sqrt_dot_p(i) - soap(1:n_soap, i) / sqrt_dot_p(i)**3 * dot_product( soap(1:n_soap, i), soap_pol_der(1:n_soap, k2) )
          
           if( j == 1 )then
             k3 = k2
           else
             soap_cart_der(1, 1:n_soap, k2) = dsin(thetas(k2)) * dcos(phis(k2)) * soap_rad_der(1:n_soap, k2) - dcos(thetas(k2)) * dcos(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) - dsin(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
             soap_cart_der(2, 1:n_soap, k2) = dsin(thetas(k2)) * dsin(phis(k2)) * soap_rad_der(1:n_soap, k2) - dcos(thetas(k2)) * dsin(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) + dcos(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
             soap_cart_der(3, 1:n_soap, k2) = dcos(thetas(k2)) * soap_rad_der(1:n_soap, k2) + dsin(thetas(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2)

             soap_cart_der(1, 1:n_soap, k3) = soap_cart_der(1, 1:n_soap, k3) - soap_cart_der(1, 1:n_soap, k2)
             soap_cart_der(2, 1:n_soap, k3) = soap_cart_der(2, 1:n_soap, k3) - soap_cart_der(2, 1:n_soap, k2)
             soap_cart_der(3, 1:n_soap, k3) = soap_cart_der(3, 1:n_soap, k3) - soap_cart_der(3, 1:n_soap, k2)
           end if
         end do
       end do


Some steps at first glance:
- the code could (has to) be splitted in 3 kernels. Why? 
- check for false dependencies. Analyze if there are any variables that could lead to false dependencies between iterations, like the index k2`
- is it efficient for GPUs to split the work over the index `i`? What about the memory access?
- is it possible to collapse some loops? Combining nested loops can reduce overhead and improve memory access patterns, leading to better GPU performance.
- what is the best memory access in a GPU? Review memory access patterns in the code. Minimize global memory access by utilizing shared memory or registers where appropriate. Ensure memory access is coalesced and aligned, maximizing GPU memory throughput



.. keypoints::

   - identify equivalent GPU libraries for CPU-based libraries and utilizing them to ensure efficient GPU utilization
   - importance of identifying the computationally intensive parts of the code that contribute significantly to the execution time
   - the need to refactor loops to suit the GPU architecture 
   - significance of memory access optimization for efficient GPU execution, including coalesced and aligned memory access patterns
   
   
