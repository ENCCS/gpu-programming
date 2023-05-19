.. _gpu-porting:

Preparing code for GPU porting
==============================

.. questions::

   - q1
   - q2

.. objectives::

   - o1
   - o2

.. instructor-note::

   - 20 min teaching
   - 10 min exercises

Taking advantage of the parallel processing capability of the GPUs requires modifying the original code. However some work is required before writing actual code running on the GPUs:

* identify (or decide)  the parts of the code targeted by the porting. These are computational intensive parts of the code such as loops or matrix operations. An observational rule is that 10% of the code takes 90% of the execution time. 
* if a cpu library is used one should identify the equivalent one on the GPUs. For example BLAS library has cu/hipBLAS, or mkl equivalents. 
* when porting a loop directly,  works needs to be done to **refactor** it in a way that is suitable for the GPUs.(example missing here). This involves splitting the loop in several steps or changing some operations to reflect the independence of the operations between different iterations or give a better memory access. Each "step" of the original loop is then mapped to a kernel which is executed many gpu threads, each gpu thread correspoding to an iteration. 

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
             soap_cart_der(1, 1:n_soap, k2) = dsin(thetas(k2)) * dcos(phis(k2)) * soap_rad_der(1:n_soap, k2) - &
                                           dcos(thetas(k2)) * dcos(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) - &
                                           dsin(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
             soap_cart_der(2, 1:n_soap, k2) = dsin(thetas(k2)) * dsin(phis(k2)) * soap_rad_der(1:n_soap, k2) - &
                                           dcos(thetas(k2)) * dsin(phis(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2) + &
                                           dcos(phis(k2)) / rjs(k2) * soap_azi_der(1:n_soap, k2)
             soap_cart_der(3, 1:n_soap, k2) = dcos(thetas(k2)) * soap_rad_der(1:n_soap, k2) + &
                                           dsin(thetas(k2)) / rjs(k2) * soap_pol_der(1:n_soap, k2)

             soap_cart_der(1, 1:n_soap, k3) = soap_cart_der(1, 1:n_soap, k3) - soap_cart_der(1, 1:n_soap, k2)
             soap_cart_der(2, 1:n_soap, k3) = soap_cart_der(2, 1:n_soap, k3) - soap_cart_der(2, 1:n_soap, k2)
             soap_cart_der(3, 1:n_soap, k3) = soap_cart_der(3, 1:n_soap, k3) - soap_cart_der(3, 1:n_soap, k2)
           end if
         end do
       end do

.. keypoints::

   - k1
   - k2
