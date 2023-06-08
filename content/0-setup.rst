.. _setup:

Setup
=====

Local installation
------------------

Since this lesson is taught using an HPC cluster, no software installation on your own computer is needed. 


Running on LUMI
---------------

Interactive job, 1 node, 1 GPU, 1 hour:  

.. code-block:: console

   $ salloc -A project_465000485 -N 1 -t 1:00:0 -p standard-g --gpus-per-node=1
   $ srun <some-command>

Exit interactive allocation with ``exit``.

Interacive terminal session on compute node:

.. code-block:: console

   $ srun --account=project_465000485 --partition=standard-g --nodes=1 --cpus-per-task=1 --ntasks-per-node=1 --gpus-per-node=1 --time=1:00:00 --pty bash
   $ <some-command>

Corresponding batch script ``submit.sh``:

.. code-block:: bash

   #!/bin/bash -l
   #SBATCH --job-name=example-job
   #SBATCH --output=examplejob.o%j
   #SBATCH --error=examplejob.e%j
   #SBATCH --partition=standard-g
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --gpus-per-node=1
   #SBATCH --time=1:00:00
   #SBATCH --account=project_465000485

   srun <some_command> 

Submit the job: ``sbatch submit.sh``

Monitor your job: ``squeue --me``

Kill job: ``scancel <JOB_ID>``

Running Julia on LUMI
^^^^^^^^^^^^^^^^^^^^^

To run Julia with AMDGPU.jl on LUMI:

.. code-block:: console

   $ srun --account=project_465000485 --partition=standard-g --nodes=1 --cpus-per-task=1 --ntasks-per-node=1 --gpus-per-node=1 --time=1:00:00 --pty bash
   
   $ module purge
   $ module use /appl/local/csc/modulefiles
   $ module load julia/1.9.0

Then in Julia session:

.. code-block:: julia

   # only needed in your first Julia session:
   using Pkg
   Pkg.resolve()

   # load AMDGPU
   using AMDGPU

Running on Google Colab
-----------------------

Google Colaboratory, commonly referred to as "Colab", is a cloud-based Jupyter notebook environment which runs in your web browser. Using it requires login with a Google account.

This is how you can get access to NVIDIA GPUs on Colab:

- Visit https://colab.research.google.com/ and sign in to your Google account
- In the menu in front of you, click "New notebook" in the bottom right corner
- After the notebook loads, go to the "Runtime" menu at the top and select "Change runtime type"
- Select "GPU" under "Hardware accelerator" and choose an available type of NVIDIA GPU (e.g. T4)
- Click "Save". The runtime takes a few seconds to load - you can see the status in the top right corner
- After the runtime has loaded, you can type ``!nvidia-smi`` to see information about the GPU.
- You can now write Python code that runs on GPUs through e.g. the numba library.


Access to code examples
-----------------------

Some exercises in this lesson rely on source code that you should download and modify in your own home directory on the cluster. All code examples are available in the same GitHub repository as this lesson itself. To download it you should use Git:

.. code-block:: console

   $ git clone https://github.com/ENCCS/gpu-programming.git
   $ cd gpu-programming/content/examples/
   $ ls

