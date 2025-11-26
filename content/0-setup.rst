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

   $ salloc -A project_465002387 -N 1 -t 1:00:00 -p standard-g --gpus-per-node=1
   $ srun <some-command>

Exit interactive allocation with ``exit``.

Interacive terminal session on compute node:

.. code-block:: console

   $ srun --account=project_465002387 --partition=standard-g --nodes=1 --cpus-per-task=1 --ntasks-per-node=1 --gpus-per-node=1 --time=1:00:00 --pty bash
   $ <some-command>

Corresponding batch script ``submit.sh``:

.. code-block:: bash

   #!/bin/bash -l
   #SBATCH --account=project_465002387
   #SBATCH --job-name=example-job
   #SBATCH --output=examplejob.o%j
   #SBATCH --error=examplejob.e%j
   #SBATCH --partition=standard-g
   #SBATCH --nodes=1
   #SBATCH --gpus-per-node=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --time=1:00:00

   srun <some_command> 

- Submit the job: ``sbatch submit.sh``
- Monitor your job: ``squeue --me``
- Kill job: ``scancel <JOB_ID>``



Running Julia on LUMI
^^^^^^^^^^^^^^^^^^^^^

In order to run Julia with ``AMDGPU.jl`` on LUMI, we use the following directory structure and assume it is our working directory.

.. code-block:: console

	.
	├── Project.toml  # Julia environment
	├── script.jl     # Julia script
	└── submit.sh     # Slurm batch script

An example of a ``Project.toml`` project file.

.. code-block:: console

	[deps]
	AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

For the ``submit.sh`` batch script, include additional content to the batch script mentioned above.

.. code-block:: bash

   #SBATCH --cpus-per-task=2
   #SBATCH --mem-per-cpu=1750

   module use /appl/local/csc/modulefiles

   module load julia
   module load julia-amdgpu

   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   julia --project=. script.jl

An example of the ``script.jl`` code is provided below.

.. code-block:: julia

   using AMDGPU
   
   A = rand(2^9, 2^9)
   A_d = ROCArray(A)
   B_d = A_d * A_d

   println("----EOF----")



Running Python
--------------

On LUMI
^^^^^^^

A singularity container containing all the necessary dependencies has been created.
To launch the container and the ``IPython`` interpreter within it, do as follows:

.. code-block:: console

   $ salloc -p dev-g -A project_465002387 -t 1:00:00 -N 1 --gpus=1
   $ cd /scratch/project_465002387/containers/gpu-programming/python-from-docker
   $ srun --pty singularity exec --no-home container_numba_hip_fixed.sif bash
   Singularity> . $HOME/.local/bin/env
   Singularity> . /.venv/bin/activate 
   Singularity> ipython


.. admonition:: Recipe for creating the container
   :class: dropdown

   For reference, the following files were used to create the above singularity container.
   First a singularity def file,

   .. code-block:: singularity

      Bootstrap: docker
      From: rocm/dev-ubuntu-24.04:6.4.4-complete

      %environment
          CUPY_INSTALL_USE_HIP=1
          ROCM_HOME=/opt/rocm
          HCC_AMDGPU_TARGET=gfx90a
          LLVM_PATH=/opt/rocm/llvm

      %post
          export CUPY_INSTALL_USE_HIP=1
          export ROCM_HOME=/opt/rocm
          export HCC_AMDGPU_TARGET=gfx90a
          export LLVM_PATH=/opt/rocm/llvm
          export PATH="$HOME/.local/bin/:$PATH"

          apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git
          curl -L https://astral.sh/uv/install.sh -o /uv-installer.sh 
          sh /uv-installer.sh && rm /uv-installer.sh

          . $HOME/.local/bin/env

          uv python install 3.12
          uv venv -p 3.12 --seed
          uv pip install --index-strategy unsafe-best-match -r /tmp/envs/requirements.txt
          uv pip freeze >> /tmp/envs/requirements_pinned.txt

          touch /usr/lib64/libjansson.so.4 /usr/lib64/libcxi.so.1 /usr/lib64/libjansson.so.4
          mkdir /var/spool/slurmd /opt/cray 
          mkdir /scratch /projappl /project /flash /appl

   and a bash script to build the container,

   .. code-block:: bash

      #!/bin/sh
      ml purge
      ml LUMI/24.03 partition/G
      ml load systools/24.03  # For proot

      export SINGULARITY_CACHEDIR="$PWD/singularity/cache"
      export SINGULARITY_TMPDIR="$FLASH/$USER/singularity/tmp"
      singularity build -B "$PWD":/tmp/envs --fix-perms --sandbox container_numba_hip_fixed.sif build_singularity.def
      
   and finally a ``requirements.txt`` file::

          jupyterlab
          jupyterlab-git
          nbclassic
          matplotlib
          numpy
          cupy
          # jax[rocm]
          jax-rocm7-pjrt @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_pjrt-0.6.0-py3-none-manylinux_2_28_x86_64.whl
          jax-rocm7-plugin @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jax_rocm7_plugin-0.6.0-cp312-cp312-manylinux_2_28_x86_64.whl
          jaxlib @ https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.6.0/jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl
          jax==0.6.2
          --extra-index-url https://test.pypi.org/simple
          numba-hip[rocm-6-4-4] @ git+https://github.com/ROCm/numba-hip.git


On Google Colab
^^^^^^^^^^^^^^^

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

