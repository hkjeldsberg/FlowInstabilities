## FlowInstabilities

<p align="center">
    <img src="https://i.ibb.co/ZGWQSp8/models.png" width="840 height="480" alt="Models used in code"/>
</p>
<p align="center">
    The five arterial models use to compute flow perturbation and corresponding eigenvalues.
</p>

Description
-----------
FlowInstabilities is a framework for studying flow instabilities through the Reynolds-Orr method. 
Initially, the Navier-Stokes equations are solved to find the base flow in the given geometry, before a eigenvalue problem is defined and solved for the perturbed flow and its corresponding eigenvalues.
In the current implementation the method is applied to cerebral arteries, to understand the risk of rupture in [cerebral aneurysms](https://en.wikipedia.org/wiki/Intracranial_aneurysm), including a simple cylinder case to demonstrate Poiseuille flow.
 The code is also applicable to other geometries in order to understand flow instabilities. 
 
Installation
------------

For reference, FlowInstabilities requires the following dependencies: FEniCS >= 2018.1.0, Numpy >= 1.13 and SLEPc4Py >= 3.0.0.
To run the cylinder case (Poiseuille flow) there is an additional dependency: mshr >= 2018.  
If you are on Windows, macOS or Linux you can install all the general dependencies through anaconda.

Usage
-----
Inside a FEniCS/SLEPc environment, execute the following command to run the main script for case 0, kinematic viscosity of 0.1 and a pressure drop of 5 mmHg

        python find_pertubation.py --case 0 --nu 0.1 --delta_p 5

The results will be located in the `Eigenmodes` folder.
Alternatively, you can solve explicitly for the base flow by executing the following command.

        python find_baseflow.py

The results will now be saved in the `Baseflow` folder.
