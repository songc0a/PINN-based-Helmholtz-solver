# PINN-based-Helmholtz-solver
**This repository reproduces the results of the paper "[Solving the frequency-domain acoustic VTI wave equation using physics-informed neural networks.](https://academic.oup.com/gji/article/225/2/846/6081098)" Geophysical Journal International 225, 846-859;  the results of the abstract "[Machine learned Green's functions that approximately satisfy the wave equation](https://library.seg.org/doi/abs/10.1190/segam2020-3421468.1)", SEG International Exposition and 90th Annual Meeting, 2638-2642;  the results of the abstract "[Wavefield solutions from machine learned functions that approximately satisfy the wave equation](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202010588)", Conference Proceedings, EAGE 2020 Annual Conference & Exhibition Online, Dec 2020, Volume 2020, p.1 - 5**

# Overview

We propose to use the physics Informed Neural Network (PINN) to solve the scattered form of acoustic isotropic and anisotropic wave equations

PINN reduces the computational cost by avoid computing the inverse of the impedance matrix, which is suitable for anisotropic large models 

The resulting scattered wavefields are free of numerical dispersion artifacts

![image](https://user-images.githubusercontent.com/31889731/116651995-f7a17000-a98c-11eb-9ef1-b18e4562f661.png)

# Installation of Tensorflow1

CPU usage: pip install --pre "tensorflow==1.15.*"

GPU usage: pip install --pre "tensorflow-gpu==1.15.*"
