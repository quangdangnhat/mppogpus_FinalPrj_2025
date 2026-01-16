# mppogpus_FinalPrj_2025
We implement five CUDA versions of Sciara-fv2 lava flow simulator: Global, Tiled, Tiled+Halo, CfAMe, and CfAMo. Roofline analysis on GTX 980 shows all versions are memory-bound (AI &lt; 0.05). CfAMo achieves 1.09× speedup via kernel fusion.
