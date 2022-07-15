This is the code used to produce the results and visualizations published in

Schmittwilken, L. & Maertens, M. (2022). Fixational eye movements enable robust edge detection. Journal of Vision, 22(5). [doi:10.1167/jov.22.8.5](https://doi.org/10.1167/jov.22.8.5)

## Description
The repository contains the following:

- The data from the psychophysical experiment of Betz et al. (2015) and the Contour Image Database by Grigorescu et al. (2003) that is used as test cases of the model: [databases](databases)

- Two Jupyter-Notebooks with a step-by-step guide through the proposed active edge detection model [active_edge-model.ipynb](jupyter-notebooks/active_edge-model.ipynb) and a demonstration of how spatial edge models work [spatial_edge-models.ipynb](jupyter-notebooks/spatial_edge-models.ipynb)

- Code to create the results shown in the paper: [simulations](simulations). To reproduce the results of test case 1 (edge detection in narrowband noise), run [main_case1.py](simulations/main_case1.py). To reproduce the results of test case 2 (contour detection in natural images), run [main_case2.py](simulations/main_case2.py)

- Code to create the visualizations from the manuscript: [visualize_results](visualize_results). In order to re-create the visualizations, first run the simulations to produce the respective results.

## Authors and acknowledgment
Code written by Lynn Schmittwilken (l.schmittwilken@tu-berlin.de)

## References
Betz, T., Shapley, R., Wichmann, F. A., & Maertens, M. (2015a). Noise masking of White's illusion exposes the weakness of current spatial filtering models of lightness perception. Journal of Vision, 15(14), 1, doi:10.1167/15.14.1

Grigorescu, C., Petkov, N., & Westenberg, M. A. (2003). Contour detection based on nonclassical receptive field inhibition. IEEE Transactions on Image Processing, 12(7), 729–739, doi:10.1109/TIP.2003.814250

