# Human-AI Collaboration with Misaligned Preferences

<summary>Table of Contents</summary>
<ol>
<li>
    <a href="#about-the-project">About The Project</a>
</li>
<li>
    <a href="#getting-started">Getting Started</a>
</li>
<li><a href="#usage">Usage</a></li>
<li><a href="#exporting-reports">Exporting Reports</a></li>
</ol>

## About The Project
The project consists of a set of experiments for the benefits of human received in the human-AI collaboration when both human and AI are noisy and may have misaligned preferences. The project conducts a set of experiments, including 

* Comparison between the performance of different AI algorithms;
* Performance of the linear mixed integer programming for finding the best algorithm;
* Tension between different welfare objectives (social welfare, uplift)

## Getting Started
The implementation is based on Python with version of `3.12.2` and make sure the following dependencies are installed:

```shell
$ pip3 install gurobipy==12.0.2 matplotlib==3.9.2 numpy==1.26.4 pandas==2.2.2
```

We note that the current experimental results on the evaluation of MIP are based on the [Gurobi](https://www.gurobi.com/) optimizer with an academic license. We have saved all the metadata of the results in the folder of `figs/` in `.json` format. To load the saved results, enable the `from_stored` option in the file `plot.py`; to rerun the MIP locally, disable it.

## Usage
We have provided a set of plotting functions in the file `plot.py`. The results generated using these functions can be found in `misalign.ipynb`. The notebook consists of the following experiments:
| Description | Figure Index |
|-------------|--------------|
|Comparison between Algorithm A1 and A3 | Fig. 3 |
|Comparison of human’s expected utility differences after collaboration with a misaligned and an aligned algorithm | Fig. 4 |
|Comparison of human’s expected utility differences after collaboration with a misaligned and an aligned algorithm (under RUM) | Fig. 5 |
|Utility gain from collaboration across different algorithm ranking | Fig. 6 |
| Utility gain from collaboration across misaligned algorithms and aligned algorithm (under RUM) | Fig. 7 |
|Running time of MIP (varying m) | Fig. 8 |
|Running time of MIP (varying n) | Fig. 9 |
|Tension between social welfare and Uplift | Fig. 10 to 13 |
|Smallest noise for uplift | Fig. 14 |




## Exporting Reports
The Jupyter notebook can also be exported as a pdf file via the following command (please make sure jupyter is installed)

```
$ jupyter nbconvert --to pdf misalign.ipynb --output report/misalign.pdf
```

We also provide an exported pdf at `report/misalign.pdf`.
