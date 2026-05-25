# Introduction
This is a graduate design project. The system aims to predict urban growth using deep learning, based on the influential Graph Attention Network (GAT) framework. The entire pipeline integrates ArcGIS and Python.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Getting Started
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps. 
## Prerequisites
This project requires the following versions of Python:

* Python == 3.11
* Python == 2.7

Python 2.7 is the version bundled with ArcMap 10.8. Installation of this software is required to run this project.

The example dataset has been uploaded to：[https://doi.org/10.5281/zenodo.20374140](https://doi.org/10.5281/zenodo.20374140)

## Installation
1. Create a Python 3.11 virtual environment
   ```sh
   conda create -n urban_growth_demo python=3.11 && conda activate urban_growth_demo
   ```
2. Clone the repo
   ```sh
   git clone [https://github.com/duanjinxinchina-eng/Urban-Growth-Prediction-System.git](https://github.com/duanjinxinchina-eng/Urban-Growth-Prediction-System.git)
   ```
3. Install necessary dependencies
   ```sh
   pip install -r requirements.txt
   ```
4. Fill in the path of python 3.11 in system.py
   ```sh
   python3_path = r" Change to your own Python 3 path"
   ```
5. Switch to the Python 2.7 compiler
6. run main.py
   ```sh
   python main.py
   ```

By following these steps, you can quickly get started with Urban-Growth-Prediction-System
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Usage Examples
- [ ] Main interface
![main interface.png](usage_example%2Fmain%20interface.png)
- [ ] Select work space
![select work space.png](usage_example%2Fselect%20work%20space.png)
- [ ] Select study area
![select study area.png](usage_example%2Fselect%20study%20area.png)
- [ ] Load data
![open load files window.png](usage_example%2Fopen%20load%20files%20window.png)
Load the land use maps for the initial and final years, as well as various characteristic data.
![load data.png](usage_example%2Fload%20data.png)
- [ ] Preprocess data. Generate adjacency matrix, feature data table, etc.
![preprocess data.png](usage_example%2Fpreprocess%20data.png)
- [ ] Run model. Enter the starting year, ending year and the number of iterations. Generally, an iteration is conducted once per year.
![run model.png](usage_example%2Frun%20model.png)
- [ ] Statistic
![statistic.png](usage_example%2Fstatistic.png)
- 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Roadmap
![roadmap.png](usage_example%2Froadmap.png)
![model.png](usage_example%2Fmodel.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

# Contact

1481405657@qq.com

Project Link: [https://github.com/duanjinxinchina-eng/Urban-Growth-Prediction-System](https://github.com/duanjinxinchina-eng/Urban-Growth-Prediction-System)
<p align="right">(<a href="#readme-top">back to top</a>)</p>
