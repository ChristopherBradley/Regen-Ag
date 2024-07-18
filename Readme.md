Aggregating scripts for collecting data from different sources

### Environment Creation
-  conda create -n geoharvest python=3.12
-  conda install geodata-harvester -c conda-forge
-  pip install jupyter podpac webob
-  pip install podpac[datatype]
-  pip install podpac[aws]
  
### Using Jupytext
- conda activate geoharvester
- jupyter-notebook
- right click the notebook and open-with jupyter notebook