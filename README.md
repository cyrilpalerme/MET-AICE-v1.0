# MET-AICE
----------------------------------------------------------------------------------------------
MET-AICE archive
----------------------------------------------------------------------------------------------
The MET-AICE forecasts are available on the THREDDS server of the Norwegian Meteorological Institute (https://thredds.met.no/thredds/catalog/aice_files/catalog.html).

----------------------------------------------------------------------------------------------
The repository "Figures" on the master branch
----------------------------------------------------------------------------------------------
This repository contains the codes used for generating the figures of the article.

----------------------------------------------------------------------------------------------
The repository "Operational_predictions" on the master branch
----------------------------------------------------------------------------------------------
This repository contains the codes used for making the predictions in the operational production chain. The operational production chain is run in a ecFlow environment (https://ecflow.readthedocs.io/en/).

----------------------------------------------------------------------------------------------
The repository "Standardization" on the master branch
----------------------------------------------------------------------------------------------
This repository contains the codes used for calculating the standardization (normalization) statistics of the predicor and target variables. 

----------------------------------------------------------------------------------------------
The repository "Training_data" on the master branch
----------------------------------------------------------------------------------------------
This repository contains the codes used for generating the datasets used for training and evaluating the deep learning models.

----------------------------------------------------------------------------------------------
The repository "Train_model" on the master branch
----------------------------------------------------------------------------------------------
This repository contains the codes used for developing the deep learning models. This contains the codes for training the deep learning models, and for making predictions during model development.

----------------------------------------------------------------------------------------------
The repository "Verification" on the master branch
----------------------------------------------------------------------------------------------
This repository contains the codes used for generating the verification statistics (verification scores). The scores are written in text files that can be read using the scripts from the "Figures" repository. There is also a script for generating a shared land sea mask between the datasets and for calculating the distance to land.


