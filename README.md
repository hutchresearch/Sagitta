## The Sagitta Pipeline
Sagitta is a deep neural network based python3 pipeline that relies on Gaia DR2 and 2MASS photometry to identify pre-main sequence (PMS) stars and derive their age estimates.

# Installation:
```pip install sagitta``` (requires Python3)

## Description
Sagitta is a python3 script that takes a Flexible Image Transport System (FITS) file as input. The only required column that must be specified for predictions to be generated is the Gaia EDR3 (or Gaia DR2) source ID column with the ```--source_id``` flag (data release can be specified via ```--version``` flag). All other missing required fields can/will be automatically downloaded when the pipeline is run. If a file is given that contains stars with and without Gaia source IDs, only the stars with values for the source ID will be run through the pipeline. In its default configuration, the pipeline will produce three predictions for each star: 1) a estimation of stellar extinction (Av), 2) the probablilty that a star is PMS (with 0 being 0% probablity and 1 being a 100% probablity), and 3) the estimated age of each star. Once the pipeline has been run and the output table has been automatically saved, the user should look at the output to determine an appropriate PMS output probablity cutoff to create their predicted PMS subset (ie. select pms > 0.8). Due to the nature of how the age model in the pipeline was trained only stars with significantly high PMS model probability output should be considered to have accurate age predictions.

Behing the scenes, Sagitta uses three seperate convolutional neural networks (CNNs) to make its predictions. The first model, denoted as the Av model, is used for generating stellar extcintion (Av) values for stars in the input table. The second model, denoted as the PMS model, is used for generating the probability that each star is pre-main sequence. The thrid model, denoted as the age model, is used for generating the predicted ages for the stars.

## Pipeline Usage Options

#### Flow Control Options

###### Turning off Av, PMS, or age predictions
In the default configuration all three models will be run with their outputs saved as columns in a output FITS file. If specified, the user can choose to not produce outputs from any of these models using the ```--no_av_prediction```, ```--no_pms_prediction```, and ```--no_age_prediction``` flags. However, in order to make PMS or age predictions, Av values must have either been generated with the Av model or the input column's name that holds that Av values should be specified with the input naming option. It is important to note however that the Av values requred for use in the PMS and age models should be generated from the Av model to provide optimal predictions.

###### Only Downloading Data
If you want to only download all of the data required for the use of the pipeline but NOT run any of the models, than you can use the ```--download_only``` flag to perform this action. It will download all required Gaia and 2MASS fields along with their associated errors, parallax, PMRA, PMDEC, PMRA_error, and PMDEC_error for every star with Gaia source ID specified.

###### Single source mode
By default, Sagitta expects a path to the table that would contain source_id of each star. If you are interested in estimating parameters of only one star, instead of a catalog, it is possible to provide source_id as an input with the flag of ```--single_object```.

###### Prediction Uncertainty Statistic Generation
Also included in the pipeline is a uncertainty statistics generator for each of the models predictions. The statistics are generated on a per-star basis by randomly varying the input parameters by their associated errors and analyzing the outputs. The number of times each star is sampled to create these output statistics is an option given to the user but it should be noted that computaional cost scales linearly with the number of times sampled. These uncertainty generators are turned off by default but can be turned on by specifying the ```--av_uncertainty```, ```--pms_uncertainty```, or ```--age_uncertainy``` flags where the number of times to sample each star follows the flag (ie using ```--age_uncertainty 10``` would generate the age model output statistics for each star by sampling each star 10 times, varying the outputs, and analying the predictions). The statistics produced for the model output includes mean, median, standard deviation, variance, minimum, and maximum.

###### Uncertainty Av Scattering Range Option
By default, because the Av values from the Av model don't contain a true uncertainty values, the amount by which they are varied in the PMS and age model uncertainty generation is performed by choosing a random value from a uniform distribution with range +/- 0.1 of the original Av. But because selecting +/- 0.1 was only done based off of current Av model output trends, the size of this range can be specified via the ```--av_scatter_range``` flag.

###### Testing Mode
It is recommended that before running the pipeline on a large set of data, that you first test that the pipeline will execute properly by using the ```--test``` flag. In this mode only the first 10000 stars of the input file will be processed with the pipeline. The output of the test run will be saved by default as "{tableIn}-test-sagitta.fits" so that you can look at the output to make sure that it is as desired.

###### Specifying an Av Input Column
Using the ```--av``` flag to specify an input Av column is ONLY recommended for situtaitons where you already have generated Av values with the pipeline and are specifing that previous output column. If this is the case, then you prevent redundant generation of Av values by using this flag. It should be known though, that in order for the pipeline to produce its best predictions the Av column used should always be generated by the Av model.

#### Data Processing Options

###### GPU Acceleration
If the system you are running the pipeline on has cuda gpu acceleration available then we strongly suggest that you specify that device with the device flag for greatly reduced compute time. For large sets of data using gpu acceleration in the pipeline can cut down runtime by over 1000%.

###### Batch Size
This is a optional tuning parameter that lets the user control the size of the batches that get put through the pipeline's models. It can be scaled up or down depending on system or GPU memory requirements for the machine it is being run on.

###### RA&DEC to L&B Conversion
If in the input table RA and DEC are specified but L and B are both not specified, then the pipeline will automatically convert RA and DEC into L and B and save their values in the output table under their default names.

#### File and Column Naming Options

###### Input File Name Specification
This is the only required input argument for the pipeline to run. It is simply just the path to the input FITS file that predictions should be generated with.

###### Input Column Name Specification
It is recommended that in the the case where the input table already contains any of the required input columns but the column name does not match our default naming schema, that column names should be specified using their associated input column name flag. **If a column is specified but some rows in that column are missing information, the missing data for those rows will not be downloaded and will instead be assigned a default value used for prediction.** The table below shows the input flag name relation. Column names are case insensitive.

| Field                     | Flag Name     | Default Name  |
| :-------------------:     | :-----------: | :-----------: |
| Gaia Source ID            | --source_id   | source_id     |
| Parallax                  | --parallax    | parallax      |
| Galactic Lat.             | --b           | b             |
| Galactic Long.            | --l           | l             |
| Gaia G Mean Magnitude     | --g           | g             |
| Gaia BP Mean Magnitude    | --bp          | bp            |
| Gaia RP Mean Magnitude    | --rp          | rp            |
| 2MASS J Mean Magnitude    | --j           | j             |
| 2MASS H Mean Magnitude    | --h           | h             |
| 2MASS K Mean Magnitude    | --k           | k             |
| Parallax Standard Error   | --eparallax   | eparallax     |
| Gaia G Band Uncertainty   | --eg          | eg            |
| Gaia BP Band Uncertainty  | --ebp         | ebp           |
| Gaia RP Band Uncertainty  | --erp         | erp           |
| 2MASS J Band Uncertainty  | --ej          | ej            |
| 2MASS H Band Uncertainty  | --eh          | eh            |
| 2MASS K Band Uncertainty  | --ek          | ek            |

If a column name is not specified but is in the required list of photometric fields then it will be downloaded and saved in the output table with its default name.

###### Output Fits File Naming Option
The user can specify the name for the output file via the ```--tableOut``` flag. If this flag is not specified then by default the output table will be named {tableIn}-sagitta.fits if NOT in testing model, or {tableIn}-test-sagitta.fits if in testing mode.

###### Output Column Naming Options
There are three flags for output column naming specification. They are the ```--av_out```, ```--pms_out```, and ```--age_out``` flags with their default values being "av", "pms", and "age" respectivly. These names correspond to the output column names from each of the three models, and will also be used in the uncertainty statistic generation output column names as well.

## Examples
Testing all three models in the pipeline on example.fits and renaming the Av and pms output columns:
```sagitta example.fits --av_out av_sagitta --pms_out pms_sagitta  --test```

Running all three models and specifying the output table name to be output.fits:
```sagitta example.fits --tableOut output.fits```

Only running the Av and PMS models:
```sagitta example.fits --no_age_prediction```

Running all three models AND generating the PMS output uncertainty statistics with the sampling rate to 5 times per star:
```sagitta example.fits --pms_uncertainty 5```

Specifying that the example.fits's source ID colum is named Gaia_DR2_ID:
```sagitta example.fits --source_id Gaia_DR2_ID --version dr2 ```

Processing only a single source:
```sagitta Gaia_EDR3_ID --version edr3 --single_object```

Pulling up the terminal help:
```sagitta --help```

## Required Packages
* [AstroPy](https://www.astropy.org/)
* [AstroQuery](https://astroquery.readthedocs.io/)
* [GalPy](https://docs.galpy.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Pytorch](https://pytorch.org/)

## Paper Reference
[Untangling the Galaxy III: Photometric Search for Pre-main Sequence Stars with Deep Learning](https://arxiv.org/abs/2012.10463)

## License
[MIT](./LICENSE)
