# deepcell_pipeline
* A command line tool to help user segment cells with Deepcell.
* Cell track along z-stack to better analyze bioinformatics.

## Install
Access https://github.com/vanvalenlab/deepcell-tf to install deepcell from their ```requirements.txt```.  
Make sure you install imageio, pandas and matplotlib as well.

## Usage
Type ```python pipeline.py --help``` in command line to get what you want.  
Example: ```python pipeline.py -i ./raw_img/roundh11_A1/ -f png -o ./pipe_test/pred/ -c ./pipe_test/track/ -m 0.1 -g 1```
