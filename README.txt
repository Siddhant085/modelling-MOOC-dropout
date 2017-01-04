0.  Team Members - Pratul Ramkumar 01FB14ECS162
                 - Siddhant Gupta  01FB14ECS240
                 - Yash Agarwal    01Fb14ECS295

    Topic        - Predicting MOOC Dropouts

1. The code is in the file named code.py, The entire code is written with respect to the python 2.7 compiler

2. Additional python libraries to be installed -
-> pandas - enables use of data frames 
-> numpy - provides powerful multidimensional array processing
-> sklearn - provides pre-written machine learning algorithms
-> matplotlib - a great visualisation tool

- The packages above can be downloaded using the 'pip' software of python. 
    sudo apt-get install python-pip (will install pip for python 2 - on Ubuntu ONLY)
- The packages can then be installed using the following commands (On any Operating System)
    pip install pandas    
    pip install numpy
    pip install -U scikit-learn
    pip install matplotlib

3. Extract all the data files from data.zip and place each file in in the main folder - that is the folder consisting this README file. The directory for the data files, should be the same as that of code.py

4. DO NOT run the entire code at once. The dataset is very large - Close to eighty lakh rows have been processed in some parts of the code, it is hence advisable to run the code in parts. The code has been delimited by 35-pound symbols(#), which act as the comments on parts of the code. It is best to run the code between two of these markers at once.

5. However, there is often a continuity across the delimiters in terms of variables and data structures. So the conventional way to run python programs from the terminal using
    python file.txt 
should be avoided

6. Use ipython notebooks instead, which is an interactive and neat environment to write and run code. The notebook opens up in the browser and runs on the localhost. 
jupyter notebooks for python can be downloaded using pip
    sudo -H pip install jupyter

7. Parts of the code can then be copied to the notebook and run.

8. If the code is to be run somewhere from the middle, extract the files from the generated_files.zip folder and run the code from wherever desired. Every delimited section has read in the required files, hence there should not be any problems in running the code staring from any delimited section.
