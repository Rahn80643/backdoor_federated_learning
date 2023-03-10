**11/20/2020: We are developing a new framework for backdoors with FL: [Backdoors101](https://github.com/ebagdasa/backdoors101).**
It extends to many new attacks (clean-label, physical backdoors, etc) and has improved user experience. Check it out!

# backdoor_federated_learning
This code includes experiments for paper "How to Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)


All experiments are done using Python 3.7 and PyTorch 1.0.

1. Create a virtual environment in conda 
    ```conda env create --file environment.yml```

2. Activate the conda environment
    ```conda activate pytorch_backdoor_FL_test```

3. Create a folder for models to be saved
    ```mkdir saved_models```

4. Open the visdom server for visualizing the training accuracy
    ```python -m visdom.server``` 
    (The server executes in background)

5. Open another terminal, activate the conda environment, and use the following command to train the model
    ```python -u training.py 2>&1 | tee -a log_20230310.log```


I encourage to contact me (eugene@cs.cornell.edu) or raise Issues in GitHub, so I can provide more details and fix bugs. 

Most of the experiments resulted by tweaking parameters in utils/params.yaml (for images) 
and utils/words.yaml (for text), you can play with them yourself.

## Reddit dataset
* Corpus parsed dataset: https://drive.google.com/file/d/1qTfiZP4g2ZPS5zlxU51G-GDCGGr23nvt/view?usp=sharing 
* Whole dataset: https://drive.google.com/file/d/1yAmEbx7ZCeL45hYj5iEOvNv7k9UoX3vp/view?usp=sharing
* Dictionary: https://drive.google.com/file/d/1gnS5CO5fGXKAGfHSzV3h-2TsjZXQXe39/view?usp=sharing


