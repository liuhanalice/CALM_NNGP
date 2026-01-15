
## Launch Experiments

Example Command:
```
python -u CL_Driver.py --no_checkpoint --log_every_epoch
```

Alteratively, dump the command line prints to a file, e.g. run.txt:
(Note: Simply dump SLURM job output to a log file when writing shell script)

* Windows:

    ```
    python -u CL_Driver.py *> run.txt 2>&1   
    ```

* Mac:

    ``` 
    python -u CL_Driver.py > run.txt 2>&1  
    ```

Example:

```
python -u CL_Driver.py --no_checkpoint --log_every_epoch --GP_train_size_per_class=100 --GP_test_size_per_class=200 --GP_train_otc_size=50 --GP_num_indcpts=100 --GP_package=gplite *> run.txt 2>&1 

python -u CL_Driver.py --no_checkpoint --log_every_epoch --epochs=30 --lambda_rec=1.0 --GP_train_size_per_class=2000 --GP_test_size_per_class=1000 --GP_train_otc_size=50 --GP_num_indcpts=1000 --GP_package=laGP *> run.txt 2>&1 
```

## Environment Setup
* CUDA: 12.6.3
* R: 4.5.1
