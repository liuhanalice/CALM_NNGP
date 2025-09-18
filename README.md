
## Launch Experiments

Example Command:
```
python -u CL_Driver.py --no_checkpoint --log_every_epoch
```

```
python -u CL_Driver.py --no_checkpoint --log_every_epoch --GP_train_size_per_class=100 --GP_test_size_per_class=200 --GP_train_otc_size=50 --GP_num_indcpts=100 --GP_package=gplite > run.txt 2>&1 
```

Alteratively, dump the command line prints to a file, e.g. run.txt:

* Windows:

    ```
    python -u CL_Driver.py *> run.txt 2>&1   
    ```

* Mac:

    ``` 
    python -u CL_Driver.py > run.txt 2>&1  
    ```
