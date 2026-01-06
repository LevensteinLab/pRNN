

## Install package with pip install

Install pRNN package (with torch):

Navigate to the repository folder on your machine. Then,
```
pip install -e . 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```


You can omit the flag ```-e```, if you don't intend on editing the package. 

To test if the package installation is successful, run:

    python test/test_imports.py


You can find a getting started tutorial at our [readthedocs](https://prnn.readthedocs.io/en/latest/quickstart.html).
