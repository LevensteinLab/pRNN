

## Install package with pip install

Install pRNN package (with torch):

```
pip install -e . 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```


You can omit the flag ```-e```, if you don't intend on editing the package. 

To test if the package installation is successful, run:

    python test/test_imports.py

