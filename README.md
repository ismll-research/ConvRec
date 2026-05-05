* raw dataset directory: `./raw/`
    * put [CARCA/Data](https://github.com/ahmedrashed-ml/CARCA) as `./raw/CARCA/`
* data directory: `./data/`

Run preprocessing:

```bash
python preprocess.py prepare --dname fashion
python preprocess.py prepare --dname beauty
python preprocess.py prepare --dname men
python preprocess.py prepare --dname game
```


Run the code: 

```bash
python entry.py convrec/men
```
