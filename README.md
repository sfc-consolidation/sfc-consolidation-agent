## Agent


### Run

```bash
$ conda create -n alpha-consolidation python=3.8.16
$ conda activate alpha-consolidation
$ poetry install
$ poetry run uvicorn app.main:app
```


### Dev

```bash
$ conda create -n alpha-consolidation python=3.8.16
$ conda activate alpha-consolidation
$ poetry install
$ poetry run uvicorn app.main:app --reload
```
