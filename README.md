# apvbt

Code for paper on "Anonymizing Personalization in Virtual Brain Twins"

Broadly the workflow is

- download/format data
- train the cross-coder
- run simulations
  - for fixed connectomes: "subject-level"
  - for connectomes sampled from cross-coder based prior: "cohort-level"
- run sbi
- compare diagnostics

At the moment, the dynamical regimes and data features used to demo
this are minimal and could be expanded.

## setup

```
pip install matplotlib tqdm vbjax sbi typed-argparse joblib
```

If you have Git LFS installed, you can just clone the repo and the cross coder
will already be trained.
