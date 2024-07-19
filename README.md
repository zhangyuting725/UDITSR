Code for "Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation" in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2024)

### Data

**Source Data**: download MT-Small data for UDITSR from https://drive.google.com/drive/folders/1Q7N5ppK7xvSud_rLGuYvoS-sgu8d0ZQS

**Structure**:

> code

> > models
> >
> > > BaseP.py
> > >
> > > UDITSR.py
> >
> > conf.py
> >
> > main.py
> >
> > utils.py

> UDITSR_data
>
> > train_data.csv
> >
> > valid_data.csv
> >
> > test_data.csv



**data format**:

- train_data

  | user | query | Item | label |
  | ---- | ----- | :--: | ----- |
  | 0    | []    |  0   | 1     |
  | 1    | []    |  1   | 1     |
  | 2    | [6]   |  2   | 1     |

- test/valid data. For each ground truth valid/test data, we randomly select 99 item that the user has not interacted with, serving as negative samples. Records sharing the same 'sample num' indicate that they are either a single positive sample or the negative samples derived based on the positive sample.

  | sample_num | user | query  | Item | label |
  | ---------- | ---- | ------ | :--: | ----- |
  | 1          | 7    | [42,6] |  7   | 1     |
  | 2          | 14   | []     | 3461 | 0     |
  | 3          | 19   | []     | 678  | 0     |

```
cd code
python3 main.py
```

