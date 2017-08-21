```
Sample input
```

```
  +----+---+---+
  | u0 | A | 2 |
  +----+---+---+
  | u0 | C | 1 |
  +----+---+---+
  | u1 | A | 7 |
  +----+---+---+
  | u1 | B | 1 |
  +----+---+---+
  | u2 | A | 2 |
  +----+---+---+
  | u2 | C | 1 |
  +----+---+---+
  | u0 | B | 2 |
  +----+---+---+
  Format is (user, item,  pref).

MapReduce1
  (user, item, pref) =
>
 (user, [(item, pref)])

  (u0, [(A, 2), (B, 2), (C, 1)])
  (u1, [(A, 7), (B, 1)])
  (u2, [(A, 2), (C, 1)])


MapReduce2:
  Map:
  (user, [(item, pref)]) =
>
 {(item1, item2): (pref1, pref2)}

  {(A, B): (2, 2)}
  {(A, C): (2, 1)}
  {(B, C): (2, 1)}
  ----------------
  {(A, B): (7, 1)}
  ----------------
  {(A, C): (2, 1)}

  Group:

  Reduce:
  {(item1, item2): [(pref1, pref2)]} =
>
 
  {(item1,  sim): (item1,  item2,  sim)}
  {(item2,  sim): (item1,  item2,  sim)}

  Input:
  {(A, B): [(2, 2), (7, 1)]}
  ----------------
  {(A, C): [(2, 1), (2, 1)]}
  ----------------
  {(B, C): [(2, 1)]}

  Output:
  The sim is faked. For sample size 2, Pearson Coefficient is always 1 or -1.  
  For data is needed to illustrate similarities.

  {(A, 0.5): (A, B, 0.5)}
  {(B, 0.5): (B, A, 0.5)}
  {(A, 0.4): (A, C, 0.4)}
  {(C, 0.4): (C, A, 0.4)}
  {(B, 0.3): (B, C, 0.3)}
  {(C, 0.3): (C, B, 0.3)}

MapReduce3:
  group on item,  sort on (item,  sim)
```



