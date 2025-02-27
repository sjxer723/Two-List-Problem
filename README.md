# Human-Algorithm Collaboration

## Demo
Here is an example of constructing distributions and a human-ai system:

```Python
from model.mallows import Mallows
from human_ai import HumanAI

m = 5
D_a = Mallows(m, 1)
D_h = Mallows(m, 1)

joint_system = HumanAI(m, D_a, D_h)

## Simulate one decision-making by the system
k = 2
joint_system.simulate_pick_and_choose(k, verbose=True):
```


## Exporting reports

```
$ jupyter nbconvert --to pdf misalignment.ipynb --output report/misalignment.pdf 
$ jupyter nbconvert --to pdf layout.ipynb --output report/layout.pdf 
```