# pip install matplotlib-venn
from matplotlib_venn import venn3_unweighted, venn3_circles
from matplotlib import pyplot as plt

# Circle 1 is ML4SE
# Circle 2 is dataset complexity
# Circle 3 is HPT
fig, ax = plt.subplots()
venn3_unweighted(subsets=(9, 0, 4, 0, 6, 0, 0),
                 set_labels = ("ML for SE", "Dataset\ncomplexity", "Hyper-parameter tuning"),
                 set_colors = ("r", "g", "b"), alpha = 0.35)
fig.set_size_inches(10, 5)
fig.savefig("venn.png", dpi=300)
