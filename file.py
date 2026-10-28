import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Part A 

#Q1
df = pd.DataFrame({
    "Student": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics": [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms": [13, 15, 9, 17, 10, 14]
})
df = df.set_index("Student")

print("Dataset:")
print(df)
print()


#Q2
print("Dimensions:", df.shape)
print()
print("Descriptive statistics:")
print(df.describe())
print()


#Q3
X = df.values
Z = (X - X.mean(axis=0)) / X.std(axis=0)

print("Standardized data:")
print(pd.DataFrame(Z, index=df.index, columns=df.columns).round(3))
print()


#Q4
R = np.corrcoef(Z, rowvar=False)

print("Correlation matrix:")
print(pd.DataFrame(R, index=df.columns, columns=df.columns).round(4))
print()


#Q5
vals, vecs = np.linalg.eig(R)

print("Eigenvalues:")
print(vals.round(4))
print()
print("Eigenvectors:")
print(vecs.round(4))
print()


#Q6
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

print("Sorted eigenvalues:")
print(vals.round(4))
print()
print("Sorted eigenvectors:")
print(vecs.round(4))
print()


#Q7
W = vecs[:, :2]

print("Two selected eigenvectors:")
print(W.round(4))
print()


#Q8
F = Z @ W

print("Projected data (from scratch):")
print(pd.DataFrame(F, index=df.index, columns=["PC1", "PC2"]).round(4))
print()


#Q9
ratio = vals / vals.sum()
cumulative = np.cumsum(ratio)

print("Explained variance ratio:", ratio.round(4))
print("Cumulative explained variance:", cumulative.round(4))
print()


# Part B 

#Q10
Z_sk = StandardScaler().fit_transform(df)


#Q11
pca = PCA(n_components=2)
F_sk = pca.fit_transform(Z_sk)


#Q12
print("Projected data (sklearn):")
print(pd.DataFrame(F_sk, index=df.index, columns=["PC1", "PC2"]).round(4))
print()


#Q13
ratio_sk = pca.explained_variance_ratio_
cumulative_sk = np.cumsum(ratio_sk)

print("Explained variance ratio:", ratio_sk.round(4))
print("Cumulative explained variance:", cumulative_sk.round(4))
print()


# Part C 

#Q14
print("Coordinates from scratch:")
print(F.round(4))
print()
print("Coordinates from sklearn:")
print(F_sk.round(4))
print()
print("Same coordinates?", np.allclose(np.abs(F), np.abs(F_sk)))
print()


#Q15
print("Eigenvalues from scratch:", vals[:2].round(4))
print("Eigenvalues from sklearn:", pca.explained_variance_.round(4))
print()
print("Ratio from scratch:", ratio[:2].round(4))
print("Ratio from sklearn:", ratio_sk.round(4))
print()


#Q16
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(F[:, 0], F[:, 1])
ax[1].scatter(F_sk[:, 0], F_sk[:, 1])

for i, name in enumerate(df.index):
    ax[0].annotate(name, (F[i, 0], F[i, 1]))
    ax[1].annotate(name, (F_sk[i, 0], F_sk[i, 1]))

ax[0].set_title("PCA from scratch")
ax[1].set_title("PCA with sklearn")

for a in ax:
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")

plt.show()


#Q17
print("Same signs?")
print(np.sign(F) == np.sign(F_sk))
print()


#Q18
print("Variance retained by PC1 and PC2:", round(cumulative[1] * 100, 2), "%")
#Interpretation:
#  Both methods give the same results. 
# The eigenvalues are a bit different because scikit-learn divides by n−1
#PC1 explains 93.49% of the information
# S4 is the best and S3 is the weakest.
#  PC2 explains 6.01% and separates students who are better in Statistics from those better in Programming .
#PC1 and PC2 together keep 99.50% of the information
# so we can go from 4 subjects to 2 components without losing information because of the strong correlation 
