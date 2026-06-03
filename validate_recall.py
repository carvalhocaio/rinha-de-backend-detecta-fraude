import faiss
import numpy as np

labels = np.load("data/labels.u8.npy")
approx = faiss.read_index("data/index.faiss")
assert isinstance(approx, faiss.IndexIVF)
approx.nprobe = 16

n = approx.ntotal
rng = np.random.default_rng(7)
sample_ids = rng.choice(n, 3_000, replace=False)
xq = approx.reconstruct_batch(sample_ids)

exact = faiss.IndexFlatL2(approx.d)
exact.add(approx.reconstruct_n(0, n))
_, Ie = exact.search(xq, 5)
_, Ia = approx.search(xq, 5)

fse = labels[Ie].sum(1) / 5
fsa = labels[Ia].sum(1) / 5
print(
    f"recall@5 ............... {np.mean([len(set(Ie[i]) & set(Ia[i])) / 5 for i in range(len(xq))]):.1%}"
)
print(f"DECISÃO (approved) bate  {np.mean((fse < 0.6) == (fsa < 0.6)):.2%}")
