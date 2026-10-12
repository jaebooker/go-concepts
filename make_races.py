import pickle, time, collections
from goconcepts.racegen import generate
t = time.time()
C = generate(24, seed=2026)
pickle.dump(C, open("data/races.pkl", "wb"))
print(len(C), "cases", round(time.time() - t), "s")
print(sorted(collections.Counter((c.variant, c.shared, c.eyes, c.outside, c.ring_libs) for c in C).items()))
print("ring size median by variant", {v: sorted(c.ring_size for c in C if c.variant == v)[len(C) // 4] for v in (0, 1)})
