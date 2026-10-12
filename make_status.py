import pickle, time, collections, sys
from goconcepts.statusgen import generate
n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
t = time.time()
C, rep = generate(n)
pickle.dump(C, open("data/status.pkl", "wb"))
print(len(C), "cases", round(time.time() - t), "s")
print(sorted(collections.Counter((c.variant, c.status, c.ring_libs, (c.shared, c.eyes, c.outside)) for c in C).items()))
