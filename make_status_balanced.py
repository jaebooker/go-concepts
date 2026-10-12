import pickle, time, collections, sys
from goconcepts.statusgen import generate_balanced
n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
t = time.time()
C = generate_balanced(n)
pickle.dump(C, open("data/status_bal.pkl", "wb"))
print(len(C), "cases", round(time.time() - t), "s")
print(sorted(collections.Counter((c.variant, c.status, c.ring_libs, (c.shared, c.eyes, c.outside)) for c in C).items()))
