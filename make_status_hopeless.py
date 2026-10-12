import pickle, sys
from goconcepts.statusgen import generate_hopeless
C = generate_hopeless(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
pickle.dump(C, open("data/status_hopeless.pkl", "wb"))
print(len(C), "cases")
