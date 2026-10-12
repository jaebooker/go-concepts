import pickle, sys
from goconcepts.statusgen import generate_sized
C = generate_sized(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
pickle.dump(C, open("data/status_sized.pkl", "wb"))
print(len(C), "cases")
