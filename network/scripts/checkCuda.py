import torch
import os
import numpy as np

model = torch.load(r"C:\Users\jason\Documents\documents\GitHub\ECE4078-G1\network\scripts\model\Test.pth")

# Label review
dataDir = "{}/".format(os.getcwd())
fileNameI = "{}/model1.txt".format(dataDir)
#np.savetxt(fileNameI, model)

f = open(fileNameI, "a")
f.write(str(model))
f.close()
