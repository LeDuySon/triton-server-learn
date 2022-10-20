import numpy as np
import faiss
import time

d = 100                    # dimension
nb = 1000000                      # database size
nq = 100                       # nb of queries
np.random.seed(1234)             # make reproducible

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

res = faiss.StandardGpuResources()  # use a single GPU
# index_flat = faiss.IndexFlatL2(d)

index_flat = faiss.read_index("index.bin") 

gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

gpu_index_flat.add(xb)         # add vectors to the index
print(gpu_index_flat.ntotal)

k = 4

results = gpu_index_flat.search(xq[:5, :], k)

print("Results: ", results)

# faiss.write_index(faiss.index_gpu_to_cpu(gpu_index_flat), "index.bin")


