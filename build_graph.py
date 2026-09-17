"""Build signed neuron->neuron CSR connectivity matrix from MaleCNS connectome-weights."""
import numpy as np, pyarrow as pa, pyarrow.ipc as ipc, pyarrow.feather as pf
import scipy.sparse as sp, time

t0 = time.time()

# --- 1. neuron set from annotations (exclude non-neuronal status labels) ---
ann = pf.read_table("body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
exclude = {'Glia', 'Orphan', 'Orphan-artifact', 'Unimportant', 'Out of scope'}
mask = ~ann.statusLabel.isin(exclude)
neurons = np.unique(ann[mask].bodyId.astype(np.int64).values)
N = len(neurons)
print(f"神经元数 N = {N:,}  (总 bodies {len(ann):,}, 排除 {(~mask).sum():,} 非神经)")

# superclass per neuron for later
sc_map = ann.set_index('bodyId').superclass
sc_arr = sc_map.reindex(neurons).fillna('unknown').values

# --- 2. neurotransmitter -> sign per neuron (Dale's law, sign of PRE-synaptic neuron) ---
nt = pf.read_table("body-neurotransmitters-male-cns-v1.0.feather").to_pandas()
nt = nt.drop_duplicates('body').set_index('body')
nt_arr = nt.reindex(neurons)['consensus_nt'].fillna('unclear').values
sign_map = {'acetylcholine': 1.0, 'gaba': -1.0, 'glutamate': -1.0, 'histamine': -1.0,
            'dopamine': 0.0, 'octopamine': 0.0, 'serotonin': 0.0, 'unclear': 0.0}
signs = np.array([sign_map.get(x, 0.0) for x in nt_arr], dtype=np.float32)
print("递质分布(神经元): " + ", ".join(f"{k}:{(nt_arr==k).sum():,}"
      for k in ['acetylcholine','gaba','glutamate','histamine','dopamine','octopamine','serotonin','unclear']))

# --- 3. stream weights, filter to neuron->neuron edges ---
sorted_bodies = neurons  # already sorted & unique
i_list, j_list, w_list = [], [], []
total_edges = kept_edges = 0
with pa.memory_map("connectome-weights-male-cns-v1.0-minconf-0.5.feather", 'r') as src:
    r = ipc.open_file(src)
    for bi in range(r.num_record_batches):
        b = r.get_batch(bi)
        pre = b['body_pre'].to_numpy()   # presynaptic body
        post = b['body_post'].to_numpy() # postsynaptic body
        w = b['weight'].to_numpy()
        total_edges += len(pre)
        # map body -> index via searchsorted
        src = np.searchsorted(sorted_bodies, pre, side='left')
        dst = np.searchsorted(sorted_bodies, post, side='left')
        src_c = np.clip(src, 0, N - 1)
        dst_c = np.clip(dst, 0, N - 1)
        ok = (sorted_bodies[src_c] == pre) & (sorted_bodies[dst_c] == post)
        if ok.any():
            i_list.append(dst[ok].astype(np.int32))   # rows = postsynaptic
            j_list.append(src[ok].astype(np.int32))   # cols = presynaptic
            w_list.append(w[ok].astype(np.float32))
            kept_edges += int(ok.sum())
        if bi % 400 == 0:
            print(f"  batch {bi}/{r.num_record_batches}: 总 {total_edges/1e6:.0f}M, 保留 {kept_edges/1e6:.1f}M, {time.time()-t0:.0f}s", flush=True)

print(f"\n总边 {total_edges:,} -> 神经元内边 {kept_edges:,} ({100*kept_edges/total_edges:.1f}%)")

i = np.concatenate(i_list); j = np.concatenate(j_list); w = np.concatenate(w_list)
del i_list, j_list, w_list

# sign by presynaptic neuron (column = src)
signed_w = (w * signs[j]).astype(np.float32)
del w

print(f"构建 CSR ({N:,}x{N:,}, {kept_edges/1e6:.1f}M nnz)...")
C = sp.coo_matrix((signed_w, (i, j)), shape=(N, N)).tocsr()
C.sum_duplicates()
C.sort_indices()
print(f"CSR nnz = {C.nnz:,}, 内存 {C.data.nbytes/1e9:.2f} GB")

# --- save ---
np.savez_compressed("graph.npz",
    indptr=C.indptr, indices=C.indices, data=C.data, N=N,
    bodyId=neurons, signs=signs, nt=nt_arr, superclass=sc_arr)
print(f"已保存 graph.npz, 总耗时 {time.time()-t0:.0f}s")

# quick stats
n_exc = int((signs > 0).sum()); n_inh = int((signs < 0).sum()); n_other = N - n_exc - n_inh
print(f"\n神经元: 兴奋 {n_exc:,} / 抑制 {n_inh:,} / 其他(递质不明或单胺) {n_other:,}")
pos = (C.data > 0).sum(); neg = (C.data < 0).sum()
print(f"突触边: 兴奋 {pos:,} / 抑制 {neg:,}")
print(f"每神经元平均出边: {C.indptr[1:].mean()-C.indptr[:-1].mean():.1f}")
