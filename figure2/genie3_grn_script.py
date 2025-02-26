import scanpy as sc
from flecs.utils import get_project_root
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from numpy import *
import time
from multiprocessing import Pool


def compute_feature_importances(estimator):
    if isinstance(estimator, BaseDecisionTree):
        return estimator.tree_.compute_feature_importances(normalize=False)
    else:
        importances = [e.tree_.compute_feature_importances(normalize=False)
                       for e in estimator.estimators_]
        importances = array(importances)
        return sum(importances,axis=0) / len(estimator)
                
                
def GENIE3_GRN(expr_data, grn_adj_mat, tree_method='RF', K='sqrt', ntrees=1000, nthreads=1):
    
    time_start = time.time()

    ngenes = expr_data.shape[1]
        
    print('Tree method: ' + str(tree_method))
    print('K: ' + str(K))
    print('Number of trees: ' + str(ntrees))
    print('\n')

    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = zeros((ngenes,ngenes))
    
    print('running jobs on %d threads' % nthreads)

    input_data = list()
    for i in range(ngenes):
        input_idx = list(np.where(grn_adj_mat[:, i])[0])
        if len(input_idx) == 0:
            input_idx = list(range(ngenes))

        input_data.append( [expr_data, i, input_idx, tree_method, K, ntrees])

    pool = Pool(nthreads)
    alloutput = pool.map(wr_GENIE3_single, input_data)

    for (i,vi) in alloutput:
        VIM[i,:] = vi
   
    VIM = transpose(VIM)

    VIM *= grn_adj_mat   
 
    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM


def wr_GENIE3_single(args):
    return([args[1], GENIE3_single(args[0], args[1], args[2], args[3], args[4], args[5])])


def GENIE3_single(expr_data,output_idx,input_idx,tree_method,K,ntrees):
    
    ngenes = expr_data.shape[1]
    
    # Expression of target gene
    output = expr_data[:,output_idx]
    
    # Normalize output data
    output = output / std(output)
    
    # Remove target gene from candidate regulators
    input_idx = input_idx[:]
    if output_idx in input_idx:
        input_idx.remove(output_idx)

    expr_data_input = expr_data[:,input_idx]
    
    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K,int) and K >= len(input_idx)):
        max_features = "auto"
    else:
        max_features = K
    
    if tree_method == 'RF':
        treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features=max_features)
    elif tree_method == 'ET':
        treeEstimator = ExtraTreesRegressor(n_estimators=ntrees,max_features=max_features)

    # Learn ensemble of trees
    treeEstimator.fit(expr_data_input,output)
    
    # Compute importance scores
    feature_importances = compute_feature_importances(treeEstimator)
    vi = zeros(ngenes)
    vi[input_idx] = feature_importances
       
    return vi
        
        
adata = sc.read_h5ad(os.path.join(get_project_root(), "figure2", "processed", "adata_processed_with_paths_magic.h5ad"))

print("Unsorted myeloid")
expr_data = np.array(adata[adata.obs['Batch_desc'] == "Unsorted myeloid"].X)
res = GENIE3_GRN(expr_data, 
                adata.varp['grn_adj_mat'],
                tree_method='RF',
                K='sqrt',
                ntrees=100,
                nthreads=96)
np.save(os.path.join(get_project_root(), "figure2", "processed", "genie3_grn_res_unsorted.npy"), res)

print("Cebpa KO")
expr_data = np.array(adata[adata.obs['Batch_desc'] == "Cebpa KO"].X)
res = GENIE3_GRN(expr_data, 
                adata.varp['grn_adj_mat'],
                tree_method='RF',
                K='sqrt',
                ntrees=100,
                nthreads=96)
np.save(os.path.join(get_project_root(), "figure2", "processed", "genie3_grn_res_cebpa.npy"), res)

print("Cebpe KO")
expr_data = np.array(adata[adata.obs['Batch_desc'] == "Cebpe KO"].X)
res = GENIE3_GRN(expr_data, 
                adata.varp['grn_adj_mat'],
                tree_method='RF',
                K='sqrt',
                ntrees=100,
                nthreads=96)
np.save(os.path.join(get_project_root(), "figure2", "processed", "genie3_grn_res_cebpe.npy"), res)