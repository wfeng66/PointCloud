from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import NumberOfPoints, Amplitude, PersistenceEntropy
import numpy as np

class tda:
    def __init__(self, homo_dim=2, fts='entropy') -> None:
        """
        homo_dim: homology dimensions, integer value
        fts: features need to return, options: entropy, numofpoints, amp
        """
        self.homology_dimensions = list(range(homo_dim + 1))
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=-1)
        self.fts = fts
        if fts == 'entropy':
            self.persistence_entropy = PersistenceEntropy()
        elif fts == 'numofpoints':
            self.NumOfPoint = NumberOfPoints()
        elif fts == 'amp':
            self.metrics = ["bottleneck", "wasserstein", "landscape", "persistence_image", "betti", "heat"]



    def forward(self, pcd):
        """
        pcd: the point cloud list to be calculated tda features. For single point cloud, add "[]", for example, tda([pcd])
        return: 2D numpy array, features extracted from inputing point cloud list
        """
        self.diag = self.persistence.fit_transform(pcd)
        if self.fts == 'entropy':
            return self.persistence_entropy.fit_transform(self.diag)
        elif self.fts == 'numofpoints':
            return self.NumOfPoint.fit_transform(self.diag)
        elif self.fts == 'amp':
            rslt = []
            for m in self.metrics:
                AMP = Amplitude(metric=m)
                amp = AMP.fit_transform(self.diag)
                rslt.append(amp)
            return np.hstack(rslt) if rslt else np.array([])


    def __call__(self, pcd):
        return self.forward(pcd)



def ransac_tda(pcd, dim=2, fts='entropy', k=10, m=500):
    """
    pcd: the point cloud list to be calculated tda features. For single point cloud, add "[]", for example, tda([pcd])
    k: the number of iterations
    m: the number of points to sample in each iteration
    return: 2D numpy array, features extracted from inputing point cloud list
    """
    tda_ = tda(homo_dim=dim, fts=fts)
    rslt = []
    for _ in range(k):
        idx = np.random.choice(len(pcd), m)
        pcd_ = [pcd[j] for j in idx]
        pcd_ = np.array(pcd_).reshape((m, 3))
        rslt.append(tda_([pcd_]))
    return np.median(rslt, axis=0)


