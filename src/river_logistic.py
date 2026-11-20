"""Small fixed L2 logistic calibrator; standard statistics, not a new method."""
from dataclasses import dataclass
import numpy as np

def sigmoid(z):
    z=np.asarray(z,float);out=np.empty_like(z)
    positive=z>=0;out[positive]=1/(1+np.exp(-z[positive]))
    ez=np.exp(z[~positive]);out[~positive]=ez/(1+ez)
    return out

@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    iterations: int
    gradient_max: float

    def predict(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('feature mismatch')
        return sigmoid(self.coefficients[0]+((x-self.mean)/self.scale)@self.coefficients[1:])

def fit_logistic(x,y,regularization=1.,max_iterations=80,tolerance=1e-7):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or x.shape[0]!=len(y) or not x.size or not np.isfinite(x).all():raise ValueError('invalid features')
    if y.shape!=(len(x),) or not np.isin(y,[0,1]).all() or len(set(y))!=2:raise ValueError('both binary classes required')
    if not np.isfinite(regularization) or regularization<=0:raise ValueError('positive fixed L2 required')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-8,1.,scale)
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale]);w=np.zeros(design.shape[1])
    penalty=np.full(len(w),regularization);penalty[0]=0
    def objective(v):
        z=design@v
        return float(np.sum(np.logaddexp(0,z)-y*z)+.5*np.dot(penalty*v,v))
    for iteration in range(max_iterations+1):
        p=sigmoid(design@w);g=design.T@(p-y)+penalty*w;norm=float(np.max(np.abs(g)))
        if norm<=tolerance:
            for a in [mean,scale,w]:a.setflags(write=False)
            return LogisticModel(mean,scale,w,iteration,norm)
        if iteration==max_iterations:break
        h=design.T@((p*(1-p))[:,None]*design)+np.diag(penalty)
        delta=np.linalg.solve(h+1e-10*np.eye(len(w)),g);base=objective(w)
        step=1.
        for _ in range(30):
            proposed=w-step*delta
            if objective(proposed)<=base-1e-4*step*float(g@delta):break
            step*=.5
        else:raise RuntimeError('line search failed')
        w=proposed
    raise RuntimeError('fixed solver did not converge; retain failure rather than tune on validation')
