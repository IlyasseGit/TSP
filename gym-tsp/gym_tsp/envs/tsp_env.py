import sys
from contextlib import closing
import matplotlib.pyplot as plt
import numpy as np
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete



#===============Pour la version de 4 actions =============
#travel0 = 0
#travel1 = 1
#travel2 = 2
#travel3 = 3

#============ List d'actions, Max 1000 actions par exemple  ============
travel=[i for i in range(1000)]
#=====================================================
MAPS = {
    "4x4": [
        "C...",
        "...C",
        ".C..",
        "...."
    ],
    "8x8": [
        "C...C...",
        "........",
        "........",
        "...C....",
        "........",
        ".......C",
        ".C......",
        ".......C"
    ],
    "10x10":['..C...C...',
 '.........C',
 'C.........',
 '..........',
 '..C.......',
 '..........',
 '.C........',
 '.......C..',
 '..........',
 '..C.......']
}


def generate_random_map(size=10, p=0.1):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        nrow,ncol=res.shape
        i=0
        for row in range(nrow):
            for col in range(ncol):
                if res[row][col]=='C':
                    i+=1
                
        if i>2:
            return True
            
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['C', '.'], (size, size), p=[p, 1-p])
        
        valid = is_valid(res)
    return ["".join(x) for x in res]
    
hist_rc=[]
hist_s=[]
ind_per_nearest=[]
Dist_C=[]    
ind_C=[]    
end_episode=False
class TspEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="8x8"):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)



        isd = np.array(desc == b'C').astype('float64').ravel()
        global ind_C
        ind_C=np.nonzero(isd)
        global n_C
        n_C=ind_C[0].size
        global Dist_C
        Dist_C=np.zeros((n_C,2+n_C))
        isd=isd[np.where(isd)[0]]

        isd/=isd.sum()
        for i in range(n_C):
            Dist_C[i,0]=ind_C[0][i]//ncol
            Dist_C[i,1]=ind_C[0][i]%ncol
        for i in range(n_C):
            for j in range(n_C):
                Dist_C[i,j+2]=np.sqrt((Dist_C[i,0]-Dist_C[j,0])**2+(Dist_C[i,1]-Dist_C[j,1])**2)
        global ind_per_nearest
        ind_per_nearest=np.zeros((n_C,n_C-1))
        for i in range(n_C):
            x=np.argsort(Dist_C[i,2:])
            ind_per_nearest[i,:]=x[1:]

        nA = n_C-1
        
        nS = n_C
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        def to_C(row, col):
            indcity=0
            for (i,j) in Dist_C[:,0:2]:
                if (row, col)==(i,j):
                    return indcity
                indcity+=1
            

        def inc(row, col, a):
            numC=to_s(row, col)
            index_C=np.where(ind_C[0]==numC)[0]
            index_C=np.ndarray.tolist(index_C)[0]
            for i in range(n_C):
                if a == travel[i]:
                    m=int(ind_per_nearest[index_C,i])
                    row = Dist_C[m,0]
                    col = Dist_C[m,1]            
            return (row, col)
        j=0
        for s in range(n_C):
                
                for a in range(nA):
                    li = P[s][a]
                    [row, col]=Dist_C[s,0:2]
                    newrow, newcol = inc(row, col, a)
                    newstate = to_C(newrow, newcol)
                    done = False
                    rew = 1/Dist_C[j,2+int(ind_per_nearest[j,a])]
                    li.append((1.0, newstate, rew, done))
                j=j+1
        global P_original
        P_original=np.copy(P)
        super(TspEnv, self).__init__(nS, nA, P, isd)
        ind_C=np.nonzero(isd)
        
        
    def inf_env(self):
        return ind_per_nearest
        
    def plot_tsp(self):
        cities = np.array(hist_rc)
        n=len(hist_rc)
        plt.gca().invert_yaxis()
#        plt.gca().invert_xaxis()

        plt.plot([cities[i][1] for i in range(n)], [cities[i][0] for i in range(n)], 'xb-');
        plt.show()

    
    def step(self, a):

         def to_s(row, col):
             return row*self.ncol + col
#         ind_s=np.where(ind_C[0]==self.s)[0][0]
         ind_s=self.s
         j=0
         ind=0

         for (row,col) in Dist_C[:,0:2]:
             
             jj=0
             for i in ind_per_nearest[j,:]:
                 
                 if i==ind_s:
                     self.P[ind][jj][0]=(self.P[ind][jj][0][0],self.P[ind][jj][0][1],self.P[ind][jj][0][2],True)
                 jj+=1     
             j+=1
             ind+=1
             
         x=super(TspEnv, self).step(a)
         global hist_rc,hist_s
         hist_rc.append((Dist_C[x[0],0],Dist_C[x[0],1]))
         hist_s.append(x[0])

         if x[2]==True:
             print('Fin épisode !!! parours d agent (row,col):')
             print(hist_rc)
#             print('Fin épisode !!! parours d agent (statut):')
#             print(hist_s)

         return x
    def reset(self):
        global hist_rc
        del hist_rc[:]
        s=super(TspEnv, self).reset()
#        self.P=np.copy(P_original)
        for ss in range(self.nS):
            for a in range(self.nA):
                
                self.P[ss][a][0]=(self.P[ss][a][0][0],self.P[ss][a][0][1],self.P[ss][a][0][2],False)
        hist_s.append(s)

        hist_rc.append((Dist_C[s,0],Dist_C[s,1]))
        return s
        
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        global hist_rc
        n_hist=len(hist_rc)
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        
        for i in range(n_hist):
            desc[int(hist_rc[i][0])][int(hist_rc[i][1])] = utils.colorize(desc[int(hist_rc[i][0])][int(hist_rc[i][1])], "red", highlight=True)
        
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["travel"+str(i) for i in range(n_C)][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        