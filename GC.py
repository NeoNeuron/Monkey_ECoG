# Author: Kai Chen
import numpy as np

def create_structure_array(x:np.ndarray, order:int)->np.ndarray:
    '''
    Prepare structure array for regression analysis.
    
    Args:
    x         : original time series
    order     : regression order
    
    Return:
    x_array   : structure array with shape (len(x)-order) by (order).
    
    '''
    N = len(x) - order
    x_array = np.zeros((N, order))
    for i in range(order):
        x_array[:, i] = x[-i-1-N:-i-1]
    return x_array


def auto_reg(x, order)->np.ndarray:
    '''
    Auto regression analysis of time series.
    
    Args:
    x         : original time series
    order     : regression order
    
    Return:
    res       : residual vector
    
    '''
    reg_array = create_structure_array(x, order)
    coef = np.linalg.lstsq(reg_array, x[order:], rcond=None)[0]
    res = x[order:] - reg_array @ coef
    return res

def joint_reg(x, y, order)->np.ndarray:
    '''
    Joint regression analysis of time series.
    
    Args:
    x         : original time series 1
    y         : original time series 2
    order     : regression order
    
    Return:
    res       : residual vector
    
    '''
    reg_array_x = create_structure_array(x, order)
    reg_array_y = create_structure_array(y, order)
    reg_array = np.hstack((reg_array_x, reg_array_y))
    coef = np.linalg.lstsq(reg_array, x[order:], rcond=None)[0]
    res = x[order:] - reg_array @ coef
    return res

def GC(x, y, order):
    '''
    Granger Causality from y to x
    
    Args:
    x         : original time series (dest)
    y         : original time series (source)
    order     : regression order
    
    Return:
    GC_value  : residual vector
    
    '''
    res_auto = auto_reg(x, order)
    res_joint = joint_reg(x, y, order)
    if res_auto.std() <= 1e-12:
        print('[WARNING]: too small residue error, closing to eps.')
    GC_value = 2.*np.log(res_auto.std()/res_joint.std())
    return GC_value
    
def GC_SI(p, order, length):
    from scipy.stats import chi2
    '''
    Significant level of GC value.
    
    Args
    p       : p-value
    order   : parameter of chi^2 distribution
    length  : length of data.
    
    Return:
    significant level of null hypothesis (GC 
        between two independent time seies)
    
    '''
    return chi2.ppf(1-p, order)/length

if __name__ == '__main__':
    from mutual_info_cy import mutual_info
    import matplotlib.pyplot as plt
    n = 100000
    order = 10
    a0 = np.random.rand(n)
    b0 = np.random.rand(n)
    a = a0.copy()
    b = b0.copy()
    for i in range(n-1):
        b[i+1] += 0.0*a[i]
        a[i+1] += -0.1*b[i]

    b_shuffle = b.copy()
    np.random.shuffle(b_shuffle)
    n_orders = 20
    GC_ab = np.zeros(n_orders)
    GC_ba = np.zeros(n_orders)
    GC_si = np.zeros(n_orders)
    GC_si1 = np.zeros(n_orders)
    for idx, order in enumerate(np.arange(len(GC_ab))+1):
        GC_ab[idx] = GC(b,a,order)
        GC_ba[idx] = GC(a,b,order)
        GC_si[idx] = GC_SI(1e-3,order, n)
        GC_si1[idx] = GC(a, b_shuffle, order)
    
    plt.semilogy(np.arange(len(GC_ab))+1, GC_ab, label='GC a->b')
    plt.semilogy(np.arange(len(GC_ab))+1, GC_ba, label='GC b->a')
    plt.semilogy(np.arange(len(GC_ab))+1, GC_si, label=r'GC SI ($p=10^{-3}$)')
    plt.semilogy(np.arange(len(GC_ab))+1, GC_si1, label=r'GC SI (shuffle)')
    plt.xlabel('Regression order')
    plt.ylabel('GC Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tmp/GC_test.png')

    # print(f'GC b->a : {GC(a,b,order):.3e}')
    # print(f'GC a->b : {GC(b,a,order):.3e}')
    # print(f'GC SI   : {GC_SI(1e-4,order, n):.3e}')
    # print(f'GC(a,b) : {mutual_info(a[1:], b[:-1]):.3e}')