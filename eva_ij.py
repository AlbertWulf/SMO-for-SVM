from select_j import select_j
from K_linear import K_linear
from cal_E import cal_E 
from bound import bound
def eva_ij(kk,j,bd,a,b,C,label):
    i = kk
    ai = a[i,0]
    aj = a[j,0]
    yi = label[i,0]
    yj = label[j,0]
    Kii = K_linear(bd[i,0:-1],bd[i,0:-1])
    Kij = K_linear(bd[i,0:-1],bd[j,0:-1])
    Kjj = K_linear(bd[j,0:-1],bd[j,0:-1])
    H = 0
    L = 0
    if label[i]!=label[j]:
        L=max(0,aj-ai)
        H=min(C,C+aj-ai)
    else:
        L=max(0,ai+aj-C)
        H=min(C,ai+aj)
    Ei = cal_E(bd,a,i,b)
    Ej = cal_E(bd,a,j,b)
    #eta = K_linear(bd[i,0:-1],bd[i,0:-1])+K_linear(bd[j,0:-1],bd[j,0:-1])-2*K_linear(bd[i,0:-1],bd[j,0:-1])
    eta = Kii+Kjj-2*Kij
    if eta <= 0:
       return 0
       #print('WARNING  eta <= 0')

    aj_new_uc = (aj+label[j]*(Ei-Ej)/eta)[0,0]
    aj_new = bound(aj_new_uc,L,H)
    ai_new = ai+yi*yj*(aj-aj_new)
    b1_new = -Ei-yi*Kii*(ai_new-ai)-yj*Kij*(aj_new-aj)+b
    b2_new = -Ej-yi*Kij*(ai_new-ai)-yj*Kjj*(aj_new-aj)+b
    if (ai_new>0 and  ai_new<C):
        b = b1_new
    elif (aj_new>0 and aj_new< C):
        b = b2_new
    else:
        b = (b1_new + b2_new)/2
    #b_new = (b1_new+b2_new)/2
    #b = b_new
    a[i] = ai_new
    a[j] = aj_new

    return b,ai_new,aj_new
