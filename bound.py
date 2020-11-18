def bound(uc,L,H):
    if uc>H:
        return H
    elif uc<L:
        return L
    else:
        return uc