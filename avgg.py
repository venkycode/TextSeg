def avg_pi(centr, i):
    # print(type(N))

    new_img = img[:, int(max(centr - sig/2, 0)): int(min(M, centr + sig/2))]
    hh=new_img.shape[0]
    ww= new_img.shape[1]
    one_part = 360/16
    #print(contours[0])
    first=list()
    for i in range(ww):
        for j in range(hh):
            if new_img[j][i]==0:
                first.append(i)
                first.append(j)
    if len(first)==0: return i
    cnt = np.zeros(8, dtype=int)
    for i in range(ww):
        for j in range(hh):
            if i== first[0] and j== first[1]: continue
            if new_img[j][i]: continue
            angle = (math.atan2(j-first[1],i-first[0])) * (180/math.pi)
            if(angle < 0):
                angle += 180
            #print(angle)
            cnt[int(math.floor(angle/one_part)) % 8] += 1

    out = i + N*(((2*cnt[1]+2*cnt[2]+cnt[3])-(cnt[5]+2*cnt[6]+2*cnt[7])) /
                 ((cnt[1]+2*cnt[2]+2*cnt[3])+2*cnt[4]+(2*cnt[5]+2*cnt[6]+cnt[7])))
    #print(out, i)
    return out