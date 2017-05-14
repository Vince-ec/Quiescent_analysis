import numpy as np

def Get_Sensitivity(filter_num):
    f=open('vtl/FILTER.RES.latest','r')
    data=f.readlines()
    rows=[]
    for i in range(len(data)):
        rows.append(data[i].split())

    i=0
    sens_data=[]
    while i < len(data):
        sdata=[]
        amount=int(rows[i][0])
        for u in range(amount):
            r=np.array(rows[i+u+1])
            sdata.append(r.astype(np.float))
        sens_data.append(sdata)
        i=i+amount+1

    sens_wave=[]
    sens_func=[]
    s_wave=[]
    s_func=[]
    for i in range(len(sens_data[filter_num-1])):
        s_wave.append(sens_data[filter_num-1][i][1])
        s_func.append(sens_data[filter_num-1][i][2])

    for i in range(len(s_func)):
        if .001 < s_func[i]:
            sens_func.append(s_func[i])
            sens_wave.append(s_wave[i])

    return np.array(sens_wave), np.array(sens_func) / np.max(sens_func)
