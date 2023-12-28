# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:03:13 2023

@author: Chen Jicheng
"""

from numpy.random import poisson
from numpy.random import seed
import numpy as np
import time

lambs = [45,55,50,100]
ps = [500,380,215,180]

def change_cps(ps):
    cps = [ps[i+1]/ps[i] for i in range(len(ps)-1)]
    return cps

def monte_carlo_step(lambs,cps,samples,sum_flag):
    sequence = []
    seed()
    for i in range(len(cps)+1):
        sequence.append([])
        for j in range(int(samples)):
            sequence[i].append(poisson(lambs[i]))
    if sum_flag:
        for i in range(len(cps)+1):
            if i != 0:
                for j in range(int(samples)):
                    sequence[i][j] += sequence[i-1][j]
    for i in range(len(cps)+1):
        sequence[i].sort(reverse=True)
    return sequence

def cps_translate(cps):
    rcps = cps.copy()
    for i in range(len(cps)-1):
        rcps[i+1] = rcps[i]
    return rcps

def little_woods(sequence,cps,samples,demand_based):
    copy_cps = cps.copy()
    if demand_based:
        ps = [1] + cps_translate(cps)
        frac_above, frac_under = [],[]
        pps = []
        for i in range(len(ps)):
            frac_above.append(ps[i]*sum(sequence[i]))
            frac_under.append(sum(sequence[i]))
            pps.append(sum(frac_above)/sum(frac_under))
        for i in range(len(cps)):
            copy_cps[i] = ps[i+1] / pps[i]
    indexs = []  
    for i in range(len(cps)):
        indexs.append(int(samples*copy_cps[i]))
    save_bounds = []
    for i in range(len(cps)):
        save_bounds.append(sequence[i][indexs[i]])
    return save_bounds

def MC_current(lambs,cps,test_time,samples,sum_flag,demand_based):
    tests = np.zeros((test_time,len(cps)))
    for i in range(test_time):
        sequence = monte_carlo_step(lambs,cps,samples,sum_flag)
        tests[i] = little_woods(sequence,cps,samples,demand_based)
    return tests.mean(axis = 0).round()

def MC_integration(lambs,cps,test_time,samples):
    return MC_current(lambs,cps,test_time,samples,True,False)

def EMSR_translate(esitim):
    for i in range(len(esitim)-1):
        esitim[i+1] += esitim[i]
    return esitim

def EMSR_a(lambs,cps,test_time,samples):
    return EMSR_translate(MC_current(lambs,cps,test_time,samples,False,False))

def EMSR_b(lambs,cps,test_time,samples):
    return MC_current(lambs,cps,test_time,samples,True,True)

def NPmin(x,y):
    result = x.copy()
    for i in range(len(x)):
        if type(y) == int or type(y) == np.int32:
            result[i] = min(result[i],y)
        else:
            result[i] = min(result[i],y[i])
    return result
    
def NPmax(x,y):
    result = x.copy()
    for i in range(len(x)):
        if type(y) == int or type(y) == np.int32:
            result[i] = max(result[i],y)
        else:
            result[i] = max(result[i],y[i])
    return result

max_vs = {}
for i in range(2, len(ps) + 1):
    max_vs[i] = 0

def find_y_t(sequence,ps,x,ys,t):
    if type(x) == int or type(x) == np.int32:
        end_index = x
    else:
        end_index = round(x.mean())
    if ys[t-2] != 0:
        end_index = ys[t-2] + 1
    start_index = ys[t-2]
    for yt in range(start_index,int(end_index)):
        save_y = ys[t-2]
        ys[t-2] = yt
        value = ps[t-1]*NPmin(sequence[t-1], x - ys[t-2]).mean() + \
            Vt_x(sequence, ps, NPmax(x - sequence[t-1], ys[t-2]), ys, t-1).mean()
        if value > max_vs[t]:
            max_vs[t] = value
        else:
            #回溯
            ys[t-2] = save_y
    return max_vs[t]

def Vt_x(sequence,ps,x,ys,t):
    if t == 1:
        trys = sequence[0]
        np.random.shuffle(trys)
        return ps[0]*NPmin(trys,x).mean()
    max_v = find_y_t(sequence, ps, x, ys, t)
    vt = max_v
    #print("Now we are testing the resolution:{0},{1}".format(ys,max_v))
    return vt
  
def Dynamic_decide(lambs,ps,capacity,test_time,samples):
    cps = change_cps(ps)
    x = capacity
    tests = np.zeros((test_time,len(cps)))
    for i in range(test_time):
        ys = [0]*len(cps)
        sequence = np.array(monte_carlo_step(lambs,cps,samples,False))
        Vt_x(sequence,ps,x,ys,len(ps))
        tests[i] = ys.copy()
        global max_vs
        for i in range(2, len(ps) + 1):
            max_vs[i] = 0
    return tests.mean(axis = 0).round()

cps = change_cps(ps)
test_time = 10
samples = 1e3
t1 = time.time()
print(MC_integration(lambs, cps, test_time, samples))
print(EMSR_a(lambs,cps,test_time,samples))
print(EMSR_b(lambs,cps,test_time,samples))
t2 = time.time()
print(Dynamic_decide(lambs,ps,220,test_time,samples))
t3 = time.time()
print("CPU ocuppied time for heuristic search and dynamic algorithm:\
      \n{0},{1},respectively".format(round((t2-t1)/3,3),round(t3-t2,3)))