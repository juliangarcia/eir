import numpy as np
import matplotlib.pyplot as plt
import cProfile

'''
parameters
---------------------------------
z = population
k = # p strategists
b = benefit
c = cost
norm = reputation assignment, given 2nd order social norm, pr of assigningG,  d = {d_GC,d_GD,d_BC,d_BD},list type
alpha = assignment error
eps = exe error
ki = private error
ups = pr not updating error
p= behaviour rule 1, k    one of {AllC,Pdisc,disc,AllD}  2 bit string, will gets converted to list
p_dash = behaviour rule 2,z-k
'''

#4 stochastic process
def calculate_plus(h,h_dash,k,z,G,G1):
    result = (k-h)/z*((h+h_dash)*G/(z-1)+(z-h-h_dash-1)*G1/(z-1))
    return result

def calculate_neg(h,h_dash,k,z,G,G1):
    result = h/z*((h+h_dash-1)/(z-1)*(1-G)+(z-h-h_dash)/(z-1)*(1-G1))
    return result


#G,G1 diff from above 
def calculate_dplus(h,h_dash,k,z,G,G1):
    result = (z-k-h_dash)/z*((h+h_dash)/(z-1)*G+(z-h-h_dash-1)/(z-1)*G1)
    return result

def calculate_dneg(h,h_dash,k,z,G,G1):
    result = h_dash/z*((h+h_dash-1)/(z-1)*(1-G)+(z-h-h_dash)/(z-1)*(1-G1))
    return result


#calculating  coop and assignment probabilities, ordered as C_PG,C_PB,G_PG,G_PB 
def calculate_coop(norm,alpha,eps,ki,p):
    coop = [0] * 4

    #incorp exe error
    p = (1-eps)*p

    #incorp assign error
    d = np.array(norm)
    d = (1-2*alpha)*d+alpha

    #incorp private error
    X= np.array([1-ki,ki])
    X_dash= np.array([ki,1-ki])

    C_PG = np.dot(X,p)
    C_PB = np.dot(X_dash,p)

    #kroncker product for p
    c1 = np.array([C_PG,1-C_PG])
    c2 = np.array([C_PB,1-C_PB])
    G_PG = np.dot(np.kron(X,c1),d)
    G_PB = np.dot(np.kron(X_dash,c2),d)
    coop[0] = C_PG
    coop[1] = C_PB
    coop[2] = G_PG
    coop[3] = G_PB
    return coop

#computes rep distribution, for a fixed k
def rep_distribution(z,k,coop,coop1,states):
    #total states
    dim = (k+1)*(z-k+1)
    matrix_H = np.zeros((dim,dim))
    for f in range(k+1):
        for g in range(z-k+1):
            pos = states[(f,g)]
            #try f
            if f < k:
                new_f = f+1
                new_pos = states[(new_f,g)]
                matrix_H[pos][new_pos] = calculate_plus(f,g,k,z,coop[2],coop[3])

            if f >0:
                new_f =f-1
                new_pos = states[(new_f,g)]
                matrix_H[pos][new_pos] = calculate_neg(f,g,k,z,coop[2],coop[3])

            #try g
            if g < z-k:
                new_g = g+1
                new_pos = states[(f,new_g)]
                matrix_H[pos][new_pos] = calculate_dplus(f,g,k,z,coop1[2],coop1[3])

            if g >0:
                new_g =g-1
                new_pos = states[(f,new_g)]
                matrix_H[pos][new_pos] = calculate_dneg(f,g,k,z,coop1[2],coop1[3])
            matrix_H[pos][pos] = 1 - np.sum(matrix_H[pos])

    #compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig((matrix_H.T))
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    sigma = w/np.sum(w)
    return sigma

def calculate_R(h,k,z,C_PG,C_PdG,C_PB,C_PdB):
    try:
        result = h/k *((k-1)/(z-1)*C_PG+(z-k)/(z-1)*C_PdG)+(k-h)/k*((k-1)/(z-1)*C_PB + (z-k)/(z-1)*C_PdB)
    except ZeroDivisionError:
        result = 0
    return result

def calculate_Rd(h_dash,k,z,C_PG,C_PdG,C_PB,C_PdB):
    try:
        result = h_dash/(z-k)*(k/(z-1)*C_PG + (z-k-1)/(z-1)*C_PdG)+ (z-k-h_dash)/(z-k)*(k/(z-1)*C_PB+(z-k-1)/(z-1)*C_PdB)
    except ZeroDivisionError:
        result = 0

    return result

def calculate_D(h,h_dash,k,z,C_PG,C_PB):
    try:
        result = h/k*((h-1+h_dash)/(z-1)*C_PG+(z-h-h_dash)/(z-1)*C_PB)+ (k-h)/k*((h+h_dash)/(z-1)*C_PG+(z-h-1-h_dash)/(z-1)*C_PB)
    except ZeroDivisionError:
        result = 0
    return result

def calculate_Dd(h,h_dash,k,z,C_PdG,C_PdB):
    try:
        result = h_dash/(z-k)*((h-1+h_dash)/(z-1)*C_PdG+(z-h-h_dash)/(z-1)*C_PdB)+(z-k-h_dash)/(z-k)*((h+h_dash)/(z-1)*C_PdG + (z-h-h_dash-1)/(z-1)*C_PdB)
    except ZeroDivisionError:
        result = 0
    return result

#calculate avg fitness of  p and p', given by coops
def calculate_avg_fitness(z,k,coop,coop1,sigma,b,c,states):
    total= 0; total2 = 0
    #avg fitness for p
    for f in range(0,k+1):
        for g in range(0,z-k+1):
            indx = states[(f,g)]
            #pr p strategist receives a donation
            R_p = calculate_R(f,k,z,coop[0],coop1[0],coop[1],coop1[1])
            #pr p strategist donates
            D_p =calculate_D(f,g,k,z,coop[0],coop[1])
            fitness_p = b*R_p - c*D_p
            total += sigma[indx] *fitness_p
            
            #pr p' receives & donates
            R_pd = calculate_Rd(g,k,z,coop[0],coop1[0],coop[1],coop1[1])
            D_pd =calculate_Dd(f,g,k,z,coop1[0],coop1[1])
            fitness_pd = b*R_pd - c*D_pd
            total2 += sigma[indx] *fitness_pd
    
    result = [total,total2]
    return result


#p=res,p'=mut
def calculate_fixation(z,intensity,p,p_dash,coop,coop1,b,c):
    #to store computed avg fitness diff
    memo = [None]*(z)
    result = 0

    for i in range(1,z):
        tot = 1
        for j in range(1,i+1):
            if memo[j] == None:
                states = {}

                #indx ctr
                ctr =0
                for m in range(j+1):
                    for n in range(z-j+1):
                        states[(m,n)] = ctr
                        ctr +=1
                sigma= rep_distribution(z,j,coop,coop1,states)          #rep distribution
                avg_1, avg_2 = calculate_avg_fitness(z,j,coop,coop1,sigma,b,c,states) #avg_fitness
                diff = avg_1- avg_2
                memo[j] = diff
            else:
                diff = memo[j]
            tot *= np.exp(-1*intensity*diff)
        result +=tot

    pr = 1/(1+result)
    return pr

#coop index
def coop_index(z,fixations,norm,alpha,eps,ki):
    result = 0
    for i in range(4):
        total = 0
        p = np.array(list(map(int, np.binary_repr(i,2))))   #for encoding behaviour rules

        #build transition matrix
        H = np.zeros((z+1,z+1))
        coop = calculate_coop(norm,alpha,eps,ki,p)     
        for f in range(z+1):
            if f > 0:
                H[f][f-1] = calculate_neg(f,0,z,z,coop[2],coop[3])
            if f < z:
                H[f][f+1] = calculate_plus(f,0,z,z,coop[2],coop[3])
            H[f][f] = 1 - sum(H[f])

        eigenvalues, eigenvectors = np.linalg.eig((H.T))
        idx = np.argmin(np.abs(eigenvalues - 1))
        w = np.real(eigenvectors[:, idx]).T
        reps = w/np.sum(w) 
        for j in range(z+1):
            d = calculate_D(j,0,z,z,coop[0],coop[1]) 
            total +=(d*reps[j])
        result += (total*fixations[i])
    return result


#in following order, {AllC,Pdisc,disc,AllD}
#calc stationary distribution of strategies
def main(z,norm,alpha,eps,ki,b,c,intensity):

    Tr = np.zeros((4,4))
    for f in range(4):
        for g in range(4):
            if f != g:

                p = list(map(int, np.binary_repr(g,2))) #map str to list[int]
                p = np.array(p)

                p_dash = list(map(int, np.binary_repr(f,2)))
                p_dash = np.array(p_dash)

                coop = calculate_coop(norm,alpha,eps,ki,p) #calculate C&G for p
                coop1 = calculate_coop(norm,alpha,eps,ki,p_dash) #calculate C&G for p'
                
                #f is resident(p') ,z-k qty
                #g is mutant(p) , k qty
                fixation = calculate_fixation(z,intensity,p,p_dash,coop,coop1,b,c)
                Tr[f][g] =1/3*fixation
        Tr[f][f] = 1 - np.sum(Tr[f])
    
    eigenvalues, eigenvectors = np.linalg.eig((Tr.T))
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    matrix_T = w/np.sum(w)
    #coop
    coop_indx = coop_index(z,matrix_T,norm,alpha,eps,ki)
    return coop_indx



###profiling
'''
pr = cProfile.Profile()
pr.enable()
main(40,[1,0,0,1],0.01,0.08,0.01,5,1,1)
pr.disable()
pr.print_stats(sort='time')

norm = [1,0,0,0]
a = main(30,norm,0.01,0.08,0.01,5,1,1)
print(a)
#plotting
'''
if __name__ == "__main__":

    plt.title('My results')

    x= [i for i in range(10,60,10)]
    x.insert(0,5)

    y = [] #SJ

    y2 = [] #SS

    y3 = [] #SH

    y4 = [] #IS
    
    y.append(main(5,[1,0,0,1],0.01,0.08,0.01,5,1,1))

    y2.append(main(5,[1,0,1,1],0.01,0.08,0.01,5,1,1))

    y3.append(main(5,[1,0,0,0],0.01,0.08,0.01,5,1,1))

    y4.append(main(5,[1,0,1,0],0.01,0.08,0.01,5,1,1))
    
    for pop in range(10, 60, 10):

        y.append(main(pop,[1,0,0,1],0.01,0.08,0.01,5,1,1))

        y2.append(main(pop,[1,0,1,1],0.01,0.08,0.01,5,1,1))

        y3.append(main(pop,[1,0,0,0],0.01,0.08,0.01,5,1,1))

        y4.append(main(pop,[1,0,1,0],0.01,0.08,0.01,5,1,1))

    plt.plot(x,y,'ro-',label = 'SJ')

    plt.plot(x,y2,'go-',label = 'SS')

    plt.plot(x,y3,'bo-',label = 'SH')

    plt.plot(x,y4,'co-',label = 'IS')

    plt.axvline(50,0,1,ls='--',color = 'black')

    plt.legend(loc='upper right')

    plt.xlim(0,50)

    plt.ylim(0,1)

    plt.xlabel('Population size(Z)')

    plt.ylabel('Cooperation Index')

    plt.savefig('results1')
    plt.show()
