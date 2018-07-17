import numpy as np
import matplotlib.pyplot as plt


'''
parameters
---------------------------------
z = population
k = # p strategists
b = benefit
c = cost
norm now incorporated into strategies
alpha = assignment error
eps = exe error
ki = private error
ups = pr not updating error
p= behaviour rule 1, k    one of {AllC,Pdisc,disc,AllD}  2 bit string, will gets converted to list
p_dash = behaviour rule 2,z-k
'''

#4 stochastic process
import numpy as np

def calculate_plus(h,h_dash,k,z,G1,G2,G3,G4):
    result = (k-h)/z*((h/(z-1)*G1+h_dash/(z-1)*G2+(k-h-1)/(z-1)*G3 + (z-k-h_dash)/(z-1)*G4))
    return result

def calculate_neg(h,h_dash,k,z,G1,G2,G3,G4):
    result = h/z*((h-1)/(z-1)*(1-G1)+(h_dash)/(z-1)*(1-G2) + (k-h)/(z-1)*(1-G3) + (z-k-h_dash)/(z-1)*(1-G4))
    return result


#G,G1 diff from above 
def calculate_dplus(h,h_dash,k,z,G1,G2,G3,G4):
    result = (z-k-h_dash)/z*((h/(z-1)*G1+h_dash/(z-1)*G2+(k-h)/(z-1)*G3 + (z-k-h_dash-1)/(z-1)*G4))
    return result

def calculate_dneg(h,h_dash,k,z,G1,G2,G3,G4):
    result = (h_dash)/z*((h)/(z-1)*(1-G1)+(h_dash-1)/(z-1)*(1-G2) + (k-h)/(z-1)*(1-G3) + (z-k-h_dash)/(z-1)*(1-G4))
    return result


#calculating  coop and assignment probabilities, ordered as C_PG,C_PB,G_PG,G_PB
def calculate_coop(norm,alpha,eps,ki,p):
    coop = [0] * 4

    #incorp exe error
    p = (1-eps)*p

    #incorp assign error
    d1 = np.array(norm)
    d1 = (1-2*alpha)*d1+alpha

    #incorp private error
    X= np.array([1-ki,ki])
    X_dash= np.array([ki,1-ki])

    C_PG = np.dot(X,p)
    C_PB = np.dot(X_dash,p)

    #kroncker product for p
    c1 = np.array([C_PG,1-C_PG])
    c2 = np.array([C_PB,1-C_PB])

    G_PG = np.dot(np.kron(X,c1),d1)
    
    G_PB =np.dot(np.kron(X_dash,c2),d1)


    coop[0] = C_PG
    coop[1] = C_PB
    coop[2] = G_PG
    coop[3] = G_PB
    return coop

# rep distribution for pairwise rating, for a fixed k
def rep_distribution(z,k,coop1,coop2,coop3,coop4,states):
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
                matrix_H[pos][new_pos] = calculate_plus(f,g,k,z,coop1[2],coop2[2],coop1[3],coop2[3])

            if f >0:
                new_f =f-1
                new_pos = states[(new_f,g)]
                matrix_H[pos][new_pos] = calculate_neg(f,g,k,z,coop1[2],coop2[2],coop1[3],coop2[3])

            #try g
            if g < z-k:
                new_g = g+1
                new_pos = states[(f,new_g)]
                matrix_H[pos][new_pos] = calculate_dplus(f,g,k,z,coop3[2],coop4[2],coop3[3],coop4[3])

            if g >0:
                new_g =g-1
                new_pos = states[(f,new_g)]
                matrix_H[pos][new_pos] = calculate_dneg(f,g,k,z,coop3[2],coop4[2],coop3[3],coop4[3])
            matrix_H[pos][pos] = 1 - np.sum(matrix_H[pos])

    #compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig((matrix_H.T))
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    sigma = w/np.sum(w)
    return sigma
####################################
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

#calculate avg fitness of  p and p', given by coops(Cs dont depend on norm)
def calculate_avg_fitness(z,k,coop1,coop3,sigma,b,c,states):
    total= 0; total2 = 0
    #avg fitness for p
    for f in range(0,k+1):
        for g in range(0,z-k+1):
            indx = states[(f,g)]
            #pr p strategist receives a donation
            R_p = calculate_R(f,k,z,coop1[0],coop3[0],coop1[1],coop3[1])
            #pr p strategist donates
            D_p =calculate_D(f,g,k,z,coop1[0],coop1[1])
            fitness_p = b*R_p - c*D_p
            total += sigma[indx] *fitness_p
            
            #pr p' receives & donates
            R_pd = calculate_Rd(g,k,z,coop1[0],coop3[0],coop1[1],coop3[1])
            D_pd =calculate_Dd(f,g,k,z,coop3[0],coop3[1])
            fitness_pd = b*R_pd - c*D_pd
            total2 += sigma[indx] *fitness_pd
    
    result = [total,total2]
    return result


#p=res,p'=mut
def calculate_fixation(z,intensity,p,p_dash,coop1,coop2,coop3,coop4,b,c):
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
                sigma= rep_distribution(z,j,coop1,coop2,coop3,coop4,states)          #rep distribution
                avg_1, avg_2 = calculate_avg_fitness(z,j,coop1,coop3,sigma,b,c,states) #avg_fitness
                diff = avg_1- avg_2
                memo[j] = diff
            else:
                diff = memo[j]
            tot *= np.exp(-1*intensity*diff)
        result +=tot

    pr = 1/(1+result)
    return pr

#coop index
def coop_index(z,fixations,alpha,eps,ki):
    result = 0
    for i in range(64):
        total = 0
        p = np.array(list(map(int, np.binary_repr(i//16,2))))   #for encoding behaviour rules
        norm = list(map(int, np.binary_repr(i%16,4)))
        

        #build transition matrix
        H = np.zeros((z+1,z+1))
        coop = calculate_coop(norm,alpha,eps,ki,p)     
        for f in range(z+1):
            if f > 0:
                H[f][f-1] = calculate_neg(f,0,z,z,coop[2],0,coop[3],0)
            if f < z:
                H[f][f+1] = calculate_plus(f,0,z,z,coop[2],0,coop[3],0 )
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


#in following order, {AllC1...16,Pdisc1...16,disc1...16,AllD1....16}
#calc stationary distribution of strategies
def main(z,alpha,eps,ki,b,c,intensity):

    Tr = np.zeros((64,64))
    for f in range(64):
        for g in range(64):
            if f != g:

                p = list(map(int, np.binary_repr(g//16,2))) #map str to list[int]
                p = np.array(p)
                norm1 = list(map(int, np.binary_repr(g%16,4)))
                

                p_dash = list(map(int, np.binary_repr(f//16,2)))
                p_dash = np.array(p_dash)
                norm2 = list(map(int, np.binary_repr(f%16,4)))

                coop1 = calculate_coop(norm1,alpha,eps,ki,p) #calculate C&G for p
                coop2 = calculate_coop(norm2,alpha,eps,ki,p) 

                coop3 = calculate_coop(norm1,alpha,eps,ki,p_dash)#calculate C&G for p'
                coop4 = calculate_coop(norm2,alpha,eps,ki,p_dash)
                
                #f is resident(p') ,z-k qty
                #g is mutant(p) , k qty
                fixation = calculate_fixation(z,intensity,p,p_dash,coop1,coop2,coop3,coop4,b,c)
                Tr[f][g] =1/63*fixation
        Tr[f][f] = 1 - np.sum(Tr[f])

    #dump into csv
    np.savetxt("trans.csv", Tr, delimiter=",")
    
    eigenvalues, eigenvectors = np.linalg.eig((Tr.T))
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    matrix_T = w/np.sum(w)
    #fixation pr
    x = [i for i in range(0,64)]

    #dump fixation into csv
    np.savetxt("fix.csv", matrix_T, delimiter=",")
    
    plt.bar(x,matrix_T)
    plt.title('Fixation probabilities of all strats')
    plt.xlabel("Strategies")
    plt.ylabel("Fixation Probabilities")

    plt.savefig('fixation.png')

    #color map matrix
    plt.matshow(Tr, cmap=plt.cm.viridis_r)
    plt.title("transition matrix for 64+PW")
    plt.colorbar()

    plt.savefig('transition.png')


    #coop
    coop_indx = coop_index(z,matrix_T,alpha,eps,ki)
    return coop_indx



start_time = time.time()
print(main(50,0.01,0.08,0.01,5,1,1))


