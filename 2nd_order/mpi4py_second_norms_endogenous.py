from mpi4py import MPI
import numpy as np
import argparse


#MPI initialization
comm_mpi = MPI.COMM_WORLD
rank_mpi = comm_mpi.Get_rank()
size_mpi = comm_mpi.Get_size()
name_mpi = MPI.Get_processor_name()
print("I the process " + str(rank_mpi) + " of " + str(size_mpi) + " from " + name_mpi)


'''

parameters

---------------------------------

z = population size

k = number of 'p' players

b = benefit

c = cost

norm = type(list),reputation assignment, pr of assigningG, d = [d_GC,d_GD,d_BC,d_BD]

alpha = observer assignment error

eps = execution error

ki = private assessment error

p = action(behaviour rule) for the k players, one of {AllD,Pdisc,disc,AllC}, type(list)

p_dash = action (behaviour rule) for the z-k players

'''


# below is the 4 stochastic process(tracking movement of number of "Good" p players and "Good" p' players)
# all players either "Good" or "Bad" reputation, assigned by an observer

# parameters:

# h = number of "Good" p players

# h_dash = number of "Good" p' players

# G: refer to coop[] in calculate_coop() below

# flag = assignment rules, can be '3p' or 'pw'

# in '3p', the observer who assigns the reputation is randomly picked from the population, excluding the player and its opponent

# in 'pw', aka. "Pairwise", the opponent of the player is also the observer who assigns the player his new reputation.


# probability of having 1 more "Good" p player
def calculate_plus(h, h_dash, k, z, G1, G2, G3, G4, flag):
    if flag == '3p':

        result = (k-h)/z*((h/(z-1)*((k-2)/(z-2)*G1+(z-k)/(z-2)*G2))+(h_dash/(z-1)*((k-1)/(z-2)*G1+(z-k-1)/(z-2)*G2)) +

                          ((k-h-1)/(z-1)*((k-2)/(z-2)*G3+(z-k)/(z-2)*G4))+((z-k-h_dash)/(z-1)*((k-1)/(z-2)*G3+(z-k-1)/(z-2)*G4)))

    elif flag == 'pw':
        result = (k-h)/z*((h/(z-1)*G1+h_dash/(z-1)*G2 +
                           (k-h-1)/(z-1)*G3 + (z-k-h_dash)/(z-1)*G4))
    return result


# probability of having 1 less "Good"(1 more "Bad") p player
def calculate_neg(h, h_dash, k, z, G1, G2, G3, G4, flag):
    if flag == '3p':

        result = h/z*(((h-1)/(z-1)*((k-2)/(z-2)*(1-G1)+(z-k)/(z-2)*(1-G2)))+(h_dash/(z-1)*((k-1)/(z-2)*(1-G1)+(z-k-1)/(z-2)*(1-G2))) +

                      ((k-h)/(z-1)*((k-2)/(z-2)*(1-G3)+(z-k)/(z-2)*(1-G4)))+((z-k-h_dash)/(z-1)*((k-1)/(z-2)*(1-G3)+(z-k-1)/(z-2)*(1-G4))))

    elif flag == 'pw':
        result = h/z*((h-1)/(z-1)*(1-G1)+(h_dash)/(z-1)*(1-G2) +
                      (k-h)/(z-1)*(1-G3) + (z-k-h_dash)/(z-1)*(1-G4))
    return result


# probability of having 1 more "Good" p' player
def calculate_dplus(h, h_dash, k, z, G1, G2, G3, G4, flag):
    if flag == '3p':

        result = (z-k-h_dash)/z*((h/(z-1)*((k-1)/(z-2)*G1+(z-k-1)/(z-2)*G2))+(h_dash/(z-1)*(k/(z-2)*G1+(z-k-2)/(z-2)*G2)) +

                                 ((k-h)/(z-1)*((k-1)/(z-2)*G3+(z-k-1)/(z-2)*G4))+((z-k-h_dash-1)/(z-1)*((k)/(z-2)*G3+(z-k-2)/(z-2)*G4)))

    elif flag == 'pw':
        result = (z-k-h_dash)/z*((h/(z-1)*G1+h_dash/(z-1) *
                                  G2+(k-h)/(z-1)*G3 + (z-k-h_dash-1)/(z-1)*G4))

    return result


# probability of having 1 more "Bad" p' player
def calculate_dneg(h, h_dash, k, z, G1, G2, G3, G4, flag):
    if flag == '3p':

        result = h_dash/z*(((h)/(z-1)*((k-1)/(z-2)*(1-G1)+(z-k-1)/(z-2)*(1-G2)))+((h_dash-1)/(z-1)*((k)/(z-2)*(1-G1)+(z-k-2)/(z-2)*(1-G2))) +

                           ((k-h)/(z-1)*((k-1)/(z-2)*(1-G3)+(z-k-1)/(z-2)*(1-G4)))+((z-k-h_dash)/(z-1)*(k/(z-2)*(1-G3)+(z-k-2)/(z-2)*(1-G4))))
    elif flag == 'pw':
        result = (h_dash)/z*((h)/(z-1)*(1-G1)+(h_dash-1)/(z-1) *
                             (1-G2) + (k-h)/(z-1)*(1-G3) + (z-k-h_dash)/(z-1)*(1-G4))
    return result


# calculating cooperation and assignment probabilities, ordered as C_PG,C_PB,G_PG,G_PB, refered to as 'G's everywhere else

# parameters:

# norm = type(list) of size 4

# alpha = type(float) assignment error

# eps = type(flaot) execution error

# ki = type(float) private error

# p = type(list) of size 2, a particular action

# a player's strategy is made up of (action) + (norm)

def calculate_coop(norm, alpha, eps, ki, p):

    coop = [0] * 4

    # incorp execution error

    p = (1-eps)*p

    # incorp assignment error
    d1 = np.array(norm)

    d1 = (1-2*alpha)*d1+alpha

    # incorp private error

    X = np.array([1-ki, ki])

    X_dash = np.array([ki, 1-ki])

    # probability a player of certain strategy cooperating with "Good" opponent
    C_PG = np.dot(X, p)

    # probability a player of certain strategy cooperating with "Bad" opponent
    C_PB = np.dot(X_dash, p)

    # kroncker product for p

    c1 = np.array([C_PG, 1-C_PG])

    c2 = np.array([C_PB, 1-C_PB])

    # probability player with action p and with "Good" opponent is assigned 'Good' reputation by the observer
    G_PG = np.dot(np.kron(X, c1), d1)

    # probability player with action p and with "Bad" opponent is assigned 'Good' reputation by the observer
    G_PB = np.dot(np.kron(X_dash, c2), d1)

    coop[0] = C_PG  # probility player with action p cooperates with Good opponent

    coop[1] = C_PB  # probility player with action p cooperates with Bad opponent

    coop[2] = G_PG  # probility player with action p assigned "Good" by an observer

    coop[3] = G_PB  # probility player with action p assigned "Good" by an observer

    return coop


# computes stationary distribution of reputations, for a fixed k(number of p individuals)

# paremeters:

# z = total population size

# k = number of strategy 'p' individuals

# see the coop[] above for definitions for coop1-4

# all players either "Good" or "Bad"

# states = dict to store (f,g), f = number of "Good" p players, g = number of "Good" p' players

def rep_distribution(z, k, coop1, coop2, coop3, coop4, states, flag):

    # total states(total possible f & g) for a size z population

    dim = (k+1)*(z-k+1)

    matrix_H = np.zeros((dim, dim))

    for f in range(k+1):

        for g in range(z-k+1):

            pos = states[(f, g)]

            # try f

            if f < k:

                new_f = f+1

                new_pos = states[(new_f, g)]

                # probability of 1 more "Good" p player

                matrix_H[pos][new_pos] = calculate_plus(
                    f, g, k, z, coop1[2], coop2[2], coop1[3], coop2[3], flag)

            if f > 0:

                new_f = f-1

                new_pos = states[(new_f, g)]
                # probability of 1 more "Bad" p player

                matrix_H[pos][new_pos] = calculate_neg(
                    f, g, k, z, coop1[2], coop2[2], coop1[3], coop2[3], flag)

            # try g

            if g < z-k:

                new_g = g+1

                new_pos = states[(f, new_g)]
                # probability of 1 more "Good" p' player

                matrix_H[pos][new_pos] = calculate_dplus(
                    f, g, k, z, coop3[2], coop4[2], coop3[3], coop4[3], flag)

            if g > 0:

                new_g = g-1

                new_pos = states[(f, new_g)]
                # probability of 1 more "Bad" p' player

                matrix_H[pos][new_pos] = calculate_dneg(
                    f, g, k, z, coop3[2], coop4[2], coop3[3], coop4[3], flag)

            # probability of remaining in current state
            matrix_H[pos][pos] = 1 - np.sum(matrix_H[pos])

    # compute eigenvector
    eigenvalues, eigenvectors = np.linalg.eig((matrix_H.T))

    idx = np.argmin(np.abs(eigenvalues - 1))

    w = np.real(eigenvectors[:, idx]).T

    sigma = w/np.sum(w)

    return sigma


# parameters:
# h = number of "Good" p players
# k = number of p players

# h_dash = number of "Good" p' players

# C_PG = #probility player with action p cooperates with Good opponent
# C_PdG = #probility player with action p' cooperates with Good opponent
# C_PB = #probility player with action p cooperates with Bad opponent
# C_PdB = #probility player with action p' cooperates with Bad opponent

# calculate_R is the probability a (p player) receives a donation
def calculate_R(h, k, z, C_PG, C_PdG, C_PB, C_PdB):

    try:

        result = h/k * ((k-1)/(z-1)*C_PG+(z-k)/(z-1)*C_PdG) + \
            (k-h)/k*((k-1)/(z-1)*C_PB + (z-k)/(z-1)*C_PdB)

    except ZeroDivisionError:

        result = 0

    return result


# calculate_Rd is the probability a(p' player) receives a donation

def calculate_Rd(h_dash, k, z, C_PG, C_PdG, C_PB, C_PdB):

    try:

        result = h_dash/(z-k)*(k/(z-1)*C_PG + (z-k-1)/(z-1)*C_PdG) + \
            (z-k-h_dash)/(z-k)*(k/(z-1)*C_PB+(z-k-1)/(z-1)*C_PdB)

    except ZeroDivisionError:

        result = 0

    return result


# calculate_D is the probability a (p player) donates

def calculate_D(h, h_dash, k, z, C_PG, C_PB):

    try:

        result = h/k*((h-1+h_dash)/(z-1)*C_PG+(z-h-h_dash)/(z-1)*C_PB) + \
            (k-h)/k*((h+h_dash)/(z-1)*C_PG+(z-h-1-h_dash)/(z-1)*C_PB)

    except ZeroDivisionError:

        result = 0

    return result


# calculate_D is the probability a (p' player) donates

def calculate_Dd(h, h_dash, k, z, C_PdG, C_PdB):

    try:

        result = h_dash/(z-k)*((h-1+h_dash)/(z-1)*C_PdG+(z-h-h_dash)/(z-1)*C_PdB)+(
            z-k-h_dash)/(z-k)*((h+h_dash)/(z-1)*C_PdG + (z-h-h_dash-1)/(z-1)*C_PdB)

    except ZeroDivisionError:

        result = 0

    return result


# parameters:
# z = total population size

# k = number of strategy 'p' individuals

# see the coop[] above for definitions for coop

# sigma = eigenvector returned by function "rep_distribution"

# b = Benefit from being donated by opponent

# c = Cost of donating to opponent

# states = dict to store (f,g), f = number of "Good" p players, g = number of "Good" p' players


def calculate_avg_fitness(z, k, coop, coop1, sigma, b, c, states):

    total = 0
    total2 = 0

    # avg fitness for p

    for f in range(0, k+1):

        for g in range(0, z-k+1):

            indx = states[(f, g)]

            # probability p strategist receives a donation

            R_p = calculate_R(f, k, z, coop[0], coop1[0], coop[1], coop1[1])

            # probability p strategist donates

            D_p = calculate_D(f, g, k, z, coop[0], coop[1])

            # calculaes fitness by (Benefit - Cost)

            fitness_p = b*R_p - c*D_p

            # overall fitness = stationary distribution of reputation * fitness in each state in the stationary distribution

            total += sigma[indx] * fitness_p

            # probability p' receives & donates, similar to above

            R_pd = calculate_Rd(g, k, z, coop[0], coop1[0], coop[1], coop1[1])

            D_pd = calculate_Dd(f, g, k, z, coop1[0], coop1[1])

            fitness_pd = b*R_pd - c*D_pd

            total2 += sigma[indx] * fitness_pd

    result = [total, total2]
    # return fitness for p and p'

    return result


# calculates fixation probability of 1 p player(action p, norm p2) taking over z-1 p'(action p', norm p3) players

# known as fixation probability in the field


# parameters:

# z = total population size

# intensity = intensity of selection presented in the fixation probability formula

# p = action of p player

# p' = action of p' player

# see the coop[] above for definitions for coop

# b = Benefit from being donated by opponent

# c = Cost of donating to opponent

# flag = assignment rules, can be '3p' or 'pw'
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def calculate_fixation(z, intensity, p, p_dash, coop1, coop2, coop3, coop4, b, c, flag):

    # to store computed avg fitness difference

    memo = [None]*(z)
    listed = np.full((z,3),999999)

    result = 0

    #AEG: A first cycle to define the order of solution
    # i=k, the number of p individuals
    for i in range(1, z):

        for j in range(1, i+1):

            if listed[j,0] == 999999:
                dim = (j+1)*(z-j+1)

                listed[j]= [j,z,dim]

    #AEG: Sorting by dimension in descending order
    listed=listed[np.argsort(-listed[:,2])]


    #AEG: Distributing the load among the existing ranks_mpi
    fair = (z-1)//size_mpi
    leftovers= (z-1)%size_mpi
    #if (rank_mpi==0):
    #    print("z=" + str(z) + ",matrices=" + str(z-1) + ",Stride=" + str(fair) + ",leftovers=" + str(leftovers))
     
    localListed = np.full((fair+1,3),999999)

    ksolve=0;
    for k in range(0,fair):
        localListed[k]=listed[rank_mpi+k*size_mpi+1]
        ksolve=ksolve+1

    lj=size_mpi-rank_mpi-1
    if (lj<leftovers):
        localListed[ksolve]=listed[(z-1)-1-lj+1]
        ksolve=ksolve+1
    


    #AEG: A second cycle to solve for the eigenvalues
    # i=k, the number of p individuals
    for r in range(0, ksolve):
        jHere = localListed[r,0]
        zHere = localListed[r,1]
        states = {}

        # indx ctr

        ctr = 0

        for m in range(jHere+1):

            for n in range(zHere-jHere+1):

                states[(m, n)] = ctr

                ctr += 1

        # reputation distribution
        sigma = rep_distribution(
            zHere, jHere, coop1, coop2, coop3, coop4, states, flag)

        avg_1, avg_2 = calculate_avg_fitness(
            zHere, jHere, coop1, coop3, sigma, b, c, states)  # avg_fitness

        memo[jHere] = avg_1 - avg_2

    #AEG: Sending the info back to rank_mpi==0
    comm_mpi.barrier()
    auxSend=np.full((1,1),999999.0)
    for k in range(0,ksolve):
        jHere=localListed[k,0]
        auxSend[0]=memo[jHere]
        #print("Sending auxSend=" + str(auxSend[0]) + "=" + str(memo[jHere]) + " from rank_mpi=" + str(rank_mpi))
        comm_mpi.send(auxSend,dest=0,tag=jHere)

    #AEG: Now receiving everything
    auxRec=np.full((1,1),999999.0)
    if (rank_mpi==0):
        for j in range(1,z):
            auxRec=comm_mpi.recv(source=MPI.ANY_SOURCE,tag=j)
            memo[j]=auxRec[0]
            #print("Received auxRec=" + str(auxRec[0]) + "=" + str(memo[j]) + " from rank_mpi=" + str(rank_mpi))

    #AEG: For avoiding errors, broadcasting memo back to all ranks
    comm_mpi.barrier()
    memo = comm_mpi.bcast(memo,root=0)

    #AEG: Third cycle for calculating a exponentials
    # i = k, the number of p individuals
    for i in range(1, z):

        tot = 1

        for j in range(1, i+1):

            diff = memo[j]

            tot *= np.exp(-1*intensity*diff)

        result += tot

    pr = 1/(1+result)

    # returns the fixation probability
    return pr


# parameters:

# z = total population size

# long term fixation probabilities

# alpha = observer assignment error

# eps = execution error

# ki = private assessment error

# flag = assignment rules, can be '3p' or 'pw'

# calculates cooperation index(average amount of donations in each state where there exist only a single type of strategy(action+norm) for the entire population z)

def coop_index(z, fixations, alpha, eps, ki, flag):

    result = 0

    # for each possible strategy

    for i in range(64):

        total = 0

        # find action of i

        p = np.array(list(map(int, np.binary_repr(i//16, 2))))

        # find norm of i

        norm = list(map(int, np.binary_repr(i % 16, 4)))

        # build transition matrix for the single variant population

        # in this case, the number of "Good", the H matrix, individuals can only increase/decrease

        H = np.zeros((z+1, z+1))

        coop = calculate_coop(norm, alpha, eps, ki, p)

        for f in range(z+1):

            if f > 0:

                H[f][f-1] = calculate_neg(f, 0, z, z,
                                          coop[2], 0, coop[3], 0, flag)

            if f < z:

                H[f][f+1] = calculate_plus(f, 0, z,
                                           z, coop[2], 0, coop[3], 0, flag)

            H[f][f] = 1 - sum(H[f])

        # compute stationary distribution of reputation

        eigenvalues, eigenvectors = np.linalg.eig((H.T))

        idx = np.argmin(np.abs(eigenvalues - 1))

        w = np.real(eigenvectors[:, idx]).T

        reps = w/np.sum(w)

        # calculate average donations by weighted average of the fraction of donations that take place in each of the monomorphic(only a single strategy exists) configurations of the population

        for j in range(z+1):

            d = calculate_D(j, 0, z, z, coop[0], coop[1])

            total += (d*reps[j])

        result += (total*fixations[i])

    # returns the average donations
    return result


# the 64 strategies are in following order, {AllD,Pdisc,disc,AllC}

# each strategy has an action {0-3} and a norm {0-15}, 4 * 16 = 64 strategies in total

# 4 actions available:

# All D= 0

#Pdisc = 1

#disc = 2

# All C = 3

# eg. strategy 60 is All C, as 60//16 = 3 which is [1,1], its norm is calculated as 60%16 = 12
# norm = 12 means [1,1,0,0] in binary format


# paremeter:

#z = population

# alpha = observer assignment error

# eps = execution error

# ki = private assessment error

# b = Benefit from being donated by opponent

# c = Cost of donating to opponent

# intensity = intensity of selection

# flag = assignment rules, can be '3p' or 'pw'

def main(z, alpha, eps, ki, b, c, intensity, flag):


    Tr = np.zeros((64, 64))
    # f is resident(p'),N-1 qty

    # g is mutant(p), 1 qty

    # entry (f,g) in Tr means the fixation probability of 1 g(mutant) strategy player taking over z-1 f(resident) strategy player

    for f in range(64):
        print("Starting f=" + str(f))
        for g in range(64):
            if f != g:

                # action of p

                # map str to list[int]
                p = list(map(int, np.binary_repr(g//16, 2)))

                p = np.array(p)

                # norm of p

                norm1 = list(map(int, np.binary_repr(g % 16, 4)))

                # action of p'

                p_dash = list(map(int, np.binary_repr(f//16, 2)))

                p_dash = np.array(p_dash)

                # norm of p'

                norm2 = list(map(int, np.binary_repr(f % 16, 4)))

                # calculate C&G for p

                # refer to coop[] above for definitions for coop

                # C: probability a 'p' individual cooperates with opponent

                # G: probability a 'p' gets assigned a 'Good' rep

                # coop1: calculates C&G for p under norm1(norm of p)
                coop1 = calculate_coop(norm1, alpha, eps, ki, p)
                # coop2: calculates C&G for p under norm2(norm of p')
                coop2 = calculate_coop(norm2, alpha, eps, ki, p)

                # calculate C&G for p'

                # coop3: calculates C&G for p' under norm1(norm of p)
                coop3 = calculate_coop(norm1, alpha, eps, ki, p_dash)
                # coop4: calculates C&G for p' under norm2(norm of p')
                coop4 = calculate_coop(norm2, alpha, eps, ki, p_dash)

                # fixation means calculating the probability (N-1) f individuals gets taken over by 1 g individual

                fixation = calculate_fixation(
                    z, intensity, p, p_dash, coop1, coop2, coop3, coop4, b, c, flag)

                # fixation = probability a mutant arising * mutant fixating

                # 1/63 is the probability a mutant with strategy g arising in a single strategy population with strategy f

                Tr[f][g] = 1/63*fixation

        # diagonals

        Tr[f][f] = 1 - np.sum(Tr[f])

    # calculate eigenvector

    eigenvalues, eigenvectors = np.linalg.eig((Tr.T))

    idx = np.argmin(np.abs(eigenvalues - 1))

    w = np.real(eigenvectors[:, idx]).T

    matrix_T = w/np.sum(w)

    name = "{z}_{alpha}_{epsilon}_{ki}_{b}_{c}_{w}_{rule}".format(
        z=z, alpha=alpha, epsilon=eps, ki=ki, b=b, c=c, w=intensity, rule=flag)

    # transition matrix

    if (rank_mpi==0):
        np.savetxt(name + "_transParallel.csv", Tr, delimiter=",")

    # eigenvector
    if (rank_mpi==0):
        np.savetxt(name + "_fixParallel.csv", matrix_T, delimiter=",")

    # calculate avg donations

    coop_indx = coop_index(z, matrix_T, alpha, eps, ki, flag)

    return coop_indx


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('pop', type=int, help='pop size')

    parser.add_argument('alpha', type=float, help='assignment error')

    parser.add_argument('eps', type=float, help='execution error')

    parser.add_argument('ki', type=float, help='private error')

    parser.add_argument('benefit', type=int, help='benefits')

    parser.add_argument('cost', type=int, help='cost')

    parser.add_argument('intensity', type=float, help='intensity of selection')

    parser.add_argument('rule', help='Endogenous rule')

    args = parser.parse_args()

    z = args.pop

    alpha = args.alpha

    eps = args.eps

    ki = args.ki

    b = args.benefit

    c = args.cost

    w = args.intensity

    flag = args.rule

    print('population = ' + str(z))

    print('assignment error = ' + str(alpha))

    print('execution error = ' + str(eps))

    print('private error = ' + str(ki))

    print('benefit = ' + str(b))

    print('cost = ' + str(c))

    print('intensity = ' + str(w))

    print('assignment rule = ' + str(flag))

    coop = main(z, alpha, eps, ki, b, c, w, flag)

    print('cooperation index = ' + str(coop))
