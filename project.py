import sys
import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# timesteps
t = 50000
q = lambda p : (1 - p)
# t axis for plotting
tintervals = np.arange(1, t + 1, 1)
#k  axis for plotting
kintervals = 0
tick_spacing_time = t/5

#network model
class Network:
    def __init__ (self):
        #current node number
        self.current = 1
        #node count
        self.n = 1.0
        #edge count
        self.m = 1.0
        #plotting counts
        self.nt = [0]
        self.mt = [0]
        #distribution counts
        self.nodesK = [1]
        self.graph = {1: [1]}

    def addNode(self, target):
        #processes possible empty graph
        if self.n != 0:
            self.graph[self.current+1] = [target]
            targetL = len(self.graph[target])

            if targetL != 0:
                self.nodesK[targetL - 1] -= 1

            if targetL == len(self.nodesK):
                self.nodesK.append(1)
            else:
                self.nodesK[targetL] += 1

            self.graph[target].append(self.current+1)
        else:
            self.graph[target] = [target]

        self.n += 1
        self.m += 1
        self.nodesK[0] += 1
        self.current += 1

    def deleteNode(self, node):
        if node == 0:
            return

        targets = self.graph[node]

        if len(targets) != 0:
            self.nodesK[len(targets) - 1] -= 1
            
        for target in targets:
            if node != target:
                l = len(self.graph[target])
                if l - 1 == 0:
                    self.nodesK[0] -= 1
                    self.graph[target].remove(node)
                    self.m -= 1
                else:
                    self.nodesK[l - 1] -= 1
                    self.nodesK[l - 2] += 1
                    self.graph[target].remove(node)
                    self.m -= 1
            else:
                self.m -= 1
        del self.graph[node]
        self.n -= 1

    def updateStats(self):
        self.nt.append(self.n)
        self.mt.append(self.m)

#numerical solutions       
class Numerical: 
    def __init__ (self):
        self.nt = [] * t
        self.mt = [] * t
        self.nodesk = []

    def numNodes(self, p, t):
        res = (p - q(p)) * t + 2 * q(p)
        self.nt.append(res)

    def numEdges(self, p, t):
        res = p * t * (p - q(p))
        self.mt.append(res)
    
    def dist(self, p, k, n):
        res = pow(k, (-1) - (2 * p / (2 * p - 1)))
        self.nodesk.append(res)

def main():
    #plots for simulation results
    setSimNodes = []
    setSimEdges = []
    #plots for numerical results
    setNumNodes = []
    setNumEdges = []

    #distribution plot data
    numDistModel = []
    numDist = []

    funcs = [logarithmic, linear, nlogarithm]
    kintervals = []

    #allows for repition of simulation
    reps = 1

    #node and edge data testing
    for func in funcs:
        results = testingCycle(func, reps)
        setSimNodes.append(results[0][0])
        setSimEdges.append(results[0][1])
        setNumNodes.append(results[1][0])
        setNumEdges.append(results [1][1])

        
    graphVars(setSimNodes, setNumNodes, 'Nodes')
    graphVars(setSimEdges, setNumEdges, 'Edges')

    count = 1
    #distribution testing
    for func in funcs:
        numDistModel, kintervals, net = runDistSimulation(func)
        numDist = runNumerical(kintervals, net.n)
        graphDist(numDistModel, numDist, kintervals, count)
        count += 1


def updateNetwork(funcs, net):
    choice = random.random()
    if choice <= (0.8): 
        birth(net, funcs)
    else: 
        death(net)

#runs numerical and simulation testing cycles
def testingCycle(funcs, reps):
    numResults = []
    simResults = []

    #creates multiple networks if there are repitions specified
    networks = map(lambda x: Network(), range(1, reps + 1))
    #placeholders for averaging
    netNodes = [0]
    netEdges = [0]
    num = Numerical()

    for t in tintervals:
        num.numNodes(0.8, t)
        num.numEdges(0.8, t)
        map(lambda net: multiSim(t, funcs, net), networks)
    numResults.append(num.nt)
    numResults.append(num.mt)
    #averages output data at each time interval for simulation
    for i in range(1, 6):
        sumNodes = 0
        sumEdges = 0
        for net in networks:
            sumNodes += net.nt[i]
            sumEdges += net.mt[i]
        netNodes.append(sumNodes/reps)
        netEdges.append(sumEdges/reps)


    simResults.append(netNodes)
    simResults.append(netEdges)
    return [simResults, numResults]


def multiSim(t, funcs, net):
    updateNetwork(funcs,net)
    if t % tick_spacing_time == 0:
                net.updateStats()


#runs model simulation for degree distribution experiment
def runDistSimulation(func):
    kintervals = []
    normalizedData = []
    net = Network()
    for t in tintervals:
        updateNetwork(func, net)
    for k in net.nodesK:
        res = k/net.n
        if res != 0:
            normalizedData.append(k/net.n)
    kintervals = np.arange(1, len(normalizedData) + 1, 1)
    return [normalizedData, kintervals, net]

#runs numerical analysis for degree distribution
def runNumerical(kintervals, n):
    num = Numerical()
    for k in kintervals:
        num.dist(0.8, k, n)
    return num.nodesk

def logarithmic(val, m):
    return math.log(len(val) + 1, 10)  / (2 * m)

def linear(val, m):
    return len(val) / (2.0 * m)

def nlogarithm(val, m):
    length = len(val)
    return len(val) * math.log(len(val) + 1, 10) / (2 * m)

def birth(net, funcs):
    nodes = []
    probs  = []
    normalized = []

    for key, val in net.graph.items():
        nodes.append(key)
        if net.m == 0:
            probs.append(1.0/len(net.graph))
        else:
            probs.append(funcs(val, net.m))

    if len(nodes) != 0:
        sumProbs = sum(probs)
        for val in probs:
            normalized.append(val / sumProbs)
        target = np.random.choice(nodes, 1, p = normalized)[0]
        net.addNode(target)
    else:
        net.addNode(net.current + 1)


def death(net):
    nodes = []
    probs  = []
    normalized = []

    for key, val in net.graph.items():
        nodes.append(key)
        probs.append(0.5)
        
    if len(nodes) != 0:
        sumProbs = sum(probs)
        for val in probs:
            normalized.append(val / sumProbs)
        node = np.random.choice(nodes, 1, p = normalized)[0]
        net.deleteNode(node)
    else:
        net.deleteNode(0)
    

def graphVars(network, num, data):
    netData = np.linspace(0, t, num = 6)
    fig, ax = plt.subplots(1,1)
    
    ax.plot(netData, network[0], 'ro', label='Distribution 1')
    ax.plot(netData, network[1], 'bs', label='Distribution 2')
    ax.plot(netData, network[2], 'g^', label='Distribution 3')
    ax.plot(tintervals, num[0], 'k', label='Analytical')
    plt.ylabel(data)
    plt.xlabel("t")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_time))
    plt.show()

def graphDist(network, num, kintervals, count):
    fig, ax = plt.subplots(1,1)

    if count == 1:
        plotLabel = 'Distribution 1'
    elif count ==2:
        plotLabel = 'Distribution 2'
    else:
        plotLabel = 'Distribution 3'

    ax.loglog(kintervals, network, 'g--', label=plotLabel)
    ax.loglog(kintervals, num, 'b', label='Analytical')
    plt.ylabel("P(k)")
    plt.xlabel("k")
    ax.legend()
    plt.show()

main()
 
