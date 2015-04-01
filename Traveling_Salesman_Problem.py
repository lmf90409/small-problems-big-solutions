#!/usr/bin/python

"""GA Data Science Course Project 2:
   Traveling Salesman Problem Main Function
   Use python Mofei_TSP.py <algorithm_number> <.tsp file> to run the program.
   And the program will print the name of the algorithm of your choice and a 
   graph shows the all the cities and the route we got from certain algorithm
   
   Input File:
   The input file should be in the format of three columns with first column 
   as number of cities, the second one as the latitude and the third the 
   longitute.
   
   All the algorithms used in this project and their calling number:
   1: Random search
   2: Nearest Neighbor
   3: Insertion Heuristic
   4: Two-opt
   5: Simulated Annealing
   6: Swarm Intelligence

   Dependencies:
   1. numpy download from: http://sourceforge.net/projects/numpy/files/NumPy/1.9.0/
   2. networkx download from: https://pypi.python.org/pypi/networkx/
   3. matplotlib download from https://github.com/matplotlib/matplotlib/downloads
   """

__author__ = 'Mofei'

import sys
import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt

filename = sys.argv[2]
F = open(filename, 'rb')
city_list = []

for line in F.readlines():
	if line[0].isdigit():
		# Create a list of cities. For each city, it's a [number, latitude, longitute] list
		city_list.append([float(item) for item in line.strip().split()])
city_number = len(city_list)

def distance(A,B):
	"Calculate the distance between two cities"
	return np.linalg.norm([B[0]-A[0],B[1]-A[1]])

distance_matrix = np.reshape([distance(A[1:],B[1:]) \
	for A in city_list for B in city_list], (city_number, city_number))

class Tour:
    "Common base class for all the route"
    def __init__(self, route, city_list):
        self.route = route
        self.city_list = city_list
        city_number = len(city_list)
    
    def distance(self):
        "The total distance of the route."
        return sum(distance(self.city_list[self.route[i-1]][1:], self.city_list[self.route[i]][1:]) for i in self.route[1:])
    
    def coordinates(self):
        return [tuple(city_list[i][1:]) for i in self.route]

# Algorithm 1: Random Search
def Random_Search_TSP(city_list):
    """Randomly generates a route based on the permutation of the total number of cities and 
       repeat the process for 100 times and choose the one with the lowest distance as the solution"""
    for i in range(100):
        present_tour = Tour(np.random.permutation(city_number), city_list)
        if i==0 or present_tour.distance() < best_tour.distance():
            best_tour = present_tour
            
    return best_tour

# Algorithm 2: Nearest Neighbor
# Lecture Note:do a quick sort based on the x coordinates and y coordinates at the beginning
def nearest_neighbor(distance_matrix, current_city, unvisited):
    """Look up the distance matrix to find the nearest city of the current one.
       (we work with the label of a city and return the label of the nearest neighbor)"""
    return unvisited[np.argmin(distance_matrix[current_city][unvisited])]

def Nearest_Neighbor_TSP_route(city_list):
    "Find the nearest unvisited city as next destination at each step and return the completed route"
    starting_city = rd.randint(0, city_number-1) #Randomly generates a starting point
    route = [starting_city]
    unvisited = range(0,city_number)
    unvisited.remove(starting_city)
    while unvisited:
        current_city = nearest_neighbor(distance_matrix, route[-1], unvisited)
        unvisited.remove(current_city)
        route.append(current_city)
    return Tour(route, city_list)

# Algorithm 3: Insertion Heuristics
def Insertion_Heuristics_TSP_route(city_list):
    """Find the nearest unvisited city from the last city in the route at each step and insert the new one 
       to the existed tour to the place with lowest total distance and return the completed tour."""
    starting_city = rd.randint(0, city_number-1) #Randomly generates a starting point
    route = [starting_city]
    unvisited = range(0,city_number)
    unvisited.remove(starting_city)
    while unvisited:
        current_city = nearest_neighbor(distance_matrix, route[-1], unvisited)
        unvisited.remove(current_city)
        distance_change = [distance_matrix[route[i],current_city]+distance_matrix[current_city, route[i+1]]-distance_matrix[route[i],route[i+1]]
                   for i in range(len(route)-1)] #Distances for all insertion except for putting the city in the last place
        distance_change.append(distance_matrix[route[-1],current_city]) #Distance of insertion the new city as the last one
        route.insert(np.argmin(distance_change)+1, current_city)
    return Tour(route, city_list)

# Algorithm 4: 2-Opt
def Two_Opt_TSP_route(city_list):
    """Call the Insertion Heuristics Method first to get a tour, then use 2-opt optimation to improve the tour.
       2-opt algorithm considers two non-adjacent edges at each step and swap the order of the edges to get a new tour.
       If the distance of the new tour is storter than the previus one, we keep it or otherwise move forward and compare
       next two pairs."""
    tour = Insertion_Heuristics_TSP_route(city_list)
    dist = tour.distance()
    flag = 0 
    while flag==0 or best_dist!= dist:
        best_dist = tour.distance()
        for i in range(city_number-3):
            for j in range(i+2, city_number-1):
                # Compare the distance of new route with the previous one 
                # i.e. whether distance(a,c)+distance(b,d) < distance(a,b)+distance(c,d)
                if distance_matrix[tour.route[i], tour.route[i+1]] + distance_matrix[tour.route[j],tour.route[j+1]] \
                > distance_matrix[tour.route[i], tour.route[j]] + distance_matrix[tour.route[i+1],tour.route[j+1]]:
                    tour.route[j], tour.route[i+1] = tour.route[i+1], tour.route[j]
        dist = tour.distance()
        flag=flag+1
    return tour

# Algorithm 5: Simulated Annealing
def Acceptance_Probability(energy, new_energy, temperature):
    "Calculate the acceptance probability"
    if new_energy < energy: # if the new solution is better, accept it
        return 1
    else: # If the new solution is worse, calculate an acceptance probability
        return np.exp((energy-new_energy) / float(temperature))

def Simulated_Annealing_TSP_route(city_list):
    """Call the Insertion Heuristics Method first to get a tour, then use simulated annealing optimation to improve the tour.
       In simulated annealing we keep a temperature variable to simulate this heating process. We initially set it high and 
       then allow it to slowly 'cool' as the algorithm runs. While this temperature variable is high the algorithm will be allowed, 
       with more frequency, to accept solutions that are worse than our current solution."""
    city_number = len(city_list)
    temp = 10000 # Set initial Temperature
    cooling_rate = 0.003 
    
    # Initialize intial solution
    current_tour = Insertion_Heuristics_TSP_route(city_list) 
    dist = current_tour.distance()
    best_tour = current_tour
    
    while temp>1: # Loop until the system is cooled
        new_tour = Insertion_Heuristics_TSP_route(city_list) # Create a new route
        pin1, pin2 = rd.randint(0, city_number-1), rd.randint(0, city_number-1) # Randomly choose two cities in the new route
        new_tour.route[pin1], new_tour.route[pin2] = new_tour.route[pin2], new_tour.route[pin1] # Swap the two cities
        
        #Get the enerny of the two solutions
        energy = current_tour.distance()
        new_energy = new_tour.distance()
        
        # Decide if we should accept the neighbor
        if Acceptance_Probability(energy, new_energy, temp) > rd.random():
            current_tour = new_tour

        # Keep track of the best solution found
        if current_tour.distance() < best_tour.distance():
                best_tour = current_tour
            
        # Cool system
        temp *= 1-cooling_rate
    
    return best_tour

# Algorithm 6: Swarm Intelligence
def Swarm_Intelligence_TSP_route(city_list):
    """Call the Insertion Heuristics Method first to get a tour, then use ant colony optimation to improve the route.
       Using a positive feedback mechanism based on an analogy with the trail laying/following behavior, to reinforce 
       to keep good solutions. Negative feedback by pheromone evaporation. Also, we introduce the elitist ants to each 
       iterations to always keep the best solution in the pool. """
    # reference: [1] Inspiration for optimization from social insect behaviour, E. Bonabeau, M. Dorigo & G. Theraulaz
    #            [2] Introduction to Multi-Agent Programming, Alexander Kleiner, Bernhard Nebel

    city_number = len(city_list)
    m = city_number # number of ants, set to be the same as the number of cities
    alpha, beta = 1, 5 # two parameters govern the respective influences of pheromone and distance(see[1])
    rho = 0.5 # evaporation rate (see[1])
    t_max = 20 # number of update iterations
    
    # Initialize the pheromone for between every pair of cities to be 10^(-6) [from reference[1]]
    pheromone_matrix = np.ones(shape=(city_number, city_number))*10**(-6)
    
    # Q is a parameter which should be set close to the optimal tour length(see[2])
    Q = 100
    e_number = (city_number*0.1)
    
    for generation in xrange(t_max):
        tours_of_generation = [] 
        flag = 0
        for k in xrange(m):
            current_city = rd.randint(0, city_number-1) #Randomly generates a starting point
            route = [current_city]
            unvisited = range(0,city_number)
            unvisited.remove(current_city)
    
            while unvisited:
                # The pheromone accumulation from the current city to all the other unvisited cities
                pheromone_next = pheromone_matrix[current_city,unvisited] ** alpha 

                # The distance from the current city to all the other unvisited cities
                distance_next = distance_matrix[current_city,unvisited] ** -beta

                # One ant hops from one city to another with the probability of a multinomial distribution
                # np.random.multinomial(n, pvals, size) function return an array of zero except for one 1, 
                # use argmax to return the index of the only 1 in the unvisited list
                current_city = unvisited[np.random.multinomial(1, (pheromone_next*distance_next)/np.inner(pheromone_next, distance_next),
                                                               size=1).argmax()]
                unvisited.remove(current_city)
                route.append(current_city)
            this_tour = Tour(route, city_list)
            if generation == 0 and flag == 0 :
                elitist_tour = this_tour
                flag = 1 #turn it off 
            if this_tour.distance() < elitist_tour.distance():
                elitist_tour = this_tour


            tours_of_generation.append(Tour(route, city_list)) # Keep track of the tours of all the ants together of one generation

        # When every ant of on generation has completed a tour, update pheromone_matrix
        delta_pheromone_matrix = np.zeros(shape=(city_number, city_number))
        for k in xrange(m):
            for i in xrange(city_number-1):
                delta_pheromone_matrix[tours_of_generation[k].route[i], tours_of_generation[k].route[i+1]] += \
                Q/tours_of_generation[k].distance()

        # Add e_number of elitist ants to the pheromone deposition
        for i in xrange(city_number-1):
            delta_pheromone_matrix[elitist_tour.route[i], elitist_tour.route[i+1]] += \
            Q/elitist_tour.distance() * e_number

        pheromone_matrix = (1-rho)*pheromone_matrix + delta_pheromone_matrix
        print "It's generation "+str(generation)+", and the shortest distance so far is " + str(elitist_tour.distance())


    return elitist_tour

# Networkx Graph
def TSP_graph(tour):
    G=nx.Graph()
    # Add nodes
    ind = 0
    for i in tour.route:
        G.add_node(ind, pos = tour.coordinates()[i])
        ind += 1
    # Add edges
    edges = [(tour.route[i],tour.route[i+1]) for i in range(0, city_number-1)]
    G.add_edges_from(edges)
    # Set the position of the nodes
    pos=nx.get_node_attributes(G,'pos')
    # Set the attributes of the graph
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    font = {'fontname'   : 'Helvetica',
            'color'      : 'k',
            'fontweight' : 'bold',
            'fontsize'   : 15}
    plt.title("Traveling Salesman Problem for "+str(city_number)+" Cities", font)
    plt.text(0.5, 0.95,'Total Distance of the Tour:'+str(round(tour.distance(), 2)), \
         ha='center', va='center', size=12, transform=ax.transAxes)
    plt.text(0.5, 0.1,'The yellow node is the starting point', \
         ha='center', va='center', size=12, transform=ax.transAxes)
    plt.axis('off')
    colors = ['#FFFF00']+ list(np.repeat('#FF4000', city_number-1))
    nx.draw(G,pos, node_color=colors, node_size=100, edge_color='#00BFFF', width=5, with_labels=False)
    plt.show()

# Call the different Algorithm according to the input number
def callRS():
	TSP_graph(Random_Search_TSP(city_list))

def callNN():
	TSP_graph(Nearest_Neighbor_TSP_route(city_list))

def callIH():
	TSP_graph(Insertion_Heuristics_TSP_route(city_list))
    
def callTO():
	TSP_graph(Two_Opt_TSP_route(city_list))
    
def callSA():
	TSP_graph(Simulated_Annealing_TSP_route(city_list))

def callSI():
  TSP_graph(Swarm_Intelligence_TSP_route(city_list))

# Main Function
def main(argv):
	TSP_method = int(sys.argv[1])

	method_dictionary = {1: 'Random search', 
	2: 'Nearest Neighbor', 
	3: 'Insertion Heuristic', 
	4: 'Two-opt', 
	5: 'Simulated Annealing',
    6: 'Swarm Intelligence'}

	options = {1: callRS, 2: callNN, 3: callIH, 4: callTO, 5: callSA, 6:callSI} 

	print 'With the '+ str(method_dictionary[TSP_method]) + ' Method, the result is:'
	options[TSP_method]()   
      	 
if __name__ == "__main__":
	main(sys.argv[1:])
# Lecture Note:
#if len(sys.argv) != 3:
#  print "Usage: Mofei_TSP.py <#> <.tsp file>, where # is the algorithm of choice\n1 = random search\n2 =n"

#Use Hash Table or Dictionary 
