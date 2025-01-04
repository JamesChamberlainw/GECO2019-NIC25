import csv 
import numpy as np
import random
import matplotlib.pyplot as plt

import copy
from collections import defaultdict


DIR = "resources/" # + "test-example-n4.txt" # loading data

def load_data(file_name):
    """
        Loads the data from the file and returns the nodes and problem_dict
    """
    weights = []
    values = []

    with open(file_name,'r') as f:
        reader = csv.reader(f, delimiter = '|')
        rows = list(reader)

        # Left this way for readability
        PROBLEM_NAME = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        KNAPSACK_DATA_TYPE = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        DIMENSION = int(rows[0][0].split(":")[1].strip())
        rows.pop(0)
        NUMBER_OF_ITEMS = int(rows[0][0].split(":")[1].strip())
        rows.pop(0)
        CAPACITY = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        SPEED_MIN = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        SPEED_MAX = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        RENTING_RATIO = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        EDGE_WEIGHT_TYPE = rows[0][0].split(":")[1].strip()
        rows.pop(0)

        problem_dict = {
            "PROBLEM_NAME": PROBLEM_NAME,
            "KNAPSACK_DATA_TYPE": KNAPSACK_DATA_TYPE,
            "DIMENSION": int(DIMENSION),
            "NUMBER_OF_ITEMS": int(NUMBER_OF_ITEMS),
            "CAPACITY": float(CAPACITY),
            "SPEED_MIN": float(SPEED_MIN),
            "SPEED_MAX": float(SPEED_MAX),
            "RENTING_RATIO": float(RENTING_RATIO),
            "EDGE_WEIGHT_TYPE": EDGE_WEIGHT_TYPE
        }

        rows.pop(0) # NODE_COORD_SECTION	(INDEX, X, Y): 
        
        nodes = []

        while rows[0][0].split()[0] != "ITEMS":
            content = []
            _content = rows.pop(0)[0].split(" ")

            for i in range(len(_content)):
                if _content[i] == "": continue
                sub_content = _content[i].split("\t")
                for j in range(len(sub_content)):
                    if sub_content[j] == "": continue
                    else: content.append(sub_content[j])    

            node = [int(content[0]), float(content[1]), float(content[2]), []]
            nodes.append(node)

        rows.pop(0) # ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):

        while rows != []:
            content = []
            _content = rows.pop(0)[0].split(" ")

            for i in range(len(_content)):
                if _content[i] == "": continue
                sub_content = _content[i].split("\t")
                for j in range(len(sub_content)):
                    if sub_content[j] == "": continue
                    else: content.append(sub_content[j])

            node_id = int(content[3])
            node_content = [int(content[1]), int(content[2]), int(content[0])] # profit, weight, bag_id

            nodes[node_id-1][3].append(node_content)
            
        return nodes, problem_dict

class Utils:
    """
        Utils class

        Distance Calculation
        Profit Calculation
        Weight Calculation

        and other utility functions such as: can_visit(), can_take() etc.
    """

    problem_dict = {}
    nodes = []

    def __init__(self, nodes, problem_dict):
        self.nodes = nodes
        self.problem_dict = problem_dict
        pass

    def __has_duplicates__(self, individual):
        """
            Checks if the list has duplicates
        """
        if len(individual) <= 1:
            # Too short to have duplicates
            return False
        
        # collect the node ids
        list = [x[0] for x in individual]

        if len(list) == len(set(list)):
            return False
        else:
            return True
    
    def get_weight(self, route):
        """
            Calculates the weight of the route

            Input: List of lists
                [[node_id, [bool, bool, ...]], ...]

            Output: int
        """

        weight = 0.0

        for node in route:
            ws = node[1]

            if len(self.nodes[node[0]-1][3]) == 0: continue

            for i in range(len(ws)):
                weight += ws[i] * self.nodes[node[0]-1][3][i][1]

        return weight
    
    def get_max_weight(self):
        """
            returns max weight / capacity 
        """
        return problem_dict["CAPACITY"]

    def get_distance(self, node1, node2):
        """
            Calculates the distance between two nodes

            Input: List of lists
                [node_id, x, y, [profit, weight, bag_id], [profit, weight, bag_id], ...]
        """
        return ((node1[1] - node2[1])**2 + (node1[2] - node2[2])**2)**0.5
    
    def get_profit(self, route):
        """
            Calculates the profit of the route

            Input: List of lists
                [[node_id, [bool, bool, ...]], ...]

            Output: int
        """

        profit = 0.0

        for node in route:
            vs = node[1]

            if len(self.nodes[node[0]-1][3]) == 0: continue

            for i in range(len(vs)):
                profit += vs[i] * self.nodes[node[0]-1][3][i][0]

        return profit
    

    def is_valid_individual(self, individual):
        """
            Checks if the individual is valid
        """
        if self.__has_duplicates__(individual):
            return 2
        
        if self.get_weight(individual) > self.problem_dict["CAPACITY"]:
            return 1
        
        return 0
    
    def __calc_velocity__(self, weight):
        """
            Calculates the velocity of the individual
        """

        if weight <= 0:
            return self.problem_dict["SPEED_MAX"]

        W_MAX = self.problem_dict["SPEED_MAX"]
        W_MIN = self.problem_dict["SPEED_MIN"]

        velocity = W_MAX - (weight / self.problem_dict["CAPACITY"]) * (W_MAX - W_MIN)

        if velocity < W_MIN or weight >= self.problem_dict["CAPACITY"]:
            velocity = W_MIN

        return velocity
    
    def get_instances_of_repeated_gene(self, individual):
        """
            Finds the instances of all repeated genes (not including the first instance)
        """

        indices_dict = defaultdict(list)

        lst = [x[0] for x in individual]

        for idx, value in enumerate(lst):
            indices_dict[value].append(idx)
        

        return [indices for indices in indices_dict.values() if len(indices) > 1]

    def fitness_calc_time(self, individual):
        """
            Calculates the fitness of the individual
        """
        
        history = []
        time = 0
        weight = 0.0
        velocity = self.problem_dict["SPEED_MAX"]

        # picks up first item at time 0 with max speed 
        history.append(individual[0])
        weight += self.get_weight([individual[0]])

        for i in range(1, len(individual)):
            distance = self.get_distance(self.nodes[individual[i-1][0]-1], self.nodes[individual[i][0]-1])
            time += distance / velocity
            weight += self.get_weight([individual[i]])
        return time
    
    def get_max_locations(self):
        """
            Returns the max number of locations a individual can visit 
        """
        return problem_dict["DIMENSION"]-1 # 0 is our fist number so 280 -> 279 for index / len()

class GA:
    """
        Genetic Algorithm class
        for GECO2019
    """

    UTIL = None

    pop = []
    pop_size = 100
    num_nodes = 0

    def __init__(self, nodes, problem_dict, pop_size = 100):
        """
            Initializes the GA with the nodes and problem_dict

            Input:
                nodes: List of lists
                    [[node_id, x, y, [profit, weight, bag_id], [profit, weight, bag_id], ...], ...]
                problem_dict: Dictionary
                    {
                        ...
                        "DIMENSION": int,              
                        "NUMBER_OF_ITEMS": int,
                        "CAPACITY": int,
                        "SPEED_MIN": float,
                        "SPEED_MAX": float,
                        "RENTING_RATIO": float,
                    }
        """
        self.pop_size = pop_size
        self.num_nodes = len(nodes)

        self.UTIL = Utils(nodes, problem_dict)

        # generate initial population
        for i in range(pop_size):
            individual = self.generate_individual()
            self.pop.append(individual)

    def generate_gene(self):
        """
            generates a single gene
        """
        gene_id = random.randint(0, self.num_nodes-1)
        return [gene_id, self.mutation_single_node_full(gene_id)]

    def generate_individual(self, retries = 3):
        """
            Generates a new individual with the given nodes

            Input: retries (int)
                Number of times to retry generating a new individual
        """
        # random number between 0 and num_nodes

        individual_id = [] #used purely to avoid duplicates and reduced complexity
        duplicate_retry = 0 # flag counter for duplicate retries

        individual = []

        incomplete = True
        while incomplete:
            # generate node/gene
            r = random.randint(0, self.num_nodes-1)

            if r not in individual_id:
                individual_id.append(r)
                # generate bags

                individual.append([r, self.mutation_single_node_full(r)])
                # TODO: remove [0] values (too lazy tonight todo it but easy fix)
            # else: retries -= 1
            if retries <= 0:
                # print("Length: ", len(individual_id))
                break
                    # [individual_id[-1], individual_bags[-1]]
            individual_validity = self.UTIL.is_valid_individual(individual)

            if individual_validity == 0: # valid node & bags
                duplicate_retry = 0 # reset duplicate retry 
            elif individual_validity == 1: # invalid node due to weight 
                individual.pop()
                retries -= 1 
            elif individual_validity == 2: # invalid node due to duplicates
                # invalid node due to duplicates
                if duplicate_retry > 15: # duplicates can occur due to chance so try again
                    # too many duplicates found
                    # print(f"Could not max out the individual capacity len: {len(individual)}")
                    break
                duplicate_retry += 1
                individual.pop()
            
        # print(self.UTIL.is_valid_individual(individual))
        # print(f"weight {self.UTIL.get_weight(individual)} / {self.UTIL.get_max_weight()}")
        return individual

    def mutation_single_node_full(self, individual_node_id):
        """
            Fully mutates a single node's selected bags in the individual

            Input: node_id (int)
                The node to mutate
        """

        bag_length = len(self.UTIL.nodes[individual_node_id][3])

        if bag_length == 1:
            return [1]
        
        if bag_length == 0:
            # print("[0] No bags available for node " + str(individual_node_id) + "!")
            return [0]
        
        # TODO: CHECK THIS FOR MORE THAN 1 BAG

        # random chance 0 or 1 for each bag (50% chance each)
        bags = [random.choice([0, 1]) for _ in range(bag_length)]

        # one bag must be selected else impossible to visit
        if sum(bags) == 0: 
            bags[random.randint(0, bag_length-1)] = 1

        return bags

    def mutation_bags(self, individual_id, gene_id):
        """
            Mutates the bags of the individuals 

            swaps the bags around to see if the solution improves 
                DOES NOT CHANGE THE VISITED NODES
        """

        individual = self.pop[individual_id]
        
        if len(individual) == 0: # too short to mutate 
            raise ValueError("Individual is length 0")

        gene = individual[gene_id]

        # choose a random bit to flit (bag to pick or or not)
        r = random.randint(0, len(gene[1])-1)

        # flip the bit 1 -> 0 or 0 -> 1
        gene[1][r] = 1 - gene[1][r]

        # one bag must be selected else impossible to visit 
        if sum(gene[1]) == 0:
            gene[1][random.randint(0, len(gene[1])-1)] = 1

        # replace the encoding
        individual[3] = gene

        print(f"Mutated: {gene}")

        return individual
    
    def mutation_new_gene(self, individual):
        """
            Mutation adds a new gene into a position in the individual 

            Note: checks need to be done after to check if this is a valid individual
        """

        # split location 
        r = random.randint(0, len(individual))
        
        # first attempt 
        new = individual[:r] + [self.generate_gene()] + individual[r:]

        # loop till a gene is found that can fit 
        while self.UTIL.__has_duplicates__(new) == True:
            new = individual[:r] + [self.generate_gene()] + individual[r:]      

        return new

    def fix_individual_validity(self, individual):
        """
            Fixes an Individual (harsh - full checks)
        """

        flag_weight = 0
        flag_repeated = 0

        # get repeated gene ids
        gene_ids = self.UTIL.get_instances_of_repeated_gene(individual)

        # if too large of an individual to have unique genes
        if len(individual) > self.UTIL.get_max_locations():
            # drop repeated first
            while len(gene_ids) > 0:
                # random gene_ids lst 
                r_x = random.randint(0, len(gene_ids)-1) # select set  
                r_y = random.randint(0, len(gene_ids[r_x])-1) # select id 
                individual.pop(gene_ids[r_x][r_y])
                
                # re-calc list 
                gene_ids = self.UTIL.get_instances_of_repeated_gene(individual)

                # if problem resolved go back to other checks 
                if len(individual) > self.UTIL.get_max_locations():
                    break
            
            # if still occurs 
            while len(individual) > self.UTIL.get_max_locations():
                # drop random genes 
                r = random.randint(0, len(individual)-1)
                individual.pop(r)
        

        # repeated gene dropping (random from remaining)
        while len(gene_ids) > 0:
            flag_repeated = 1 
            # random gene_ids lst 
            r_x = random.randint(0, len(gene_ids)-1) # select set  
            r_y = random.randint(0, len(gene_ids[r_x])-1) # select id 
            individual.pop(gene_ids[r_x][r_y])
            
            # re-calc list 
            gene_ids = self.UTIL.get_instances_of_repeated_gene(individual)


        # weight checking / removal of excess 
        while self.UTIL.get_weight(individual) > self.UTIL.get_max_weight():
            flag_weight = 1
            # too heavy so need to drop a random gene
            individual.pop(random.randint(0, len(individual)-1))


        # if not overweight and no repeated flag hit this is completed 
        if flag_weight == 1 and flag_repeated == 0: # weights is fixed already and no repeated 
            return individual
        else:
            # repopulate with genes to add more to the bag
            new = copy.deepcopy(individual)

            # if still under weight try to add more 
            while self.UTIL.get_weight(new) < self.UTIL.get_max_weight():
                individual = copy.deepcopy(new) # replace 

                # find new gene 
                new = self.mutation_new_gene(new)
                print(f"new gene check {self.UTIL.is_valid_individual(new)}")
            
            # left due to usefulness as debug code 
            # print(f"weight%: {self.UTIL.get_weight(individual) / self.UTIL.get_max_weight()}")
            # print(len(individual))        
        # if flag_weight == 0 or flag_repeated == 1:
        #     print(f"flags: {flag_weight} : {flag_repeated}")

        return individual

    def crossover(self, parent1, parent2):
        """
            Crossover where produced child is one side parent1 and the other parent2
        
            Note: parent1 and parent2 should be selected randomly as it will always take the first half of parent1 and the second half of parent2 
            then splice them together. Plus error checking to ensure it is
        """

        # choose a random point to crossover
        r = random.randint(0, len(parent1)-1)

        # create the child
        child = parent1[:r] + parent2[r:]

        child = self.fix_individual_validity(child)

        print(self.UTIL.is_valid_individual(child))

        return child

    def selection(self):
        pass

    def gen_fitness(self):
        """
            Generates the fitness of the population
        """

        x = []
        y = []

        for individual in self.pop:
            y.append(-self.UTIL.get_profit(individual))
            x.append(self.UTIL.fitness_calc_time(individual))
        
        return x, y


def pareto_front(solutions):
    """
    Finds the Pareto front from a list of solutions.
    """
    pareto_front = []
    
    for i, sol_i in enumerate(solutions):
        is_dominated = False
        for j, sol_j in enumerate(solutions):
            if np.all(np.array(sol_j) <= np.array(sol_i)) and np.any(np.array(sol_j) < np.array(sol_i)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(sol_i)
    
    return pareto_front

def get_consecutive_pareto_fronts(solutions):
    fronts = []
    remaining_solutions = solutions[:]

    while remaining_solutions != []:
        front = pareto_front(remaining_solutions)
        remaining_solutions = [sublist for sublist in remaining_solutions if sublist not in front]
        fronts.append(front)
    
    return fronts

nodes, problem_dict = load_data(DIR + "a280-n1395.txt")

ga = GA(nodes, problem_dict)

# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)
# __gene = ga.generate_gene()
# print(__gene)

print(ga.UTIL.is_valid_individual(ga.crossover(ga.pop[0], ga.pop[1])))
# print(len(ga.pop[0]))
# print(ga.UTIL.is_valid_individual(ga.pop[0]))
# print(ga.UTIL.get_instances_of_repeated_gene(ga.pop[0]))


# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)
# ga.pop[0] = ga.mutation_bags(0, 0)

print(len(ga.pop[0]))
print(len(ga.pop[1]))

ga.pop[0] = ga.crossover(ga.pop[0], ga.pop[1])

# fitness_time = ga.UTIL.fitness_calc_time(ga.pop[0])

# print(ga.UTIL.__calc_velocity__(0))
# print(ga.UTIL.__calc_velocity__(10))

# print(ga.UTIL.__calc_velocity__(ga.UTIL.problem_dict["CAPACITY"]))
# print(ga.UTIL.__calc_velocity__(ga.UTIL.problem_dict["CAPACITY"]))

x, y = ga.gen_fitness()
x_label = "time"
y_label = "-profit"

solutions = [[xi, yi] for xi, yi in zip(x, y)]

# Get consecutive Pareto fronts
pareto_fronts = get_consecutive_pareto_fronts(solutions)

# Plot all the solutions
solutions_np = np.array(solutions)
plt.scatter(solutions_np[:, 0], solutions_np[:, 1], color='gray', label="All solutions")

# colormap with distinct colours
cmap = plt.get_cmap('tab10', len(pareto_fronts)) 

# Plot each Pareto front with a line connecting the points 
for i, front in enumerate(pareto_fronts):
    front_np = np.array(sorted(front, key=lambda x: x[0]))
    color = cmap(i)  # Get the color for the i-th front
    plt.scatter(front_np[:, 0], front_np[:, 1], label=f'Front {i+1}', color=color)
    plt.plot(front_np[:, 0], front_np[:, 1], color=color, linestyle='-', marker='o')

# Labels and title
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title('Consecutive Pareto Fronts')

# Show legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()